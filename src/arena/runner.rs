//! The runner: play one [`Contender`] through one [`GameSetup`], deterministically.

use crate::arena::outcome::{ClearCounts, GameOutcome, Termination};
use crate::arena::{Contender, GameSetup};
use crate::engine::{Engine, EngineEvent};
use crate::player::drive_engine;

/// Derive the controller's RNG seed from the game seed.
///
/// Kept distinct from the engine's seed (so the bot's error-injection / tie-break
/// stream doesn't correlate with the piece sequence) but fully determined by it,
/// so a game is reproducible from a single `seed`. The constant is the golden-ratio
/// odd integer used by SplitMix64.
fn controller_seed(seed: u64) -> u64 {
    seed ^ 0x9E37_79B9_7F4A_7C15
}

/// Play `contender` through `setup` with the given engine `seed`, returning the
/// measured [`GameOutcome`].
///
/// Deterministic: the result is a pure function of `(contender, setup, seed)` —
/// the engine and controller carry the no-clock / seeded-RNG contract, and this
/// runner adds no entropy. Always terminates: on top-out, on reaching the piece
/// budget, or on the hard frame cap (whichever comes first), recording which.
///
/// # Panics
///
/// If the event-tallied line count disagrees with the engine's final snapshot —
/// a self-check that turns any accounting bug (here or in the engine) into a loud
/// failure rather than a silently-wrong measurement.
pub fn play(contender: &Contender, setup: &GameSetup, seed: u64) -> GameOutcome {
    let mut engine = Engine::new(setup.config().clone(), seed);
    let mut controller = contender.build(controller_seed(seed));

    let mut acc = Accumulator::default();
    let mut termination = Termination::HitFrameCap;
    let mut frames = 0u32;

    for _ in 0..setup.max_frames() {
        frames += 1;
        for event in drive_engine(&mut engine, &mut *controller) {
            acc.record(&event);
        }
        if acc.game_over {
            termination = Termination::ToppedOut;
            break;
        }
        if acc.pieces as usize >= setup.max_pieces() {
            termination = Termination::ReachedPieceBudget;
            break;
        }
    }

    let snapshot = engine.snapshot();

    // Reliability cross-check: the per-lock line counts we summed from the event
    // stream must equal the engine's own authoritative line total. A mismatch is a
    // bug (here or in the engine) and must never pass silently.
    assert_eq!(
        acc.lines, snapshot.lines as u32,
        "arena line accounting ({}) diverged from the engine snapshot ({})",
        acc.lines, snapshot.lines,
    );

    GameOutcome {
        seed,
        pieces_placed: acc.pieces,
        lines_cleared: snapshot.lines as u32,
        clears: acc.clears,
        back_to_back_awards: acc.b2b_awards,
        final_score: snapshot.score as u32,
        final_level: snapshot.level,
        frames,
        termination,
    }
}

/// Running tallies derived from the engine event stream during a game.
#[derive(Default)]
struct Accumulator {
    pieces: u32,
    lines: u32,
    clears: ClearCounts,
    b2b_awards: u32,
    game_over: bool,
}

impl Accumulator {
    fn record(&mut self, event: &EngineEvent) {
        match event {
            // One Locked event per placed piece, carrying that lock's line count.
            EngineEvent::Locked { lines_cleared, .. } => {
                self.pieces += 1;
                self.lines += *lines_cleared as u32;
            }
            // The clear *type* and B2B bonus come from the scoring action.
            EngineEvent::ScoreAwarded {
                action,
                back_to_back_bonus,
                ..
            } => {
                self.clears.record(action);
                if *back_to_back_bonus {
                    self.b2b_awards += 1;
                }
            }
            EngineEvent::GameOver { .. } => self.game_over = true,
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{EngineConfig, EngineSnapshot, InputFrame};
    use crate::player::PlayerController;

    /// Always hard-drops in place: stacks pieces in the spawn columns and tops out
    /// quickly. Deterministic and stateless — a clean baseline for the harness.
    struct HardDrop;
    impl PlayerController for HardDrop {
        fn poll(&mut self, _snapshot: &EngineSnapshot) -> InputFrame {
            InputFrame {
                hard_drop: true,
                dt_seconds: 1.0 / 60.0,
                ..InputFrame::default()
            }
        }
    }

    /// Never presses anything: pieces fall under gravity but the controller makes
    /// no decisions. Used to exercise the frame cap.
    struct Idle;
    impl PlayerController for Idle {
        fn poll(&mut self, _snapshot: &EngineSnapshot) -> InputFrame {
            InputFrame {
                dt_seconds: 1.0 / 60.0,
                ..InputFrame::default()
            }
        }
    }

    fn hard_drop() -> Contender {
        Contender::new("hard-drop", |_seed| Box::new(HardDrop))
    }

    #[test]
    fn play_is_deterministic() {
        let setup = GameSetup::standard("standard", 30);
        let a = play(&hard_drop(), &setup, 12345);
        let b = play(&hard_drop(), &setup, 12345);
        assert_eq!(a, b);
    }

    #[test]
    fn hard_drop_in_place_tops_out() {
        // A generous budget the stack-in-place controller can't reach: it buries
        // itself in the spawn columns first.
        let setup = GameSetup::standard("standard", 1_000);
        let outcome = play(&hard_drop(), &setup, 7);
        assert_eq!(outcome.termination, Termination::ToppedOut);
        assert!(outcome.topped_out());
        assert!(outcome.pieces_placed < 1_000);
    }

    #[test]
    fn small_budget_is_reached_before_topping_out() {
        // Three pieces stack only a few rows high — nowhere near a top-out.
        let setup = GameSetup::standard("standard", 3);
        let outcome = play(&hard_drop(), &setup, 7);
        assert_eq!(outcome.termination, Termination::ReachedPieceBudget);
        assert_eq!(outcome.pieces_placed, 3);
    }

    #[test]
    fn frame_cap_is_respected() {
        // Two frames is far too few for the idle controller to lock a piece, so
        // neither the budget nor a top-out is reached first.
        let setup = GameSetup::standard("standard", 1_000).with_frame_cap(2);
        let outcome = play(&Contender::new("idle", |_seed| Box::new(Idle)), &setup, 7);
        assert_eq!(outcome.termination, Termination::HitFrameCap);
        assert_eq!(outcome.frames, 2);
    }

    #[test]
    fn outcome_reconciles_with_the_engine_snapshot() {
        // The cross-check inside `play` already asserts this; here we exercise it
        // end-to-end on a real game and confirm a non-trivial board state.
        let setup = GameSetup::new("wide", EngineConfig::default(), 50);
        let outcome = play(&hard_drop(), &setup, 99);
        // The harness ran a real game (frames advanced, pieces placed) and the
        // reconciliation assert inside `play` passed.
        assert!(outcome.frames > 0);
        assert!(outcome.pieces_placed > 0);
    }
}
