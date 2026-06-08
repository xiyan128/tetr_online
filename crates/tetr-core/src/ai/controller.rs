//! The AI player controller (AI3.5): the model-agnostic shell.
//!
//! [`AiController`] is a [`PlayerController`]: it reads an [`EngineSnapshot`] and
//! returns the next [`InputFrame`], exactly like the keyboard controller, so it
//! drops into the same engine-driver seam (see `src/level/mod.rs`).
//!
//! It is the *shell* around an AI brain, and it is deliberately **model-agnostic**:
//! it knows nothing about search, evaluators, or weights — only a
//! [`Policy`](crate::ai::Policy), driven through a [`DecisionRunner`]. The same
//! shell drives a greedy search, a future beam, or a neural policy unchanged. It
//! owns the two concerns a brain has no opinion on:
//!
//! - **When** to act — a [`Handicap::reaction`] delay before playing a new piece.
//! - **How** to act — rendering the policy's chosen placement to engine input, one
//!   pulse per poll.
//!
//! The *other* half of the handicap — how *perfectly* to play — lives in the policy,
//! not here (a search softmax-samples a near-best, a net raises its temperature).
//! Keeping it there is what lets this shell stay model-blind.
//!
//! # One pulse per poll
//!
//! The engine consumes at most one action per `step` (a hold, a rotation, one
//! lateral cell, a soft/hard drop). So the controller never returns a compound
//! frame: it renders the chosen placement to a `Vec<InputFrame>` (via
//! [`placement_to_inputs`]) once per piece and **drains it one frame per poll**,
//! emitting a neutral frame (`dt` only) when the queue is empty or it is still
//! "reacting". This matches how the keyboard controller feeds the engine.
//!
//! # The poll state machine
//!
//! Each [`poll`](AiController::poll):
//! 1. **Detect a new piece.** If the active piece's identity changed (or the board
//!    did under the same piece), the current plan is stale: cancel the in-flight
//!    decision, clear the queued frames, and start a fresh think.
//! 2. **React.** Accumulate the poll's `dt` into a reaction timer; while it is below
//!    [`Handicap::reaction`], emit neutral frames. The decision is *submitted* at
//!    the start of the think (so it can compute off-thread during the delay) but not
//!    *applied* until the delay elapses.
//! 3. **Apply.** Once the reaction elapses and the runner has a [`Decision`], render
//!    it to frames and enqueue them.
//! 4. **Emit.** Pop the next queued frame, or a neutral frame if none.
//!
//! # Determinism
//!
//! The shell adds no randomness and no clock — it only integrates the poll `dt`
//! (deterministic given the poll cadence). All AI randomness lives in the policy's
//! own seeded RNG. So a fixed `(engine seed, ai seed, handicap)` reproduces an
//! identical game every run — the determinism the M2 headless benchmark relies on.
//!
//! # No Bevy here
//!
//! Like the rest of the AI core this module has **no Bevy imports**: it is a plain
//! `PlayerController`, unit-testable by stepping an [`Engine`](crate::engine::Engine)
//! against it in a loop. Only the integration layer (`src/level`) is Bevy-aware.

use core::time::Duration;
use std::collections::VecDeque;

use crate::ai::handicap::Handicap;
use crate::ai::plan::placement_to_inputs;
use crate::ai::policy::{Decision, Policy, SearchPolicy};
use crate::ai::runner::{DecisionRunner, SyncRunner};
use crate::ai::state::SearchState;
use crate::engine::{EngineSnapshot, InputFrame, PieceType};
use crate::player::PlayerController;

/// A default seed for the AI's own RNG, distinct from the engine's default seed so
/// the two streams never accidentally align.
pub const DEFAULT_AI_SEED: u64 = 0xA1_5E_ED;

/// An AI [`PlayerController`]: a model-agnostic shell that drives a
/// [`Policy`](crate::ai::Policy) (via a [`DecisionRunner`]) and feeds its chosen
/// placement to the engine one pulse per poll, with a reaction-delay handicap.
pub struct AiController {
    /// Where the decision is computed (synchronous by default; an off-thread runner
    /// drops in for a heavier policy with no other change).
    runner: Box<dyn DecisionRunner>,
    /// Frames left to emit for the current placement, front = next to play.
    plan: VecDeque<InputFrame>,
    /// Reaction delay before acting on a new piece (the shell-level handicap).
    reaction: Duration,
    /// Identity of the piece the current decision is for, to detect a new piece.
    /// `None` before the first piece is seen.
    planning_for: Option<PieceSignature>,
    /// Reaction time accumulated for the current piece (integrates poll `dt`).
    think_elapsed: f32,
    /// Whether a decision has been submitted for the current piece (so we submit
    /// once per piece, not every poll while reacting).
    submitted: bool,
}

/// A cheap fingerprint of "which piece, on which board" so the controller can tell
/// a genuinely new planning situation from re-polling the same one.
///
/// Piece *type* alone is ambiguous (the next piece can repeat across a bag
/// boundary); pairing it with the locked-cell count distinguishes "same piece,
/// same board" (keep the current decision) from "new piece spawned" or "board
/// changed under us" (re-decide). Cheap to compute from a snapshot and `Eq`.
#[derive(Clone, Copy, PartialEq, Eq)]
struct PieceSignature {
    piece_type: PieceType,
    /// Number of locked cells on the board (changes on any lock/clear).
    locked_cells: usize,
}

impl PieceSignature {
    /// Derive the signature of a snapshot's current planning situation, if it has
    /// an active piece.
    fn of(snapshot: &EngineSnapshot) -> Option<Self> {
        let piece_type = snapshot.active.as_ref()?.piece_type;
        Some(Self {
            piece_type,
            locked_cells: snapshot.board_cells.len(),
        })
    }
}

impl AiController {
    /// The shipped Tier-1 bot: a greedy [`SearchPolicy`] (default evaluator) behind
    /// a synchronous runner, with the given `handicap` and RNG `seed`.
    pub fn new(handicap: Handicap, seed: u64) -> Self {
        let policy = SearchPolicy::greedy(handicap.imperfection, seed);
        Self::with_policy(Box::new(policy), handicap.reaction)
    }

    /// A controller with the **default** handicap (a beatable opponent) and the
    /// default AI seed — the convenient construction for the game and sandbox.
    pub fn beatable() -> Self {
        Self::new(Handicap::default(), DEFAULT_AI_SEED)
    }

    /// A controller around an explicit [`Policy`], wrapped in the synchronous
    /// runner. The brain seam: pass any policy (a custom-weighted search, a future
    /// neural net) and the shell drives it. `reaction` is the shell-level handicap;
    /// the policy carries its own imperfection + RNG.
    pub fn with_policy(policy: Box<dyn Policy>, reaction: Duration) -> Self {
        Self::with_runner(Box::new(SyncRunner::new(policy)), reaction)
    }

    /// A controller around an explicit [`DecisionRunner`], for swapping in an
    /// off-thread / time-sliced runner without changing the controller logic.
    pub(crate) fn with_runner(runner: Box<dyn DecisionRunner>, reaction: Duration) -> Self {
        Self {
            runner,
            plan: VecDeque::new(),
            reaction,
            planning_for: None,
            think_elapsed: 0.0,
            submitted: false,
        }
    }

    /// Begin planning for a freshly seen piece: cancel the in-flight decision, reset
    /// the reaction timer and queued frames, and remember what we are deciding for.
    fn begin_new_piece(&mut self, signature: PieceSignature) {
        self.runner.cancel();
        self.plan.clear();
        self.think_elapsed = 0.0;
        self.submitted = false;
        self.planning_for = Some(signature);
    }

    /// Try to apply a ready decision: render the chosen placement to frames and
    /// enqueue them. Returns `true` if a decision was handled (frames enqueued or an
    /// explicit "no move"), `false` if the runner had nothing yet.
    fn try_apply_decision(&mut self, obs: &SearchState) -> bool {
        let Some(decision) = self.runner.poll() else {
            return false;
        };
        match decision {
            // No legal placement: nothing to do (the engine tops out on its own);
            // leave the plan empty so we emit neutral frames.
            Decision::None => true,
            Decision::Place(placement) => {
                // Render against the board the maneuver happens on, from the active
                // piece's current pose — the inputs `placement_to_inputs` round-trips.
                let frames = placement_to_inputs(&obs.board.to_array2d(), &obs.active, &placement);
                self.plan = frames.into();
                true
            }
        }
    }
}

impl PlayerController for AiController {
    fn poll(&mut self, snapshot: &EngineSnapshot) -> InputFrame {
        // (1) No active piece (pre-spawn / game over): nothing to decide; idle.
        let Some(signature) = PieceSignature::of(snapshot) else {
            return neutral();
        };

        // (1b) New planning situation? Reset and start a fresh think.
        if self.planning_for != Some(signature) {
            self.begin_new_piece(signature);
        }

        // If we still have queued frames for the current piece, keep draining them
        // (don't re-decide mid-maneuver). One pulse per poll. Maneuver frames carry
        // their own `dt == 0` so positioning advances no gravity — emit as-is.
        if let Some(frame) = self.plan.pop_front() {
            return frame;
        }

        // Build the observation once; reused for submit + render.
        let Some(obs) = SearchState::from_snapshot(snapshot) else {
            return neutral();
        };

        // (2) Submit the decision once per piece (so it can compute during the
        // reaction delay), then accumulate the reaction timer.
        if !self.submitted {
            self.runner.submit(obs.clone());
            self.submitted = true;
        }
        self.think_elapsed += NOMINAL_DT;
        if self.think_elapsed < self.reaction.as_secs_f32() {
            return neutral(); // still "reacting"
        }

        // (3) Reaction elapsed: apply the decision if the runner is ready, then emit
        // its first maneuver frame.
        if self.try_apply_decision(&obs) {
            if let Some(frame) = self.plan.pop_front() {
                return frame;
            }
        }

        // (4) Nothing to emit yet (decision not ready, or no legal move): idle.
        neutral()
    }
}

/// Nominal per-poll `dt` for the controller's reaction-timer integration: the fixed
/// sim slice at `SIM_HZ` (60 Hz). The driver steps the engine at that rate, so
/// integrating the reaction delay in these units paces it in real seconds. Maneuver
/// frames carry their *own* `dt == 0` (gravity-free positioning) and are emitted
/// unchanged; only neutral "reacting"/idle frames advance time.
const NOMINAL_DT: f32 = 1.0 / 60.0;

/// A neutral frame: advance one sim slice of time, press nothing. Emitted while
/// reacting or idle so gravity and the lock timer keep ticking between maneuvers.
fn neutral() -> InputFrame {
    InputFrame {
        dt_seconds: NOMINAL_DT,
        ..InputFrame::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{Engine, EngineConfig};

    /// Step an engine with the controller for up to `max_frames`, returning how
    /// many pieces locked. Stops early on game over.
    ///
    /// Steps the controller's frame **as emitted** (no `dt` override) via the same
    /// contract as [`drive_engine`](crate::player::drive_engine): maneuver frames
    /// carry `dt == 0` so gravity does not desync the placement, neutral frames
    /// carry the sim slice.
    fn play(controller: &mut AiController, seed: u64, max_frames: usize) -> (usize, usize, bool) {
        let mut engine = Engine::new(EngineConfig::default(), seed);
        let mut locks = 0usize;
        let mut topped_out = false;
        for _ in 0..max_frames {
            let snapshot = engine.snapshot();
            if snapshot.game_over.is_some() {
                topped_out = true;
                break;
            }
            // Count locks exactly via the engine's Locked events (robust to line
            // clears, which shrink the board-cell count).
            let events = engine.step(controller.poll(&snapshot));
            locks += events
                .iter()
                .filter(|e| matches!(e, crate::engine::EngineEvent::Locked { .. }))
                .count();
        }
        (locks, engine.snapshot().lines, topped_out)
    }

    #[test]
    fn controller_survives_and_clears_lines_with_default_weights() {
        // Play-QUALITY regression guard (not just "places pieces"): with the shipped
        // survival reward profile, the no-handicap bot keeps a clean board and clears
        // lines for a long run instead of stacking into a fast top-out. The old
        // Cold-Clear *downstacking* default — meant for a multi-ply beam — buried a
        // 1-ply greedy by ~piece 40-126, which this test would now catch.
        let mut controller = AiController::new(Handicap::perfect(), DEFAULT_AI_SEED);
        let (locks, lines, topped_out) = play(&mut controller, 7, 8_000);
        assert!(
            !topped_out,
            "survival weights must not top out within 8k frames (placed {locks}, cleared {lines})"
        );
        assert!(
            lines >= 40,
            "the bot should be clearing lines; cleared only {lines} (placed {locks})"
        );
    }

    #[test]
    fn controller_executes_the_planners_first_placement_faithfully() {
        // The fidelity guarantee AI3.5 owns: the controller lands its first piece
        // exactly where the *planner* intends. We plan independently from the first
        // snapshot, then drive a perfect (instant, no-error) controller until the
        // piece locks, and assert the resulting board equals the planner's
        // simulated placement. (Plan-to-input pose fidelity is separately pinned by
        // `plan.rs`'s round-trip test; this proves the controller *sequences* it.)
        use crate::ai::eval::LinearEvaluator;
        use crate::ai::search::{GreedyPlanner, Planner, PlannerStep, SearchBudget};
        use crate::ai::SearchState;
        use crate::engine::{lock_and_clear, Board, CellKind};

        let mut controller = AiController::new(Handicap::perfect(), DEFAULT_AI_SEED);
        let mut engine = Engine::new(EngineConfig::default(), 7);
        engine.step(InputFrame::default()); // spawn the first piece
        let snapshot = engine.snapshot();
        let config = snapshot.config.clone();

        // Planner's intended board after placing the first piece (hold-aware, like
        // the controller's own policy).
        let state = SearchState::from_snapshot(&snapshot).unwrap();
        let plan = match GreedyPlanner::new().plan(
            &state,
            &LinearEvaluator::default(),
            SearchBudget::greedy(),
        ) {
            PlannerStep::Done(Some(plan)) => plan,
            other => panic!("expected a plan, got {other:?}"),
        };
        let mut intended = state.board.to_array2d();
        lock_and_clear(&plan.placement.piece, &mut intended);

        // Drive the controller until the board first changes (the piece locked).
        let mut guard = 0;
        loop {
            engine.step(controller.poll(&engine.snapshot()));
            let after = engine.snapshot();
            if !after.board_cells.is_empty() {
                let mut real = Board::with_top_margin(
                    config.board_width,
                    config.visible_height,
                    config.buffer_height,
                );
                for c in &after.board_cells {
                    real.set(c.x, c.y, CellKind::Some(c.piece_type));
                }
                assert_eq!(
                    real.cell_coords(),
                    intended.cell_coords(),
                    "controller-executed placement diverged from the planner's intent"
                );
                break;
            }
            guard += 1;
            assert!(guard < 300, "the first piece never locked");
        }
    }

    #[test]
    fn neutral_frame_while_reacting() {
        // With a long reaction delay, the very first polls emit neutral frames (no
        // button pressed) — the bot is "reacting" before it acts.
        let handicap = Handicap {
            reaction: core::time::Duration::from_millis(500),
            ..Handicap::perfect()
        };
        let mut controller = AiController::new(handicap, DEFAULT_AI_SEED);
        let mut engine = Engine::new(EngineConfig::default(), 1);
        engine.step(InputFrame::default()); // spawn the first piece
        let snapshot = engine.snapshot();

        let frame = controller.poll(&snapshot);
        assert!(
            !pressed_anything(&frame),
            "first poll under a long reaction delay must be neutral"
        );
        assert!(frame.dt_seconds > 0.0, "neutral frames still advance time");
    }

    #[test]
    fn determinism_same_seed_same_game() {
        // Two controllers with the same AI seed, driving two engines with the same
        // engine seed, must produce byte-identical games — the determinism the
        // headless benchmark relies on. Uses the default (imperfect) handicap so the
        // RNG path is actually exercised.
        let play_to_snapshot = |ai_seed: u64| {
            let mut controller = AiController::new(Handicap::default(), ai_seed);
            let mut engine = Engine::new(EngineConfig::default(), 42);
            for _ in 0..2_000 {
                let snap = engine.snapshot();
                if snap.game_over.is_some() {
                    break;
                }
                engine.step(controller.poll(&snap));
            }
            engine.snapshot()
        };

        let a = play_to_snapshot(DEFAULT_AI_SEED);
        let b = play_to_snapshot(DEFAULT_AI_SEED);
        assert_eq!(a, b, "same (engine, ai) seed must reproduce the game");
    }

    #[test]
    fn imperfection_uses_only_the_ai_rng() {
        // A controller with a high imperfection must still be fully reproducible from
        // its seed (RNG is owned + seeded, never the engine's). Different AI seeds
        // may diverge; the SAME seed must not.
        let high_error = Handicap {
            imperfection: 0.9,
            ..Handicap::perfect()
        };
        let run = |seed: u64| {
            let mut c = AiController::new(high_error, seed);
            let mut engine = Engine::new(EngineConfig::default(), 7);
            for _ in 0..1_500 {
                let snap = engine.snapshot();
                if snap.game_over.is_some() {
                    break;
                }
                engine.step(c.poll(&snap));
            }
            engine.snapshot().board_cells.len()
        };
        assert_eq!(
            run(123),
            run(123),
            "same AI seed reproduces even with errors"
        );
    }

    fn pressed_anything(frame: &InputFrame) -> bool {
        frame.left
            || frame.right
            || frame.soft_drop
            || frame.hard_drop
            || frame.rotate_clockwise
            || frame.rotate_counterclockwise
            || frame.hold
    }
}
