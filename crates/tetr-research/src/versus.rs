//! The **versus** harness: head-to-head matches under the engine's garbage
//! rules. Both bots play the identical piece sequence and a player loses by
//! topping out; the garbage RULES (cancellation, rising after clear-less
//! locks, the per-lock cap, hole choice) are the ENGINE's — see tetr-core's
//! garbage module — and this driver only routes each side's net `AttackSent`
//! to the other side's pending queue.
//!
//! This is Cold Clear 2's home turf — the metric that actually decides "beat
//! CC2", as opposed to one-sided downstacking. The legacy harness-side
//! garbage scheduler (scripted pressure + the TBP referee's external
//! bookkeeping) lives in [`crate::versus_legacy`], deliberately quarantined.

use rayon::prelude::*;
use tetr_core::engine::{Engine, EngineEvent, EngineSnapshot, InputFrame};
use tetr_core::player::{PlayerController, drive_engine};

use crate::accounting::controller_seed;
use crate::marathon::marathon_config;

/// Frames a single piece may take before we treat the bot as wedged (~4.3s at 60 Hz
/// — far beyond any real per-piece search, so only a genuinely stuck bot trips it).
pub(crate) const MAX_PIECE_FRAMES: u32 = 256;

/// Drive one player's bot until it locks a single piece (or tops out / stalls).
/// Returns `(net attack sent by that placement, topped_out)`.
///
/// Attack accounting is the **engine's**: [`EngineEvent::AttackSent`] already
/// carries the post-cancellation net (the engine offsets its own pending queue
/// at lock time), and pending garbage rises by the engine's guideline timing —
/// after a clear-less lock, capped per lock. The caller's only job is routing
/// the net attack to the opponent's queue. (When nothing was ever queued the
/// pending queue is empty and net == gross — which is how the TBP referee path
/// keeps its own external bookkeeping.)
pub(crate) fn versus_step_piece(
    engine: &mut Engine,
    bot: &mut dyn PlayerController,
) -> (u32, bool) {
    let mut attack = 0u32;
    let mut topped = false;
    for _ in 0..MAX_PIECE_FRAMES {
        let mut locked = false;
        for event in drive_engine(engine, bot) {
            match &event {
                EngineEvent::AttackSent { lines } => attack += lines,
                EngineEvent::Locked { .. } => locked = true,
                EngineEvent::GameOver { .. } => topped = true,
                _ => {}
            }
        }
        if topped || locked {
            break;
        }
    }
    (attack, topped)
}

/// Result of a single versus match (A = first bot, B = second).
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub enum VersusResult {
    AWins,
    BWins,
    Draw,
}

/// Decide a versus match from each side's final state. A topout loses outright — a
/// death-loss takes **priority** over attack count; otherwise the larger attack total
/// wins and equal totals draw. Shared by [`play_versus`] and the CC2 referee harness
/// (`cc2-baseline`) so both score matches identically. `A` is the first side, `B` the
/// second (e.g. in the CC2 harness A = ours, B = CC2).
pub fn decide_versus(a_topped: bool, b_topped: bool, a_attack: u32, b_attack: u32) -> VersusResult {
    use std::cmp::Ordering;
    match (a_topped, b_topped) {
        (true, false) => VersusResult::BWins, // A died
        (false, true) => VersusResult::AWins, // B died
        // Both alive, or both dead the same tick: the bigger attacker wins, ties draw.
        _ => match a_attack.cmp(&b_attack) {
            Ordering::Greater => VersusResult::AWins,
            Ordering::Less => VersusResult::BWins,
            Ordering::Equal => VersusResult::Draw,
        },
    }
}

#[derive(Debug, Clone, Copy, serde::Serialize)]
pub struct VersusOutcome {
    pub seed: u64,
    pub result: VersusResult,
    pub plies: u32,
    pub attack_a: u32,
    pub attack_b: u32,
    pub a_topped: bool,
    pub b_topped: bool,
}

/// Play one versus match between bot A and bot B. Both face the identical piece
/// sequence (same engine seed), so the result reflects decision quality, not piece
/// luck. A player loses by topping out; if the ply cap is reached with both alive,
/// the higher total attack wins.
///
/// "Attack" here is **net** attack — the post-cancellation spillover that actually
/// lands on the opponent's queue (the standard "garbage sent" notion), not gross
/// lines generated. Lines spent cancelling your own incoming queue count for
/// survival but not for this tiebreaker, so a pure digging bot under pressure can
/// record zero attack while playing well.
pub fn play_versus(
    make_a: &dyn Fn(u64) -> Box<dyn PlayerController>,
    make_b: &dyn Fn(u64) -> Box<dyn PlayerController>,
    seed: u64,
    max_plies: u32,
) -> VersusOutcome {
    play_versus_format(
        make_a,
        make_b,
        seed,
        VersusFormat {
            max_plies,
            rain_period: 0,
        },
    )
}

/// Match-format knobs for [`play_versus_format`].
#[derive(Clone, Copy, Debug)]
pub struct VersusFormat {
    /// Ply cap; a capped game falls back to the net-attack tiebreak.
    pub max_plies: u32,
    /// Environmental "rain": every `rain_period` plies (`0` = off) BOTH sides
    /// get one garbage line queued through the normal rules (cancellable,
    /// capped rising). Mirror matches between strong bots almost never kill
    /// (measured ≤6% decisive even attack-tuned at 400 plies), which starves
    /// every survival-sensitive objective; symmetric rain forces matches
    /// decisive while staying fair — same-seeded engines even draw identical
    /// hole columns for the same rain batch.
    pub rain_period: u32,
}

/// [`play_versus`] under an explicit [`VersusFormat`] (rain, ply cap).
pub fn play_versus_format(
    make_a: &dyn Fn(u64) -> Box<dyn PlayerController>,
    make_b: &dyn Fn(u64) -> Box<dyn PlayerController>,
    seed: u64,
    format: VersusFormat,
) -> VersusOutcome {
    let max_plies = format.max_plies;
    // Level rises but never ends the game here (only top-out / the cap do).
    // The versus rules — cancellation, rising after clear-less locks, the
    // garbage cap, hole choice — are the ENGINE's (see tetr-core's garbage
    // module); this driver only routes each side's net attack to the other
    // side's pending queue.
    let mut a_engine = Engine::new(marathon_config(), seed);
    let mut b_engine = Engine::new(marathon_config(), seed);
    let mut a_bot = make_a(controller_seed(seed));
    let mut b_bot = make_b(controller_seed(seed));
    let (mut a_attack, mut b_attack) = (0u32, 0u32);
    let (mut a_topped, mut b_topped) = (false, false);
    let mut plies = 0u32;

    'match_loop: for ply in 0..max_plies {
        // Environmental rain (see [`VersusFormat::rain_period`]): symmetric
        // queued pressure, before either side moves this ply.
        if format.rain_period > 0 && ply % format.rain_period == format.rain_period - 1 {
            a_engine.queue_garbage(1);
            b_engine.queue_garbage(1);
        }
        // Alternate first mover so neither side gets a structural send-first edge.
        let order = if ply % 2 == 0 { [0u8, 1] } else { [1, 0] };
        for &who in &order {
            plies += 1;
            // Route the attack BEFORE checking death: the engine already
            // encodes the rule (a lock-out lock emits no AttackSent), so any
            // attack that WAS emitted — e.g. a real clear whose next spawn
            // block-outs — legitimately left the board and must reach the
            // opponent's queue and the stats. The driver never second-guesses
            // the event stream.
            if who == 0 {
                let (atk, topped) = versus_step_piece(&mut a_engine, &mut *a_bot);
                if atk > 0 {
                    b_engine.queue_garbage(atk);
                    a_attack += atk;
                }
                if topped {
                    a_topped = true;
                    break 'match_loop;
                }
            } else {
                let (atk, topped) = versus_step_piece(&mut b_engine, &mut *b_bot);
                if atk > 0 {
                    a_engine.queue_garbage(atk);
                    b_attack += atk;
                }
                if topped {
                    b_topped = true;
                    break 'match_loop;
                }
            }
        }
    }

    let result = decide_versus(a_topped, b_topped, a_attack, b_attack);

    VersusOutcome {
        seed,
        result,
        plies,
        attack_a: a_attack,
        attack_b: b_attack,
        a_topped,
        b_topped,
    }
}

/// Aggregate versus stats over a seed set.
#[derive(Debug, Clone)]
pub struct VersusStats {
    pub games: usize,
    pub a_wins: usize,
    pub b_wins: usize,
    pub draws: usize,
    pub mean_attack_a: f32,
    pub mean_attack_b: f32,
    pub outcomes: Vec<VersusOutcome>,
}

impl VersusStats {
    pub fn a_win_rate(&self) -> f32 {
        self.a_wins as f32 / self.games.max(1) as f32
    }
}

/// Evaluate bot A vs bot B over `seeds`.
pub fn evaluate_versus(
    make_a: &(dyn Fn(u64) -> Box<dyn PlayerController> + Sync),
    make_b: &(dyn Fn(u64) -> Box<dyn PlayerController> + Sync),
    seeds: &[u64],
    max_plies: u32,
) -> VersusStats {
    evaluate_versus_format(
        make_a,
        make_b,
        seeds,
        VersusFormat {
            max_plies,
            rain_period: 0,
        },
    )
}

/// [`evaluate_versus`] under an explicit [`VersusFormat`].
pub fn evaluate_versus_format(
    make_a: &(dyn Fn(u64) -> Box<dyn PlayerController> + Sync),
    make_b: &(dyn Fn(u64) -> Box<dyn PlayerController> + Sync),
    seeds: &[u64],
    format: VersusFormat,
) -> VersusStats {
    // One thread per match (rayon); collection is order-stable, and each match
    // is a pure function of its seed, so the parallel stats are bit-identical
    // to the sequential ones (pinned by `parallel_evaluation_matches_sequential`).
    let outcomes: Vec<VersusOutcome> = seeds
        .par_iter()
        .map(|&seed| play_versus_format(make_a, make_b, seed, format))
        .collect();
    let a_wins = outcomes
        .iter()
        .filter(|o| o.result == VersusResult::AWins)
        .count();
    let b_wins = outcomes
        .iter()
        .filter(|o| o.result == VersusResult::BWins)
        .count();
    let draws = outcomes.len() - a_wins - b_wins;
    let n = outcomes.len().max(1) as f32;
    let mean_attack_a = outcomes.iter().map(|o| o.attack_a as f32).sum::<f32>() / n;
    let mean_attack_b = outcomes.iter().map(|o| o.attack_b as f32).sum::<f32>() / n;
    VersusStats {
        games: outcomes.len(),
        a_wins,
        b_wins,
        draws,
        mean_attack_a,
        mean_attack_b,
        outcomes,
    }
}

/// A controller wrapper that hides the pending-garbage queue from its inner
/// bot: the snapshot it forwards has `pending_garbage` emptied, so the bot
/// plans as if no attack were queued — the *blind* arm of the
/// garbage-awareness A/B. Everything else (weights, search, venue, seeds) stays
/// identical, so a win-rate gap between a wrapped and an unwrapped copy of the
/// same bot measures exactly the value of seeing (and modeling) the queue.
pub struct BlindToGarbage(pub Box<dyn PlayerController>);

impl PlayerController for BlindToGarbage {
    fn poll(&mut self, snapshot: &EngineSnapshot) -> InputFrame {
        if snapshot.pending_garbage.is_empty() {
            return self.0.poll(snapshot); // nothing to hide: skip the clone
        }
        let mut blinded = snapshot.clone();
        blinded.pending_garbage.clear();
        self.0.poll(&blinded)
    }
}

#[cfg(test)]
mod versus_rules_tests {
    use super::*;
    use crate::accounting::fold_combo;
    use tetr_core::ai::{AiController, Handicap};

    /// THE accounting gate for moving attack into the engine: over a real bot
    /// game with nothing queued (pending empty ⇒ net == gross), the engine's
    /// AttackSent events must total exactly what the research-side fold
    /// (`fold_combo` + `attack_lines`, the convention every APP baseline was
    /// recorded under) computes from the same event stream.
    #[test]
    fn engine_attack_events_match_the_research_fold() {
        let mut engine = Engine::new(marathon_config(), 11);
        let mut bot = AiController::new(Handicap::perfect(), 99);
        let mut combo = 0u32;
        let (mut fold_total, mut event_total) = (0u32, 0u32);
        for _ in 0..4_000 {
            if engine.snapshot().game_over.is_some() {
                break;
            }
            for event in drive_engine(&mut engine, &mut bot) {
                if let Some(clear) = fold_combo(&event, &engine, &mut combo) {
                    fold_total += clear.attack;
                }
                if let EngineEvent::AttackSent { lines } = event {
                    event_total += lines;
                }
            }
        }
        assert!(fold_total > 0, "the bot must have attacked at least once");
        assert_eq!(
            event_total, fold_total,
            "engine-side attack must reproduce the research fold bit-for-bit"
        );
    }

    /// The parallel evaluation must be bit-identical to playing the same seeds
    /// sequentially: matches are pure functions of their seed and collection
    /// is order-stable, so threading may change *when* a match runs but never
    /// *what* it returns. This is the gate that lets every suite go wide
    /// without re-recording a single baseline.
    #[test]
    fn parallel_evaluation_matches_sequential() {
        let make = |seed: u64| -> Box<dyn PlayerController> {
            Box::new(AiController::new(Handicap::perfect(), seed))
        };
        let seeds: Vec<u64> = crate::seeds::seed_set(8);
        let format = VersusFormat {
            max_plies: 40,
            rain_period: 4,
        };
        let parallel = evaluate_versus_format(&make, &make, &seeds, format);
        let sequential: Vec<VersusOutcome> = seeds
            .iter()
            .map(|&s| play_versus_format(&make, &make, s, format))
            .collect();
        assert_eq!(parallel.outcomes.len(), sequential.len());
        for (p, s) in parallel.outcomes.iter().zip(&sequential) {
            assert_eq!(
                (p.seed, p.result, p.plies, p.attack_a, p.attack_b),
                (s.seed, s.result, s.plies, s.attack_a, s.attack_b),
                "parallel and sequential evaluation diverged on seed {}",
                p.seed
            );
        }
    }

    /// A whole match is a pure function of its seed: same seed, same bots ⇒
    /// identical outcome (the property SPRT and win-rate climbs rely on).
    #[test]
    fn play_versus_is_deterministic() {
        let make = |seed: u64| -> Box<dyn PlayerController> {
            Box::new(AiController::new(Handicap::perfect(), seed))
        };
        let run = || {
            let o = play_versus(&make, &make, 42, 40);
            (
                o.result, o.plies, o.attack_a, o.attack_b, o.a_topped, o.b_topped,
            )
        };
        assert_eq!(run(), run());
    }
}

#[cfg(test)]
mod versus_decision_tests {
    use super::{VersusResult, decide_versus};

    #[test]
    fn topout_loses_before_attack_is_compared() {
        // A dies with *more* attack dealt → B still wins: a death-loss takes priority.
        assert_eq!(decide_versus(true, false, 100, 0), VersusResult::BWins);
        assert_eq!(decide_versus(false, true, 0, 100), VersusResult::AWins);
    }

    #[test]
    fn both_alive_higher_attack_wins_and_ties_draw() {
        assert_eq!(decide_versus(false, false, 5, 3), VersusResult::AWins);
        assert_eq!(decide_versus(false, false, 3, 5), VersusResult::BWins);
        assert_eq!(decide_versus(false, false, 4, 4), VersusResult::Draw);
    }

    #[test]
    fn double_topout_falls_back_to_attack_dealt() {
        // Both topped the same tick: decide by attack landed before dying.
        assert_eq!(decide_versus(true, true, 7, 2), VersusResult::AWins);
        assert_eq!(decide_versus(true, true, 2, 2), VersusResult::Draw);
    }
}
