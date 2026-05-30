//! Difficulty knobs for the AI controller (AI3.5).
//!
//! [`DifficultyConfig`] is the small bundle of tunables that turn the *same*
//! evaluator + planner into a weaker or stronger opponent without touching the
//! search itself. The plan (M2 §AI3.5, open question OQ2) calls for a
//! **beatable / even** default rather than peak strength: a perfect bot is no
//! fun, and no web source survived verification on human-like degradation, so
//! these defaults are pragmatic and meant to be tuned empirically in the AI3.6
//! sandbox.
//!
//! # The four knobs
//!
//! - [`think_time`](DifficultyConfig::think_time) — a reaction delay before the
//!   bot starts acting on a freshly spawned piece. Models human "look, then
//!   move" latency and paces the bot so it does not place instantly.
//! - [`error_rate`](DifficultyConfig::error_rate) — the probability the bot does
//!   *not* play its top-scored placement. The controller blends this into a
//!   top-N softmax sample over candidate placements (a higher rate widens the
//!   sampling), so mistakes look like plausible suboptimal choices rather than
//!   random flailing.
//! - [`max_depth`](DifficultyConfig::max_depth) — lookahead plies handed to the
//!   planner via [`SearchBudget`](crate::ai::search::SearchBudget). Tier-1 greedy
//!   ignores depth beyond 1; it is here so raising difficulty later (Tier-2)
//!   needs no controller change.
//! - [`nodes_per_tick`](DifficultyConfig::nodes_per_tick) — the per-poll search
//!   node budget, the unit a cooperative WASM time-slice is measured in. Cheap
//!   for greedy (which finishes in one call) but load-bearing once a Tier-2 beam
//!   is time-sliced across frames.
//!
//! # Determinism
//!
//! Pure data — no Bevy, no RNG, no clock. The randomness `error_rate` drives
//! lives in the controller's own seeded RNG, never here.

use core::time::Duration;

/// Tunable opponent difficulty: think-time, error rate, lookahead depth, and the
/// per-tick search budget. Cloneable plain data; the controller reads it each
/// poll so it can be swapped live (e.g. a difficulty slider).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DifficultyConfig {
    /// Reaction delay before the bot acts on a new piece. Accumulated from the
    /// per-frame `dt` the controller is polled with; while it has not elapsed the
    /// controller emits neutral frames ("thinking").
    pub think_time: Duration,
    /// Probability in `0.0..=1.0` that the bot plays a *non-optimal* placement.
    /// `0.0` is a flawless bot (always the top placement); higher values widen a
    /// top-N softmax sample over candidates. Clamped on use.
    pub error_rate: f32,
    /// Lookahead plies for the planner (current piece = depth 1). Greedy Tier-1
    /// ignores anything past 1; kept so a deeper Tier-2 planner needs no
    /// controller change.
    pub max_depth: u8,
    /// Search node budget per poll — the unit of a cooperative time-slice on
    /// threadless WASM. Greedy finishes in one call and effectively ignores it; a
    /// future beam honours it to stay responsive.
    pub nodes_per_tick: u32,
}

impl DifficultyConfig {
    /// A flawless, instant bot: no think-time, no error, greedy depth. Useful as
    /// a deterministic baseline for tests and the strongest setting.
    pub fn perfect() -> Self {
        Self {
            think_time: Duration::ZERO,
            error_rate: 0.0,
            max_depth: 1,
            nodes_per_tick: DEFAULT_NODES_PER_TICK,
        }
    }

    /// The number of top placements the controller's softmax samples from when
    /// injecting an error. A small window keeps mistakes *plausible* (a near-best
    /// alternative) rather than catastrophic.
    pub const ERROR_SAMPLE_WINDOW: usize = 4;
}

/// A sensible per-tick node budget. Greedy ignores it; sized so a future beam
/// gets meaningful progress per frame without stalling.
const DEFAULT_NODES_PER_TICK: u32 = 2_000;

impl Default for DifficultyConfig {
    /// A **beatable / even** opponent (M2 plan default): a short reaction delay,
    /// a small error rate so it occasionally misplaces, and greedy depth. Tuned
    /// to be fun to play against, not to win.
    fn default() -> Self {
        Self {
            // ~12 fixed slices at 60 Hz: visible but not sluggish.
            think_time: Duration::from_millis(200),
            // Misplaces roughly one piece in eight.
            error_rate: 0.12,
            max_depth: 1,
            nodes_per_tick: DEFAULT_NODES_PER_TICK,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_beatable_not_perfect() {
        let d = DifficultyConfig::default();
        // A beatable opponent reacts (non-zero think-time) and errs sometimes.
        assert!(d.think_time > Duration::ZERO, "should pause to 'think'");
        assert!(d.error_rate > 0.0, "should make occasional mistakes");
        assert!(
            d.error_rate < 0.5,
            "but still play well most of the time (error rate < 50%)"
        );
    }

    #[test]
    fn perfect_has_no_handicap() {
        let d = DifficultyConfig::perfect();
        assert_eq!(d.think_time, Duration::ZERO);
        assert_eq!(d.error_rate, 0.0);
    }
}
