//! The AI handicap: how a strong bot is deliberately weakened.
//!
//! [`Handicap`] is the small, model-agnostic bundle that turns a strong bot into a
//! beatable opponent. It is split across the AI player's two layers, and consumed
//! at construction:
//!
//! - [`reaction`](Handicap::reaction) is consumed by the **controller shell** — a
//!   delay before the bot acts on a new piece (human "look, then move" latency).
//! - [`imperfection`](Handicap::imperfection) is consumed by the **policy** (the
//!   brain) — how sub-optimally it plays. Each model interprets it: the search
//!   policy softmax-samples a near-best placement; a neural policy would raise its
//!   sampling temperature.
//!
//! The name is deliberately *not* `Difficulty`: this is the mechanism, not the
//! player-facing Easy/Medium/Hard label a menu shows (a UI level maps *onto* a
//! `Handicap`). Search *capability* (lookahead depth, node budget) is **not** here
//! — that belongs to the search policy, not the handicap.
//!
//! The default is **beatable / even** rather than peak strength: a perfect bot
//! is no fun. No published source on human-like degradation survived
//! verification, so the defaults are pragmatic, meant to be tuned empirically
//! against real play.
//!
//! # Determinism
//!
//! Pure data — no Bevy, no RNG, no clock. The randomness `imperfection` drives
//! lives in the policy's own seeded RNG, never here.

use core::time::Duration;

/// The AI's handicap: a deliberate, model-agnostic weakening of an otherwise-strong
/// bot into a beatable opponent. Cloneable plain data; `reaction` is consumed by the
/// controller shell, `imperfection` by the policy.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Handicap {
    /// Reaction delay before the bot acts on a new piece. The controller integrates
    /// the per-frame `dt` it is polled with; while it has not elapsed the controller
    /// emits neutral frames ("reacting").
    pub reaction: Duration,
    /// How imperfectly the bot plays, in `0.0..=1.0`. `0.0` is flawless (always the
    /// policy's best move); higher values make it more likely to play a plausible
    /// near-best instead. The policy interprets it (a search: a softmax over
    /// candidates; a net: sampling temperature). Clamped on use.
    pub imperfection: f32,
}

impl Handicap {
    /// A flawless, instant bot: no reaction delay, no imperfection. The strongest
    /// setting and a deterministic baseline for tests.
    pub fn perfect() -> Self {
        Self {
            reaction: Duration::ZERO,
            imperfection: 0.0,
        }
    }
}

impl Default for Handicap {
    /// A **beatable / even** opponent: a short reaction delay and
    /// a small imperfection so it occasionally misplaces. Tuned to be fun to play
    /// against, not to win.
    fn default() -> Self {
        Self {
            // ~12 fixed slices at 60 Hz: visible but not sluggish.
            reaction: Duration::from_millis(200),
            // Misplaces roughly one piece in eight.
            imperfection: 0.12,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_beatable_not_perfect() {
        let h = Handicap::default();
        // A beatable opponent reacts (non-zero delay) and errs sometimes.
        assert!(h.reaction > Duration::ZERO, "should pause to 'react'");
        assert!(h.imperfection > 0.0, "should make occasional mistakes");
        assert!(
            h.imperfection < 0.5,
            "but still play well most of the time (imperfection < 50%)"
        );
    }

    #[test]
    fn perfect_has_no_handicap() {
        let h = Handicap::perfect();
        assert_eq!(h.reaction, Duration::ZERO);
        assert_eq!(h.imperfection, 0.0);
    }
}
