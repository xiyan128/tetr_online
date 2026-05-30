//! The AI brain: a model-agnostic decision [`Policy`].
//!
//! A [`Policy`] maps an [`Observation`] of the game to a [`Decision`] — *which
//! placement to play*. It is deliberately model-agnostic: the greedy search
//! ([`SearchPolicy`]) implements it today, and a future neural-net or MCTS policy
//! implements the *same* trait, so the [`AiController`](crate::ai::AiController)
//! shell drives any of them with no change. The controller knows nothing about
//! search, evaluators, or weights — only `Policy`.
//!
//! # Why placement-level
//!
//! Decisions are at *placement* level, not per-frame input: search, a neural
//! policy, and MCTS all decide "where does this piece go", and the shell renders
//! the chosen placement to [`InputFrame`](crate::engine::InputFrame)s via
//! [`placement_to_inputs`](crate::ai::placement_to_inputs). The same placement-
//! level action is the unit a future self-play / training environment would step.
//!
//! # Determinism
//!
//! A policy is a deterministic function of its observation and its own seeded RNG
//! (set at construction) — no clock, no OS entropy. Any *imperfection* (deliberate
//! suboptimal play, the handicap) is the policy's own concern: a search degrades
//! via a softmax over candidates, a net via sampling temperature. Keeping it inside
//! the policy is what lets the controller shell stay model-blind.

mod search;

pub use search::SearchPolicy;

use crate::ai::movegen::Placement;
use crate::ai::state::SearchState;

/// What a [`Policy`] observes: the game state it decides from.
///
/// Today this is the [`SearchState`] (board + active piece + hold + queue + bag);
/// a neural policy encodes it into tensors itself. It is aliased so the [`Policy`]
/// seam reads as model-agnostic and a richer observation type can replace it later
/// without churning every signature.
pub type Observation = SearchState;

/// A policy's decision for the current piece.
#[derive(Clone, Debug)]
pub enum Decision {
    /// Play this placement. The controller renders it to input frames; the path
    /// may begin with a hold swap (see [`Placement::used_hold`]).
    Place(Placement),
    /// No legal placement — the board is effectively topped out, nothing to do.
    None,
}

/// The AI brain: decide which placement to play from an [`Observation`].
///
/// Model-agnostic — search, neural, and hybrid policies all implement it, and the
/// [`AiController`](crate::ai::AiController) shell drives any of them. `Send` so a
/// policy can run off-thread on native targets (the off-thread runner moves it to a
/// worker and back).
pub trait Policy: Send {
    /// Decide the placement to play for `obs`. Any randomness (tie-breaking, the
    /// imperfection handicap) uses the policy's own seeded RNG — never the engine's.
    fn decide(&mut self, obs: &Observation) -> Decision;
}
