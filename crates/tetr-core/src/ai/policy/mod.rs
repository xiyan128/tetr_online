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

/// Progress of an in-flight policy decision (see [`Policy::think`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PolicyProgress {
    /// More thinking would still improve the decision (budget unspent, search not
    /// exhausted) — keep calling [`Policy::think`].
    Working,
    /// The decision has reached its contract quality (budget spent or search
    /// exhausted): [`Policy::take`] returns it at full strength.
    Ready,
}

/// The AI brain: decide which placement to play from an [`Observation`].
///
/// Model-agnostic — search, neural, and hybrid policies all implement it, and the
/// [`AiController`](crate::ai::AiController) shell drives any of them. `Send` so a
/// policy can run off-thread on native targets (the off-thread runner moves it to a
/// worker and back).
///
/// # One-shot and incremental driving
///
/// [`decide`](Policy::decide) is the blocking, direct-drive verb: one call, full
/// budget — what headless benchmarks and the synchronous runner use.
///
/// The three incremental verbs let a venue spread the same decision across frames
/// (or across a thread boundary) without the policy knowing where it runs:
///
/// 1. [`reroot`](Policy::reroot) — point the thinking at `obs`. Cheap when
///    already rooted there (an in-flight search continues).
/// 2. [`think`](Policy::think) — spend up to a quantum of work; returns
///    [`PolicyProgress::Ready`] once the decision's budget contract is met.
/// 3. [`take`](Policy::take) — the decision **now** (anytime): the best plan so
///    far plus the policy's own error model. Valid even before `Ready` — a
///    deadline-pressed caller gets the strongest answer available.
///
/// One-shot policies (a neural forward pass, a trivial heuristic) keep the
/// defaults: instantly `Ready`, with `take` delegating to `decide`. The blocking
/// `decide` of an incremental policy must equal `reroot` + `think`-until-`Ready` +
/// `take` — same verbs, fused.
pub trait Policy: Send {
    /// Decide the placement to play for `obs` in one blocking call (full budget).
    /// Any randomness (tie-breaking, the imperfection handicap) uses the policy's
    /// own seeded RNG — never the engine's.
    fn decide(&mut self, obs: &Observation) -> Decision;

    /// Point the in-flight thinking at `obs` (a no-op when already rooted there).
    /// One-shot policies have no in-flight state: the default does nothing.
    fn reroot(&mut self, _obs: &Observation) {}

    /// Spend up to `quantum` units of work on the current root. One-shot policies
    /// are always instantly [`PolicyProgress::Ready`] (the default).
    fn think(&mut self, _quantum: u32) -> PolicyProgress {
        PolicyProgress::Ready
    }

    /// Take the decision for `obs` **now** — anytime, from whatever has been
    /// thought so far (re-rooting first if the caller never did). The default
    /// delegates to the blocking [`decide`](Policy::decide).
    fn take(&mut self, obs: &Observation) -> Decision {
        self.decide(obs)
    }
}
