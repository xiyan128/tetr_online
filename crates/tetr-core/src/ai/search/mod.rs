//! The placement planner: a paradigm-agnostic search seam (AI3.3).
//!
//! A *planner* turns a [`SearchState`] into a decision: which placement to play
//! next, and the [`Move`] path to execute it. The [`Planner`] trait is deliberately
//! **search-paradigm-agnostic** — it says nothing about *how* the decision is made.
//! The shipped [`greedy`] planner is a one-piece greedy search (Tier-1), but
//! research finding [6] shows the strong reference bot (Cold Clear) uses a
//! transposition-deduplicating DAG / Monte-Carlo search, so a future Tier-2 planner
//! must be able to drop in behind this same trait with no rework of the evaluator,
//! movegen, controller, or plan-to-input layers.
//!
//! # Why a node budget and an incremental step
//!
//! [`Planner::plan`] takes a [`SearchBudget`] (a node cap + a depth cap) and returns
//! a [`PlannerStep`]: either [`PlannerStep::Done`] with a [`PlacementPlan`], or
//! [`PlannerStep::NeedMoreBudget`] to be polled again. This shape exists for the
//! cross-platform compute runner (AI3.5): on threadless WASM the search advances a
//! bounded slice per frame (cooperative time-slicing), so a multi-ply Tier-2 search
//! can yield and resume. The Tier-1 greedy planner always finishes in one call and
//! returns [`PlannerStep::Done`] immediately, ignoring the node cap.
//!
//! # Determinism
//!
//! Pure Rust, no Bevy, no RNG, no clock — like [`crate::engine`]. A planner is a
//! deterministic function of `(state, evaluator, budget)`. Tie-breaking among
//! equally-scored placements is resolved by movegen's canonical placement order
//! (stable), never by randomness; any error injection the AI wants lives in the
//! controller's own seeded RNG, never here.

pub mod beam;
pub mod best_first;
pub mod greedy;

pub use beam::BeamPlanner;
pub use best_first::BestFirstPlanner;
pub use greedy::GreedyPlanner;

use crate::ai::eval::{EvalContext, Evaluator};
use crate::ai::movegen::{Move, Placement};
use crate::ai::state::SearchState;
use crate::engine::{classify_t_spin, BitBoard};

/// How much work a planner may do in one [`Planner::plan`] call.
///
/// `nodes` caps how many placements/states the search may expand before it must
/// yield (the unit a WASM time-slice is measured in); `max_depth` caps lookahead
/// plies. The greedy Tier-1 planner finishes in one call and ignores `nodes`; a
/// future beam/DAG planner honours both so it can be polled incrementally.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SearchBudget {
    /// Maximum states to expand before yielding. `0` means "unbounded this call".
    pub nodes: u32,
    /// Maximum lookahead plies (current piece = depth 1).
    pub max_depth: u8,
}

impl SearchBudget {
    /// A budget for a one-shot, single-ply search (the greedy default): unbounded
    /// nodes, depth 1.
    pub fn greedy() -> Self {
        Self {
            nodes: 0,
            max_depth: 1,
        }
    }

    /// A budget for a multi-ply beam search up to `max_depth` plies, expanding a
    /// whole generation per `plan` call (`nodes == 0`, unbounded per call). The beam
    /// *width* is a [`BeamPlanner`](crate::ai::search::BeamPlanner) field, not part of
    /// the budget. `beam(1)` reproduces the greedy single-ply decision.
    pub fn beam(max_depth: u8) -> Self {
        Self {
            nodes: 0,
            max_depth,
        }
    }
}

impl Default for SearchBudget {
    fn default() -> Self {
        Self::greedy()
    }
}

/// A planner's chosen placement, ready for the plan-to-input layer (AI3.4).
///
/// Carries the [`Placement`] (its resting pose + the [`Move`] path movegen found)
/// and the score the evaluator gave it, so a caller can compare plans or surface a
/// debug overlay. The path already includes a leading [`Move::Hold`] when the plan
/// chose to hold first (see [`Placement::used_hold`]).
#[derive(Clone, Debug)]
pub struct PlacementPlan {
    /// The placement to execute.
    pub placement: Placement,
    /// The evaluator's total score for this placement (`Value + Reward`), in the
    /// evaluator's arbitrary units — higher is better. For ranking/printing only;
    /// the controller does not re-derive it.
    pub score: i32,
}

impl PlacementPlan {
    /// The [`Move`] path to execute this plan (a borrow of the placement's path).
    pub fn path(&self) -> &[Move] {
        &self.placement.path
    }

    /// Whether this plan begins with a hold swap.
    pub fn uses_hold(&self) -> bool {
        self.placement.used_hold
    }
}

/// Score one placement exactly as the engine's lock path would: classify the
/// T-spin against the pre-lock `board`, lock the placement's piece into a clone,
/// then evaluate the resulting board + lock outcome. Returns `Value + Reward` as a
/// single `i32` (higher is better).
///
/// This is the **one** place per-placement scoring lives. Both the greedy planner
/// ([`GreedyPlanner`]) and the imperfection sampler in `policy::search` score through it,
/// so the two can never silently disagree on what a placement is worth (the
/// DRY/SRP fix the SOLID review flagged).
pub(crate) fn score_placement(
    board: &BitBoard,
    placement: &Placement,
    eval: &dyn Evaluator,
    ctx: EvalContext,
) -> i32 {
    // Classify the T-spin against the board *before* the lock mutates it (engine
    // order), then lock into a copy and evaluate the result.
    let mut board = *board;
    let t_spin = classify_t_spin(&placement.piece, &board);
    let lock = board.lock_piece(&placement.piece);
    let (value, reward) = eval.evaluate_cols(&lock, &board, t_spin, ctx);
    (value + reward).0
}

/// The result of one [`Planner::plan`] call.
///
/// A one-shot planner returns [`PlannerStep::Done`] immediately. An incremental
/// planner returns [`PlannerStep::NeedMoreBudget`] until its search converges, so a
/// cooperative time-sliced runner can poll it across frames.
#[derive(Clone, Debug)]
pub enum PlannerStep {
    /// The search produced a plan. `None` means the state had **no** legal
    /// placement (e.g. the board is already topped out), which the controller
    /// treats as "do nothing / game is lost".
    Done(Option<PlacementPlan>),
    /// The search needs to be polled again with more budget (incremental planners).
    NeedMoreBudget,
}

/// A placement planner. Object-safe (`&mut dyn Planner`) so the controller can hold
/// one behind a trait object and swap Tier-1 for Tier-2 without code changes.
///
/// `Send + Sync` so the search may run off-thread on native targets (AI3.5).
pub trait Planner: Send + Sync {
    /// Plan the next placement from `state`, scoring candidates with `eval` under
    /// `budget`. See [`PlannerStep`] for the one-shot vs incremental contract.
    fn plan(
        &mut self,
        state: &SearchState,
        eval: &dyn Evaluator,
        budget: SearchBudget,
    ) -> PlannerStep;
}
