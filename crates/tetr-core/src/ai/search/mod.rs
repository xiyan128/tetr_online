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

use smallvec::SmallVec;

use crate::ai::eval::{EvalContext, Evaluator, Reward, Value};
use crate::ai::movegen::{generate_with_hold, spawn_piece, Move, Placement};
use crate::ai::state::SearchState;
use crate::engine::{classify_t_spin, LockOutcome, PieceType, TSpinKind};

/// How much total work a planner may spend on one decision.
///
/// `max_depth` caps lookahead plies for every planner. `nodes` caps total node
/// expansions per decision for the node-counted planner ([`BestFirstPlanner`]);
/// the planners that bound their work another way ignore it — greedy finishes in
/// one shot, and the beam is bounded by its width × depth. Time-slicing (how much
/// of the budget runs per [`Planner::plan`] call before yielding
/// [`PlannerStep::NeedMoreBudget`]) is each planner's own contract, not part of
/// the budget.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SearchBudget {
    /// Total node expansions per decision for node-counted planners (best-first).
    /// `0` means uncapped — the depth cap / frontier exhaustion alone terminates.
    /// Ignored by greedy and the beam.
    pub nodes: u32,
    /// Maximum lookahead plies (current piece = depth 1).
    pub max_depth: u8,
}

impl SearchBudget {
    /// A budget for a one-shot, single-ply search (the greedy default): depth 1,
    /// no node cap (greedy ignores it).
    pub fn greedy() -> Self {
        Self {
            nodes: 0,
            max_depth: 1,
        }
    }

    /// A budget for a multi-ply beam search up to `max_depth` plies. The beam is
    /// bounded by its *width* (a [`BeamPlanner`](crate::ai::search::BeamPlanner)
    /// field) × depth and ignores `nodes`. `beam(1)` reproduces the greedy
    /// single-ply decision.
    pub fn beam(max_depth: u8) -> Self {
        Self {
            nodes: 0,
            max_depth,
        }
    }

    /// A budget for the best-first planner: `nodes` total expansions per decision
    /// (its quality dial; `0` = uncapped, terminate on depth / frontier alone — pass
    /// a real cap in production) under a `max_depth` ply cap.
    pub fn best_first(nodes: u32, max_depth: u8) -> Self {
        Self { nodes, max_depth }
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

/// Fork `parent`, classify the T-spin against the PRE-lock board (engine order), and
/// commit `placement` into the clone via [`SearchState::commit_placement`] — **the** one
/// place the per-child "fork → classify → commit" ritual lives. Returns the advanced
/// child plus the lock's `(LockOutcome, t_spin)`, which the scoring callers feed to the
/// evaluator (singly via [`score_child`], or batched by the beam).
///
/// The classify-against-the-pre-lock-board-**then**-commit order is load-bearing — it
/// mirrors the engine's own lock path (`api.rs::lock_active_piece`) — so keep it exact.
pub(crate) fn commit_child(
    parent: &SearchState,
    placement: &Placement,
) -> (SearchState, LockOutcome, Option<TSpinKind>) {
    let mut child = parent.clone();
    let t_spin = classify_t_spin(&placement.piece, &child.board);
    let lock = child.commit_placement(placement);
    (child, lock, t_spin)
}

/// Score one child placement: [`commit_child`], then evaluate the resulting board + lock
/// in a single [`Evaluator::evaluate_cols`] call. Returns the advanced child and its
/// `(Value, Reward)`.
///
/// This is the **one** place single-placement scoring lives: the greedy planner (via
/// [`score_placement`]), the imperfection sampler in `policy::search` (likewise), and
/// best-first's `children` all route through it, so they can never silently disagree on
/// what a placement is worth (the DRY/SRP fix the SOLID review flagged). The beam instead
/// builds its children with [`commit_child`] and scores a whole generation in one
/// [`Evaluator::evaluate_batch`] — the seam the neural value net needs.
pub(crate) fn score_child(
    parent: &SearchState,
    placement: &Placement,
    eval: &dyn Evaluator,
    ctx: EvalContext,
) -> (SearchState, Value, Reward) {
    let (child, lock, t_spin) = commit_child(parent, placement);
    let (value, reward) = eval.evaluate_cols(&lock, child.board.view(), t_spin, ctx);
    (child, value, reward)
}

/// Score one placement as the engine's lock path would, returning `Value + Reward` as a
/// single `i32` (higher is better) — a thin [`score_child`] wrapper for the two callers
/// that rank by the scalar and discard the child state: the greedy planner
/// ([`GreedyPlanner`]) and the imperfection sampler in `policy::search`.
pub(crate) fn score_placement(
    parent: &SearchState,
    placement: &Placement,
    eval: &dyn Evaluator,
    ctx: EvalContext,
) -> i32 {
    let (_, value, reward) = score_child(parent, placement, eval, ctx);
    (value + reward).0
}

/// Hold-aware enumeration of `state`'s active-piece placements in canonical movegen
/// order — the single seam the greedy, beam, and best-first planners share. The
/// movegen BFS re-derives reachable poses, so a hold swap only needs an on-board
/// spawn, which the board's own `(width, height)` always provides.
pub(crate) fn hold_placements(state: &SearchState) -> Vec<Placement> {
    let (w, h) = (state.board.width(), state.board.height());
    generate_with_hold(
        &state.board,
        &state.active,
        state.hold,
        state.queue.first().copied(),
        move |pt| spawn_piece(pt, w, h),
    )
}

/// The final decision shared by both multi-ply planners: the ply-1 placement whose
/// backed-up score is maximal, with the **first** maximum winning (`>` scan over
/// `root_best` in canonical order) so the result is deterministic (BEAM.md §4).
/// `roots` and `root_best` are index-aligned (root `i`'s best back-up is `root_best[i]`).
pub(crate) fn best_root_plan(roots: &[Placement], root_best: &[i32]) -> PlacementPlan {
    let mut best_i = 0usize;
    let mut best_score = root_best[0];
    for (i, &score) in root_best.iter().enumerate().skip(1) {
        if score > best_score {
            best_score = score;
            best_i = i;
        }
    }
    PlacementPlan {
        placement: roots[best_i].clone(),
        score: best_score,
    }
}

/// The cheap, exact identity of a search-**root** state — the shared core of the
/// beam's stale-run detector ([`BeamPlanner`]) and best-first's transposition key
/// ([`BestFirstPlanner`]), which were near-identical structs before.
///
/// Every field is compared by value (the derived [`PartialEq`]); equality is exact, so
/// two states match only when they are byte-for-byte the same root. The board identity
/// is the **column bitboard** (`columns()`) — a complete, allocation-free fingerprint —
/// *not* `cell_coords()`, so building a key allocates only the two `SmallVec`s.
/// [`Eq`]/[`Hash`] are derived too so best-first can layer a `root_index` on top and use
/// the result as a transposition-table key.
#[derive(Clone, PartialEq, Eq, Hash)]
pub(crate) struct RootKey {
    active: PieceType,
    active_origin: (isize, isize),
    active_rotation: u8,
    hold: Option<PieceType>,
    queue: SmallVec<[PieceType; 8]>,
    b2b: bool,
    combo: u32,
    board: SmallVec<[u64; 16]>,
}

impl RootKey {
    /// Snapshot `state`'s root identity (active pose, hold, revealed queue, B2B/combo
    /// chain, and the column-bitboard board fingerprint).
    pub(crate) fn of(state: &SearchState) -> Self {
        Self {
            active: state.active.piece_type(),
            active_origin: state.active.origin(),
            active_rotation: state.active.rotation() as u8,
            hold: state.hold,
            queue: state.queue.iter().copied().collect(),
            b2b: state.b2b,
            combo: state.combo,
            board: state.board.columns().into(),
        }
    }
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
