//! The placement search: an anytime, re-rootable session seam.
//!
//! A search turns a [`SearchState`] into a decision: which placement to play next,
//! and the [`Move`] path to execute it. The [`Mind`] trait is the **session**
//! contract every search paradigm implements — the one-shot greedy argmax, the
//! batch-shaped beam, and the transposition best-first today; an MCGS or
//! neural-guided search later (research finding \[6\]: the strong reference bot,
//! Cold Clear, is a transposition-deduplicating DAG search) — so a stronger brain
//! drops in with no rework of the evaluator, movegen, controller, or
//! plan-to-input layers.
//!
//! # Two currencies: work and time
//!
//! A [`Mind`] measures effort in **nodes** ([`Mind::think`] spends a node quantum;
//! [`Mind::nodes_expanded`] meters it) and never reads a clock. Callers own
//! **time** — frames, reaction windows, deadlines — and convert it to work. That
//! split is what lets every venue drive the same session:
//!
//! - the blocking drain ([`think_to_completion`]) for headless benchmarks, tests,
//!   and the synchronous runner — exact budgets, reproducible decisions;
//! - a cooperative interactive venue that spends a small quantum per frame on the
//!   main thread (no per-piece hitch);
//! - a thread/worker venue that thinks continuously and is asked for
//!   [`Mind::best`] when the controller's deadline lands.
//!
//! [`Mind::best`] is **anytime**: valid as soon as [`Mind::reroot`] seeds the
//! ply-1 roots, and refined by every [`Mind::think`]. Re-rooting at the state the
//! session is already rooted at is a cheap fingerprint compare, so callers
//! re-assert the root every poll and a stale in-flight search can never leak into
//! a new decision.
//!
//! # Determinism
//!
//! Pure Rust, no Bevy, no RNG, no clock — like [`crate::engine`]. A mind's
//! decision is a deterministic function of `(state, evaluator, depth cap, total
//! nodes thought)`; quantum *granularity* never changes the answer, only when it
//! arrives. Tie-breaking among equally-scored placements is resolved by movegen's
//! canonical placement order (stable), never by randomness; any error injection
//! the AI wants lives in the policy's own seeded RNG, never here.

pub mod beam;
pub mod best_first;
pub mod pc_coverage;

pub use beam::{BeamPlanner, RootFilter};
pub use best_first::BestFirstPlanner;
pub use pc_coverage::{PcCoverageConfig, PcCoveragePlanner, PcCoverageUnit};

use smallvec::SmallVec;

use crate::ai::eval::{EvalContext, Evaluator, Reward, Value};
use crate::ai::movegen::{Move, Placement, generate_with_hold, spawn_piece};
use crate::ai::state::SearchState;
use crate::engine::{LockOutcome, PieceType, TSpinKind, classify_t_spin};

/// How much total work one decision may spend.
///
/// `max_depth` caps lookahead plies for every mind. `nodes` caps total node
/// expansions per decision for node-budgeted operating points; `SearchBudget::beam`
/// leaves it uncapped because the beam is bounded by its width × depth. The budget
/// is the **caller's** meter (checked against [`Mind::nodes_expanded`], as
/// [`think_to_completion`] does): the mind itself never sees it, only the per-call
/// think quantum.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SearchBudget {
    /// Total node expansions per decision for node-metered minds (best-first).
    /// `0` means uncapped — the depth cap / frontier exhaustion alone terminates.
    /// `SearchBudget::beam` sets this to `0`; beam work is bounded by width × depth.
    pub nodes: u32,
    /// Maximum lookahead plies (current piece = depth 1).
    pub max_depth: u8,
}

impl SearchBudget {
    /// A budget for a one-shot, single-ply search — the greedy baseline's
    /// operating point (best-first at depth 1 IS the single-ply argmax; the
    /// dedicated greedy planner died in the no-compat sweep, its decisions
    /// gate-pinned identical): depth 1, no node cap (the seed generation is
    /// drained without expansion).
    pub fn single_ply() -> Self {
        Self {
            nodes: 0,
            max_depth: 1,
        }
    }

    /// A budget for a multi-ply beam search up to `max_depth` plies. The beam is
    /// bounded by its *width* (a [`BeamPlanner`] field) × depth, so this leaves
    /// `nodes` uncapped. `beam(1)` reproduces the greedy single-ply decision.
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
        Self::single_ply()
    }
}

/// A planner's chosen placement, ready for the plan-to-input layer.
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

/// The backed-up score of a **dead** branch (lock-out, overflowing rise, or a
/// blocked spawn — see [`SearchState::dead`]): far below any real evaluation so
/// survival always outranks death, yet far above `i32::MIN` so summing path
/// rewards onto it can never overflow. Death is absolute — accumulated rewards
/// along the path cannot rescue a line the engine would end.
pub(crate) const DEATH_SCORE: i32 = -100_000_000;

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
/// builds its children with [`commit_child`] and stages a generation before publishing
/// the next frontier.
pub(crate) fn score_child(
    parent: &SearchState,
    placement: &Placement,
    eval: &dyn Evaluator,
    ctx: EvalContext,
) -> (SearchState, Value, Reward) {
    let (child, lock, t_spin) = commit_child(parent, placement);
    if child.dead {
        // The engine's game ends on this branch: the board the evaluator would
        // read is the truncated remnant of a death, not a position — score it
        // as death and skip the eval.
        return (child, Value(DEATH_SCORE), Reward(0));
    }
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
/// order — the single seam the greedy, beam, best-first, and policy-argmax
/// planners share (public: research minds enumerate through it too, so they
/// can never disagree with the in-crate planners about what a move is). The
/// movegen BFS re-derives reachable poses, so a hold swap only needs an on-board
/// spawn, which the board's own `(width, height)` always provides.
pub fn hold_placements(state: &SearchState) -> Vec<Placement> {
    let (w, h) = (state.board.width(), state.board.height());
    generate_with_hold(
        &state.board,
        &state.active,
        state.hold,
        state.queue.first().copied(),
        move |pt| spawn_piece(pt, w, h),
    )
}

/// [`hold_placements`] without path tracking — the same placements in the same
/// canonical order, empty paths. For INTERIOR search plies only: their
/// placements advance state by pose ([`SearchState::commit_placement`]) and are
/// never rendered to inputs, so per-node path bookkeeping is pure waste
/// (Stage-0 deferred lever #1). Never use for ply-1 roots (their paths ARE the
/// input synthesis).
pub fn hold_placements_pathless(state: &SearchState) -> Vec<Placement> {
    let (w, h) = (state.board.width(), state.board.height());
    crate::ai::movegen::generate_with_hold_pathless(
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
    /// Pending garbage is part of a state's future (it decides when and where
    /// rows rise), so two states differing only here must never transpose.
    pending: crate::engine::garbage::BatchQueue,
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
            pending: state.pending.clone(),
        }
    }
}

/// Progress report from [`Mind::think`]: whether more thinking can still matter.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ThinkProgress {
    /// Expandable work remains — more [`Mind::think`] can improve [`Mind::best`].
    Working,
    /// The search is exhausted (frontier drained / depth cap reached): further
    /// `think` calls are no-ops and [`Mind::best`] is final for this root.
    Exhausted,
}

/// An anytime, re-rootable search session — the AI's thinking core.
///
/// One `Mind` owns one in-flight search (frontier, transposition table, per-root
/// back-ups) and is driven through three verbs:
///
/// 1. [`reroot`](Self::reroot) — point the session at a decision state. A no-op
///    when already rooted there (thinking continues across calls); otherwise the
///    stale run is discarded and the ply-1 roots are seeded, which makes
///    [`best`](Self::best) immediately valid.
/// 2. [`think`](Self::think) — spend up to a quantum of node expansions.
/// 3. [`best`](Self::best) — the best ply-1 plan found so far, at any time.
///
/// Object-safe (`&mut dyn Mind`) so a policy holds one behind a trait object and
/// swaps paradigms without code changes; `Send + Sync` so a venue may move the
/// session off-thread on native targets.
pub trait Mind: Send + Sync {
    /// Root the session at `state`, seeding the ply-1 placements (scored with
    /// `eval`) under a `max_depth` ply cap. Re-rooting at the **same**
    /// `(state, max_depth)` is a cheap fingerprint compare that preserves the
    /// in-flight search; any other root discards it and re-seeds. After this call
    /// [`best`](Self::best) reflects `state` (it is `Some` unless the state has
    /// no legal placement).
    fn reroot(&mut self, state: &SearchState, eval: &dyn Evaluator, max_depth: u8);

    /// Advance the current root's search by up to `quantum` node expansions.
    ///
    /// Minds should treat `quantum` as an upper bound on node expansions. A mind may
    /// still choose coarser publication points for [`best`](Self::best), but the
    /// quantum itself only chooses *suspension points*: schedules with equal total
    /// work reach the identical final answer regardless of slicing. Without a
    /// rooted run this is a no-op reporting [`ThinkProgress::Exhausted`].
    fn think(&mut self, quantum: u32, eval: &dyn Evaluator) -> ThinkProgress;

    /// The best ply-1 plan for the current root **right now** (anytime — backed
    /// up from everything thought so far). `None` before any
    /// [`reroot`](Self::reroot), or when the root has no legal placement (topped
    /// out).
    fn best(&self) -> Option<PlacementPlan>;

    /// Node expansions spent on the current root so far — the meter a caller
    /// checks against [`SearchBudget::nodes`]. Resets when a re-root discards
    /// the run.
    fn nodes_expanded(&self) -> u32;
}

/// Drive `mind` to its final decision for `state` in one blocking call: reroot,
/// think until the node budget is spent or the search exhausts, return the best.
///
/// This is the **direct-drive** venue — headless benchmarks, tests, and the
/// blocking [`SyncRunner`](crate::ai::runner::SyncRunner) — where exact budgets
/// and zero pacing matter. Interactive venues instead spread the same total work
/// across frames and read the same answer (quantum granularity never changes a
/// decision).
pub fn think_to_completion(
    mind: &mut dyn Mind,
    state: &SearchState,
    eval: &dyn Evaluator,
    budget: SearchBudget,
) -> Option<PlacementPlan> {
    // Bounded so a misbehaving mind (one that never exhausts under an uncapped
    // budget) cannot spin forever; real minds finish in a few coarse calls.
    const MAX_THINK_CALLS: u32 = crate::ai::MAX_THINK_CALLS;

    mind.reroot(state, eval, budget.max_depth);
    for _ in 0..MAX_THINK_CALLS {
        let remaining = match budget.nodes {
            0 => u32::MAX, // uncapped: depth / frontier exhaustion terminates
            cap => cap.saturating_sub(mind.nodes_expanded()),
        };
        if remaining == 0 {
            break;
        }
        if mind.think(remaining, eval) == ThinkProgress::Exhausted {
            break;
        }
    }
    mind.best()
}
