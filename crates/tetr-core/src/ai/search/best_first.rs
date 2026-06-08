//! A best-first graph-search planner — a Tier-2 alternative to the [`beam`](super::beam).
//!
//! # Why, beyond the beam
//!
//! The [`BeamPlanner`](super::BeamPlanner) truncates each generation to a fixed width,
//! so a line that scores low early but pays off later — a combo build-up, a deep
//! T-spin tower — is pruned before its payoff is visible. This planner instead expands
//! the **single most promising frontier node** each step (best-first; the exploitation
//! limit of MCTS), bounded by a **node budget** rather than a fixed beam width, so the
//! search spends its budget going deep where the eval is most optimistic. Finding 6/§9
//! ([`super`] module docs) flags exactly this — Cold Clear's transposition-DAG search.
//!
//! # Transposition table (the DAG)
//!
//! Different placement / hold orders reach the **same** position; that position has one
//! future regardless of the path, so the paths are interchangeable — keep the
//! highest-scoring. A per-root [`StateKey`] → best-score map drops the rest, so the
//! budget is spent on *distinct* positions, not re-derivations. The key is **per-root**
//! (it carries the ply-1 `root_index`): a state shared by two different first moves is
//! explored under each, so each root's best line is credited correctly.
//!
//! # Determinism (matches [`super`] §Determinism)
//!
//! No RNG, no clock. The frontier is a max-heap keyed by `(score, insertion order)`:
//! among equal scores the **earliest-enqueued** (canonical movegen order) pops first.
//! Back-up uses `>` so the first maximum wins, mirroring the beam / greedy rule.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use rustc_hash::FxHashMap;
use smallvec::SmallVec;

use crate::ai::eval::{EvalContext, Evaluator, Reward};
use crate::ai::movegen::{self, Placement};
use crate::ai::search::{PlacementPlan, Planner, PlannerStep, SearchBudget};
use crate::ai::state::SearchState;
use crate::engine::{classify_t_spin, PieceType};

/// Default total node-expansion budget per decision (across `plan` calls). Roughly an
/// order of magnitude more reachable positions than a width-16 depth-6 beam, spent
/// best-first rather than uniformly.
pub const DEFAULT_NODE_BUDGET: u32 = 4000;

/// Nodes expanded per `plan` call before yielding (the WASM time-slice unit). The
/// blocking [`SearchPolicy::plan_best`](crate::ai::SearchPolicy) loops until `Done`.
const EXPAND_CHUNK: u32 = 1024;

/// Identity of a search state for transposition: same key ⇒ same future, so two paths
/// reaching it are interchangeable. **Per-root** (`root_index` is part of the key) so a
/// position shared by two ply-1 moves is kept once *per root* and credited correctly.
#[derive(Clone, PartialEq, Eq, Hash)]
struct StateKey {
    root_index: usize,
    board: SmallVec<[u64; 16]>,
    active: PieceType,
    active_origin: (isize, isize),
    active_rotation: u8,
    hold: Option<PieceType>,
    queue: SmallVec<[PieceType; 8]>,
    b2b: bool,
    combo: u32,
}

impl StateKey {
    fn of(state: &SearchState, root_index: usize) -> Self {
        Self {
            root_index,
            board: state.board.columns().into(),
            active: state.active.piece_type(),
            active_origin: state.active.origin(),
            active_rotation: state.active.rotation() as u8,
            hold: state.hold,
            queue: state.queue.iter().copied().collect(),
            b2b: state.b2b,
            combo: state.combo,
        }
    }
}

/// One frontier node: a forked state plus the path bookkeeping the back-up needs.
struct Node {
    state: SearchState,
    /// Summed per-move [`Reward`] from the root to here (folded into the leaf Value).
    acc_reward: Reward,
    /// Which ply-1 root this descends from (the move the decision ultimately returns).
    root_index: usize,
    /// `(leaf_value + acc_reward).0` — the best-first priority.
    score: i32,
    /// Plies from the root (root placements are depth 1).
    depth: u8,
    /// Enqueue sequence number — the deterministic tie-breaker for equal scores.
    order: u64,
}

// The heap orders by score (max-first); ties go to the EARLIER-enqueued node (lower
// `order`), i.e. canonical movegen order — so the search is fully deterministic.
impl Ord for Node {
    fn cmp(&self, other: &Self) -> Ordering {
        self.score
            .cmp(&other.score)
            .then_with(|| other.order.cmp(&self.order))
    }
}
impl PartialOrd for Node {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl PartialEq for Node {
    fn eq(&self, other: &Self) -> bool {
        self.score == other.score && self.order == other.order
    }
}
impl Eq for Node {}

/// The in-flight search carried between `plan` calls.
struct Run {
    /// Ply-1 placements in canonical movegen order; `root_index` indexes this.
    roots: Vec<Placement>,
    /// Best leaf score seen per ply-1 root — the back-up target (`i32::MIN` = unseen).
    root_best: Vec<i32>,
    frontier: BinaryHeap<Node>,
    /// Per-root best score at which each distinct state was enqueued (transposition).
    table: FxHashMap<StateKey, i32>,
    /// Nodes expanded so far this decision (against the node budget).
    expanded: u32,
    next_order: u64,
    /// State identity this run was seeded from (a `usize::MAX` root_index sentinel), to detect a
    /// stale in-flight run when `plan` is called for a new decision.
    fingerprint: StateKey,
}

/// A deterministic best-first graph-search planner with per-root transposition.
pub struct BestFirstPlanner {
    node_budget: u32,
    run: Option<Run>,
}

impl BestFirstPlanner {
    /// A planner with the given total node-expansion budget per decision.
    pub fn new(node_budget: u32) -> Self {
        Self {
            node_budget: node_budget.max(1),
            run: None,
        }
    }

    /// Hold-aware placements of `state`'s active piece, canonical movegen order (the
    /// same seam the beam and greedy planners use).
    fn placements(state: &SearchState) -> Vec<Placement> {
        let (w, h) = (state.board.width(), state.board.height());
        movegen::generate_with_hold(
            &state.board,
            &state.active,
            state.hold,
            state.queue.first().copied(),
            move |pt| movegen::spawn_piece(pt, w, h),
        )
    }

    /// Generate + score every child of `parent` (one per placement), in canonical
    /// order: `(child_state, score, acc_reward)`. Mirrors the beam's per-child scoring
    /// (classify the T-spin pre-lock, `commit_placement`, `value + acc`).
    fn children(
        parent: &SearchState,
        parent_acc: Reward,
        eval: &dyn Evaluator,
    ) -> Vec<(SearchState, i32, Reward)> {
        // The clear's attack depends on the chain *before* the move (the parent's).
        let ctx = EvalContext {
            combo: parent.combo,
            b2b: parent.b2b,
        };
        Self::placements(parent)
            .into_iter()
            .map(|placement| {
                let mut child = parent.clone();
                let t_spin = classify_t_spin(&placement.piece, &child.board);
                let lock = child.commit_placement(&placement);
                let (value, reward) = eval.evaluate_cols(&lock, &child.board, t_spin, ctx);
                let acc = parent_acc + reward;
                let score = (value + acc).0;
                (child, score, acc)
            })
            .collect()
    }

    /// Record a child: credit its root's back-up, then enqueue it unless the
    /// transposition table already holds an equal-or-better path to the same state.
    fn admit(run: &mut Run, child: SearchState, score: i32, acc: Reward, root_index: usize, depth: u8) {
        // `>`: the first maximum wins (canonical order), matching the beam / greedy.
        if score > run.root_best[root_index] {
            run.root_best[root_index] = score;
        }
        let key = StateKey::of(&child, root_index);
        if run.table.get(&key).is_some_and(|&best| best >= score) {
            return; // an equal-or-better path to this state is already enqueued
        }
        run.table.insert(key, score);
        run.frontier.push(Node {
            state: child,
            acc_reward: acc,
            root_index,
            score,
            depth,
            order: run.next_order,
        });
        run.next_order += 1;
    }

    /// Seed the run: the ply-1 root placements become the depth-1 frontier, each its own
    /// `root_index`. `None` if the state has no legal placement (topped out).
    fn seed(&self, state: &SearchState, eval: &dyn Evaluator) -> Option<Run> {
        let roots = Self::placements(state);
        if roots.is_empty() {
            return None;
        }
        let mut run = Run {
            root_best: vec![i32::MIN; roots.len()],
            roots,
            frontier: BinaryHeap::new(),
            table: FxHashMap::default(),
            expanded: 0,
            next_order: 0,
            fingerprint: StateKey::of(state, usize::MAX),
        };
        // Score the roots from the decision point's chain (same as the beam's seed).
        for (i, (child, score, acc)) in Self::children(state, Reward(0), eval).into_iter().enumerate() {
            Self::admit(&mut run, child, score, acc, i, 1);
        }
        Some(run)
    }

    /// Expand up to `EXPAND_CHUNK` best nodes (or until the node budget / frontier is
    /// exhausted). A node at `max_depth`, or with an empty queue (no concrete next
    /// piece — speculation past the visible queue is left to a future revision), is a
    /// leaf: its score already credited its root, so it is simply dropped.
    fn expand_chunk(&self, run: &mut Run, eval: &dyn Evaluator, max_depth: u8) {
        let mut this_call = 0u32;
        while this_call < EXPAND_CHUNK && run.expanded < self.node_budget {
            let Some(node) = run.frontier.pop() else {
                break;
            };
            if node.depth >= max_depth || node.state.queue.is_empty() {
                continue; // leaf — already backed up
            }
            run.expanded += 1;
            this_call += 1;
            let child_depth = node.depth + 1;
            for (child, score, acc) in Self::children(&node.state, node.acc_reward, eval) {
                Self::admit(run, child, score, acc, node.root_index, child_depth);
            }
        }
    }

    /// The decision: the ply-1 root with the maximal backed-up score (first max wins).
    fn best_plan(run: &Run) -> Option<PlacementPlan> {
        let mut best_i = 0usize;
        let mut best_score = run.root_best[0];
        for (i, &score) in run.root_best.iter().enumerate().skip(1) {
            if score > best_score {
                best_score = score;
                best_i = i;
            }
        }
        Some(PlacementPlan {
            placement: run.roots[best_i].clone(),
            score: best_score,
        })
    }
}

impl Default for BestFirstPlanner {
    fn default() -> Self {
        Self::new(DEFAULT_NODE_BUDGET)
    }
}

impl Planner for BestFirstPlanner {
    fn plan(
        &mut self,
        state: &SearchState,
        eval: &dyn Evaluator,
        budget: SearchBudget,
    ) -> PlannerStep {
        // Drop a stale run from a previous, different decision.
        if let Some(run) = &self.run {
            if run.fingerprint != StateKey::of(state, usize::MAX) {
                self.run = None;
            }
        }

        let mut run = match self.run.take() {
            None => match self.seed(state, eval) {
                Some(run) => run,
                None => return PlannerStep::Done(None), // topped out
            },
            Some(run) => run,
        };

        self.expand_chunk(&mut run, eval, budget.max_depth);

        if run.expanded >= self.node_budget || run.frontier.is_empty() {
            let plan = Self::best_plan(&run);
            self.run = None;
            PlannerStep::Done(plan)
        } else {
            self.run = Some(run);
            PlannerStep::NeedMoreBudget
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::eval::LinearEvaluator;
    use crate::ai::search::GreedyPlanner;
    use crate::engine::{Engine, EngineConfig, InputFrame};

    /// Drive a planner to a single `Done` plan (mirrors `SearchPolicy::plan_best`).
    fn drive(
        planner: &mut dyn Planner,
        state: &SearchState,
        eval: &dyn Evaluator,
        budget: SearchBudget,
    ) -> Option<PlacementPlan> {
        for _ in 0..100_000 {
            match planner.plan(state, eval, budget) {
                PlannerStep::Done(plan) => return plan,
                PlannerStep::NeedMoreBudget => continue,
            }
        }
        panic!("planner never finished");
    }

    /// A real engine snapshot after the first spawn (hold + full queue present).
    fn engine_state(seed: u64) -> SearchState {
        let mut engine = Engine::new(EngineConfig::default(), seed);
        engine.step(InputFrame::default());
        SearchState::from_snapshot(&engine.snapshot()).expect("active piece present")
    }

    #[test]
    fn best_first_is_deterministic() {
        // The same state planned twice (depth 4, budget 2000) yields the identical plan.
        let state = engine_state(7);
        let eval = LinearEvaluator::default();
        let mut a = BestFirstPlanner::new(2000);
        let mut b = BestFirstPlanner::new(2000);
        let pa = drive(&mut a, &state, &eval, SearchBudget::beam(4)).unwrap();
        let pb = drive(&mut b, &state, &eval, SearchBudget::beam(4)).unwrap();
        assert_eq!(pa.placement.origin(), pb.placement.origin());
        assert_eq!(pa.placement.rotation(), pb.placement.rotation());
        assert_eq!(pa.placement.path, pb.placement.path);
        assert_eq!(pa.score, pb.score);
    }

    #[test]
    fn best_first_depth1_equals_greedy() {
        // At max_depth 1 every root is a leaf (no expansion), so the decision is the
        // greedy single-ply argmax — identical to GreedyPlanner (the seam-faithful gate).
        let state = engine_state(42);
        let eval = LinearEvaluator::default();
        let mut bf = BestFirstPlanner::new(2000);
        let mut greedy = GreedyPlanner::new();
        let bp = drive(&mut bf, &state, &eval, SearchBudget::beam(1)).unwrap();
        let gp = drive(&mut greedy, &state, &eval, SearchBudget::greedy()).unwrap();
        assert_eq!(bp.placement.origin(), gp.placement.origin());
        assert_eq!(bp.placement.rotation(), gp.placement.rotation());
        assert_eq!(bp.placement.path, gp.placement.path);
    }
}
