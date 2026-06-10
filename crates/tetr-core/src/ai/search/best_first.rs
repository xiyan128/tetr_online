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

use crate::ai::eval::{EvalContext, Evaluator, Reward};
use crate::ai::movegen::Placement;
use crate::ai::search::{
    best_root_plan, hold_placements, score_child, Planner, PlannerStep, RootKey, SearchBudget,
};
use crate::ai::state::SearchState;

/// Nodes expanded per `plan` call before yielding — the cooperative time-slice unit,
/// deliberately far below any real node budget (the shipped budget is 150, the bench
/// default 4000) so an incremental runner actually observes
/// [`PlannerStep::NeedMoreBudget`] instead of the whole search running in one call.
/// The blocking [`SearchPolicy::plan_best`](crate::ai::SearchPolicy) just loops until
/// `Done`, so slice size never changes a decision — only call granularity.
const EXPAND_CHUNK: u32 = 64;

/// Identity of a search state for transposition: same key ⇒ same future, so two paths
/// reaching it are interchangeable. **Per-root** (`root_index` is part of the key) so a
/// position shared by two ply-1 moves is kept once *per root* and credited correctly;
/// the state identity itself is the shared [`RootKey`] (the beam uses the same fields
/// for its stale-run check).
#[derive(Clone, PartialEq, Eq, Hash)]
struct StateKey {
    root_index: usize,
    key: RootKey,
}

impl StateKey {
    fn of(state: &SearchState, root_index: usize) -> Self {
        Self {
            root_index,
            key: RootKey::of(state),
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
///
/// The total node-expansion budget per decision comes from
/// [`SearchBudget::nodes`] (see [`SearchBudget::best_first`]) — the planner itself
/// only carries the in-flight run.
#[derive(Default)]
pub struct BestFirstPlanner {
    run: Option<Run>,
}

impl BestFirstPlanner {
    /// A fresh planner (no in-flight run). The node budget is supplied per call via
    /// [`SearchBudget::best_first`].
    pub fn new() -> Self {
        Self::default()
    }

    /// Generate + score every child of `parent` (one per placement), in canonical
    /// order: `(child_state, score, acc_reward)`. Each child is built + scored by the
    /// shared [`score_child`] (fork → classify pre-lock → `commit_placement` →
    /// `evaluate_cols`); `acc` folds this move's reward into the path total.
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
        hold_placements(parent)
            .into_iter()
            .map(|placement| {
                let (child, value, reward) = score_child(parent, &placement, eval, ctx);
                let acc = parent_acc + reward;
                let score = (value + acc).0;
                (child, score, acc)
            })
            .collect()
    }

    /// Record a child: credit its root's back-up, then enqueue it unless the
    /// transposition table already holds an equal-or-better path to the same state.
    fn admit(
        run: &mut Run,
        child: SearchState,
        score: i32,
        acc: Reward,
        root_index: usize,
        depth: u8,
    ) {
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
    fn seed(state: &SearchState, eval: &dyn Evaluator) -> Option<Run> {
        let roots = hold_placements(state);
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
        for (i, (child, score, acc)) in Self::children(state, Reward(0), eval)
            .into_iter()
            .enumerate()
        {
            Self::admit(&mut run, child, score, acc, i, 1);
        }
        Some(run)
    }

    /// Expand up to `EXPAND_CHUNK` best nodes (or until the node budget / frontier is
    /// exhausted). A node at `budget.max_depth`, or with an empty queue (no concrete
    /// next piece — speculation past the visible queue is left to a future revision),
    /// is a leaf: its score already credited its root, so it is simply dropped.
    fn expand_chunk(run: &mut Run, eval: &dyn Evaluator, budget: SearchBudget) {
        let mut this_call = 0u32;
        while this_call < EXPAND_CHUNK && !budget_spent(run.expanded, budget) {
            let Some(node) = run.frontier.pop() else {
                break;
            };
            if node.depth >= budget.max_depth || node.state.queue.is_empty() {
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
}

/// Whether `expanded` has consumed the decision's node budget (`nodes == 0` =
/// uncapped: only depth / frontier exhaustion terminate).
fn budget_spent(expanded: u32, budget: SearchBudget) -> bool {
    budget.nodes != 0 && expanded >= budget.nodes
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
            None => match Self::seed(state, eval) {
                Some(run) => run,
                None => return PlannerStep::Done(None), // topped out
            },
            Some(run) => run,
        };

        Self::expand_chunk(&mut run, eval, budget);

        if budget_spent(run.expanded, budget) || run.frontier.is_empty() {
            let plan = best_root_plan(&run.roots, &run.root_best);
            self.run = None;
            PlannerStep::Done(Some(plan))
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
    use crate::ai::search::{GreedyPlanner, PlacementPlan};
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
                PlannerStep::NeedMoreBudget => {}
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
        let mut a = BestFirstPlanner::new();
        let mut b = BestFirstPlanner::new();
        let budget = SearchBudget::best_first(2000, 4);
        let pa = drive(&mut a, &state, &eval, budget).unwrap();
        let pb = drive(&mut b, &state, &eval, budget).unwrap();
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
        let mut bf = BestFirstPlanner::new();
        let mut greedy = GreedyPlanner::new();
        let bp = drive(&mut bf, &state, &eval, SearchBudget::best_first(2000, 1)).unwrap();
        let gp = drive(&mut greedy, &state, &eval, SearchBudget::greedy()).unwrap();
        assert_eq!(bp.placement.origin(), gp.placement.origin());
        assert_eq!(bp.placement.rotation(), gp.placement.rotation());
        assert_eq!(bp.placement.path, gp.placement.path);
    }

    #[test]
    fn best_first_time_slices_at_the_production_budget() {
        // The cooperative-yield contract: EXPAND_CHUNK (the per-call slice) sits below
        // the shipped node budget, so the first `plan` call must yield NeedMoreBudget
        // rather than running the whole decision — and the final plan must not depend
        // on slice granularity (the sliced run equals a fresh full drive).
        let state = engine_state(7);
        let eval = LinearEvaluator::default();
        let budget = SearchBudget::best_first(150, 6);

        let mut sliced = BestFirstPlanner::new();
        assert!(
            matches!(
                sliced.plan(&state, &eval, budget),
                PlannerStep::NeedMoreBudget
            ),
            "a 150-node decision must span multiple {EXPAND_CHUNK}-node slices"
        );
        let sliced_plan = drive(&mut sliced, &state, &eval, budget).unwrap();

        let full_plan = drive(&mut BestFirstPlanner::new(), &state, &eval, budget).unwrap();
        assert_eq!(sliced_plan.placement.origin(), full_plan.placement.origin());
        assert_eq!(sliced_plan.placement.path, full_plan.placement.path);
        assert_eq!(sliced_plan.score, full_plan.score);
    }
}
