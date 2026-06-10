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
    best_root_plan, hold_placements, score_child, Mind, PlacementPlan, RootKey, ThinkProgress,
};
use crate::ai::state::SearchState;

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

/// The in-flight session carried between [`Mind::think`] calls.
struct Run {
    /// Ply-1 placements in canonical movegen order; `root_index` indexes this.
    /// Empty when the root state had no legal placement (topped out).
    roots: Vec<Placement>,
    /// Best leaf score seen per ply-1 root — the back-up target (`i32::MIN` = unseen).
    root_best: Vec<i32>,
    frontier: BinaryHeap<Node>,
    /// Per-root best score at which each distinct state was enqueued (transposition).
    table: FxHashMap<StateKey, i32>,
    /// Nodes expanded so far on this root (the [`Mind::nodes_expanded`] meter).
    expanded: u32,
    next_order: u64,
    /// State identity this run was seeded from — the [`Mind::reroot`] fingerprint
    /// that keeps the session live across calls and discards it on a new root.
    root: RootKey,
    /// Ply cap the run was seeded under; part of the root identity (a different
    /// cap is a different search).
    max_depth: u8,
}

/// A deterministic best-first graph-search [`Mind`] with per-root transposition.
///
/// Node-grain: [`Mind::think`] honors its quantum exactly, so the total
/// node-expansion budget per decision is entirely the caller's meter (see
/// [`SearchBudget::best_first`] and
/// [`think_to_completion`](crate::ai::search::think_to_completion)) — the mind
/// itself only carries the in-flight run.
#[derive(Default)]
pub struct BestFirstPlanner {
    run: Option<Run>,
}

impl BestFirstPlanner {
    /// A fresh planner (no in-flight run). The node budget is supplied by the
    /// caller's meter via [`SearchBudget::best_first`].
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

    /// Seed a fresh run for `state`: the ply-1 root placements become the depth-1
    /// frontier, each its own `root_index`. A topped-out state (no legal
    /// placement) seeds an *empty* run — the fingerprint still records it, so
    /// re-rooting at the same dead state stays a no-op.
    fn seed(state: &SearchState, eval: &dyn Evaluator, max_depth: u8) -> Run {
        let roots = hold_placements(state);
        let mut run = Run {
            root_best: vec![i32::MIN; roots.len()],
            roots,
            frontier: BinaryHeap::new(),
            table: FxHashMap::default(),
            expanded: 0,
            next_order: 0,
            root: RootKey::of(state),
            max_depth,
        };
        // Score the roots from the decision point's chain (same as the beam's seed).
        for (i, (child, score, acc)) in Self::children(state, Reward(0), eval)
            .into_iter()
            .enumerate()
        {
            Self::admit(&mut run, child, score, acc, i, 1);
        }
        run
    }

    /// Expand up to `quantum` best nodes (or until the frontier drains). A node at
    /// the run's depth cap, or with an empty queue (no concrete next piece —
    /// speculation past the visible queue is left to a future revision), is a
    /// leaf: its score already credited its root, so it is simply dropped without
    /// counting against the quantum.
    fn expand(run: &mut Run, quantum: u32, eval: &dyn Evaluator) {
        let mut spent = 0u32;
        while spent < quantum {
            let Some(node) = run.frontier.pop() else {
                break;
            };
            if node.depth >= run.max_depth || node.state.queue.is_empty() {
                continue; // leaf — already backed up
            }
            run.expanded += 1;
            spent += 1;
            let child_depth = node.depth + 1;
            for (child, score, acc) in Self::children(&node.state, node.acc_reward, eval) {
                Self::admit(run, child, score, acc, node.root_index, child_depth);
            }
        }
    }
}

impl Mind for BestFirstPlanner {
    fn reroot(&mut self, state: &SearchState, eval: &dyn Evaluator, max_depth: u8) {
        let root = RootKey::of(state);
        if self
            .run
            .as_ref()
            .is_some_and(|run| run.root == root && run.max_depth == max_depth)
        {
            return; // already rooted here: the in-flight search continues
        }
        self.run = Some(Self::seed(state, eval, max_depth));
    }

    fn think(&mut self, quantum: u32, eval: &dyn Evaluator) -> ThinkProgress {
        let Some(run) = self.run.as_mut() else {
            return ThinkProgress::Exhausted; // never rooted: nothing to think about
        };
        Self::expand(run, quantum, eval);
        if run.frontier.is_empty() {
            ThinkProgress::Exhausted
        } else {
            ThinkProgress::Working
        }
    }

    fn best(&self) -> Option<PlacementPlan> {
        let run = self.run.as_ref()?;
        if run.roots.is_empty() {
            return None; // topped out: no legal placement existed at the root
        }
        Some(best_root_plan(&run.roots, &run.root_best))
    }

    fn nodes_expanded(&self) -> u32 {
        self.run.as_ref().map_or(0, |run| run.expanded)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::eval::LinearEvaluator;
    use crate::ai::search::{think_to_completion, GreedyPlanner, SearchBudget};
    use crate::engine::{Engine, EngineConfig, InputFrame};

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
        let pa = think_to_completion(&mut a, &state, &eval, budget).unwrap();
        let pb = think_to_completion(&mut b, &state, &eval, budget).unwrap();
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
        let bp =
            think_to_completion(&mut bf, &state, &eval, SearchBudget::best_first(2000, 1)).unwrap();
        let gp = think_to_completion(&mut greedy, &state, &eval, SearchBudget::greedy()).unwrap();
        assert_eq!(bp.placement.origin(), gp.placement.origin());
        assert_eq!(bp.placement.rotation(), gp.placement.rotation());
        assert_eq!(bp.placement.path, gp.placement.path);
    }

    #[test]
    fn quantum_granularity_never_changes_the_decision() {
        // The session contract a cooperative venue relies on: thinking the shipped
        // 150-node budget in 16-node frame slices reaches the identical decision as
        // one blocking 150-node call — the quantum only chooses suspension points.
        let state = engine_state(7);
        let eval = LinearEvaluator::default();
        let budget = SearchBudget::best_first(150, 6);

        let mut fine = BestFirstPlanner::new();
        fine.reroot(&state, &eval, budget.max_depth);
        assert_eq!(
            fine.think(16, &eval),
            ThinkProgress::Working,
            "a 150-node decision must span multiple 16-node slices"
        );
        while fine.nodes_expanded() < budget.nodes {
            let quantum = (budget.nodes - fine.nodes_expanded()).min(16);
            if fine.think(quantum, &eval) == ThinkProgress::Exhausted {
                break;
            }
        }
        let fine_plan = fine.best().unwrap();

        let coarse_plan =
            think_to_completion(&mut BestFirstPlanner::new(), &state, &eval, budget).unwrap();
        assert_eq!(fine_plan.placement.origin(), coarse_plan.placement.origin());
        assert_eq!(fine_plan.placement.path, coarse_plan.placement.path);
        assert_eq!(fine_plan.score, coarse_plan.score);
    }

    #[test]
    fn best_is_anytime_valid_from_seeding_onward() {
        // The anytime contract: `best()` is a legal root placement immediately after
        // reroot (before any think), and after every quantum thereafter.
        let state = engine_state(11);
        let eval = LinearEvaluator::default();
        let legal = hold_placements(&state);
        let is_legal = |plan: &PlacementPlan| {
            legal.iter().any(|p| {
                p.origin() == plan.placement.origin()
                    && p.rotation() == plan.placement.rotation()
                    && p.path == plan.placement.path
            })
        };

        let mut mind = BestFirstPlanner::new();
        mind.reroot(&state, &eval, 6);
        let seeded = mind.best().expect("best is valid right after seeding");
        assert!(is_legal(&seeded), "seeded best is a real root placement");

        for _ in 0..20 {
            let progress = mind.think(16, &eval);
            let current = mind.best().expect("best stays valid while thinking");
            assert!(is_legal(&current), "anytime best is a real root placement");
            if progress == ThinkProgress::Exhausted {
                break;
            }
        }
    }

    #[test]
    fn rerooting_the_same_state_continues_the_run() {
        // The fingerprint contract: re-asserting the same (state, depth) preserves
        // the in-flight search; a different state — or a different depth cap —
        // discards it and re-seeds.
        let state = engine_state(7);
        let eval = LinearEvaluator::default();
        let mut mind = BestFirstPlanner::new();

        mind.reroot(&state, &eval, 6);
        mind.think(32, &eval);
        assert_eq!(mind.nodes_expanded(), 32);

        mind.reroot(&state, &eval, 6);
        assert_eq!(
            mind.nodes_expanded(),
            32,
            "same (state, depth): the run must continue, not restart"
        );

        mind.reroot(&state, &eval, 4);
        assert_eq!(
            mind.nodes_expanded(),
            0,
            "a different depth cap is a different search"
        );

        mind.think(8, &eval);
        mind.reroot(&engine_state(42), &eval, 4);
        assert_eq!(
            mind.nodes_expanded(),
            0,
            "a different root state discards the stale run"
        );
    }
}
