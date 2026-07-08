//! The deterministic, batch-shaped beam Tier-2 planner (Colder Clear, STEP 2).
//!
//! A CC2-style **beam search** behind the same [`Mind`] session seam the greedy
//! Tier-1 planner implements, so it drops in with no controller / policy / runner
//! change. See `BEAM.md` (this directory) for the design record; the load-bearing
//! pins:
//!
//! 1. **Determinism (BEAM.md §1).** Zero RNG, no clock. The only tie-breaker is
//!    movegen's canonical placement order: children are pushed in
//!    `(parent-order, movegen-order)` and ranked by score descending with that
//!    enumeration index as the ascending tie-break, so a tie resolves to the
//!    earlier-enumerated node. Back-up uses `>` (not `>=`) so the **first** maximum
//!    wins, mirroring `greedy.rs`'s rule.
//! 2. **Hold-aware transition (BEAM.md §3).** A node forks the [`SearchState`] and
//!    advances through a [`Placement`] with [`SearchState::commit_placement`], the
//!    Step-0 transition that models a hold swap and deals the bag exactly once.
//! 3. **One generation at a time (BEAM.md §4/§7).** Every child of a generation
//!    is scored before any child becomes the next expandable frontier. The planner
//!    may pause within a generation, but it publishes the next frontier only when
//!    the whole generation is complete.
//! 4. **Depth-1 == greedy (BEAM.md §8).** The first generation scores each
//!    placement as a one-ply search does (clone, classify pre-lock, lock,
//!    `value + reward`), so a `max_depth == 1` beam reproduces the greedy
//!    decision (pinned against `SearchBudget::single_ply` best-first).
//! 5. **Transposition pruning (opt-in, [`BeamPlanner::transposing`]).** Before
//!    width truncation, nodes that reach the SAME future state under the same
//!    ply-1 root collapse to their highest-scoring derivation — a sound
//!    max-merge (identical states share identical futures; only accumulated
//!    reward differs), so the width is spent on distinct futures. The dedup
//!    key includes the bag remainder (different bags ⇒ different speculative
//!    futures) and never crosses roots (each root needs its own backed-up
//!    value). `BeamPlanner::new` beams are byte-for-byte unchanged — every
//!    recorded beam baseline stays reproducible.
//!
//! As a session the beam is generation-staged but node-sliced: [`Mind::think`]
//! expands up to `quantum` frontier nodes while accumulating a staged next
//! generation. [`Mind::best`] remains the backed-up ply-1 argmax from completed
//! generations only; partial generation scores commit atomically when the
//! generation finishes.

use rustc_hash::FxHashSet;

use crate::ai::eval::{EvalContext, Evaluator, Leaf, Reward, Value};
use crate::ai::movegen::Placement;
use crate::ai::search::{
    Mind, PlacementPlan, RootKey, ThinkProgress, best_root_plan, commit_child, hold_placements,
};
use crate::ai::state::SearchState;
use crate::engine::{LockOutcome, PieceType, TSpinKind};

/// Multiplicative pessimism applied to a speculative branch's *reward* contribution
/// per speculative ply (BEAM.md §5): we cannot rely on a piece we have not seen, so
/// its reward is discounted while the resulting board's static [`Value`] is kept
/// whole (the board is real regardless of which bag piece arrives).
const SPEC_DECAY: f32 = 0.75;

/// One node in the beam frontier: a forked search state plus the bookkeeping the
/// back-up and the final ply-1 decision need (BEAM.md §2).
#[derive(Clone)]
struct BeamNode {
    /// The forked, owned search state at this node. Cheap to clone (no engine, no
    /// RNG, no timers).
    state: SearchState,
    /// Sum of per-move [`Reward`]s along the path from the root to this node
    /// (Cold Clear's reward-folds-into-value design). Folded into a leaf's static
    /// [`Value`] at scoring time.
    acc_reward: Reward,
    /// Index into the ply-1 root frontier this node descends from. The whole subtree
    /// carries the *same* `root_index`, so the best leaf can credit the ply-1 move
    /// that owns it.
    root_index: usize,
    /// A per-branch reward discount carried from speculation (BEAM.md §5). `1.0`
    /// until the branch crosses into speculative plies; multiplied by [`SPEC_DECAY`]
    /// at each speculative expansion so deeper speculative rewards count for less.
    spec_weight: f32,
}

/// One not-yet-scored child of the generation being expanded: the forked state, its
/// lock results, and the node bookkeeping, in **one** struct — so the batch inputs
/// and the post-score node build read the same element and can never fall out of
/// lockstep.
struct PendingChild {
    state: SearchState,
    lock: LockOutcome,
    t_spin: Option<TSpinKind>,
    /// The chain context the child scores under (the PARENT's pre-placement state).
    ctx: EvalContext,
    /// A score already computed and shared by the speculative dedup (one board-only
    /// evaluation covers a placement's whole bag fan). `None` = evaluate this child
    /// in [`score_pending_into`]'s batch — the single scoring funnel either way.
    ///
    /// [`score_pending_into`]: BeamPlanner::score_pending_into
    score: Option<(Value, Reward)>,
    /// Which ply-1 root this descends from.
    root_index: usize,
    /// The parent's accumulated path reward; this move's reward folds in at scoring.
    parent_acc: Reward,
    /// The branch's speculative reward discount (`1.0` on concrete branches).
    spec_weight: f32,
}

/// The in-flight session carried on the planner between [`Mind::think`] calls
/// (BEAM.md §4).
struct BeamRun {
    /// The ply-1 placements, in canonical movegen order. `root_index` indexes this.
    /// Empty when the root state had no legal placement (topped out).
    roots: Vec<Placement>,
    /// Best leaf score seen so far per root (the back-up target). `i32::MIN` = unseen.
    root_best: Vec<i32>,
    /// The current frontier (already truncated to `<= beam_width`).
    frontier: Vec<BeamNode>,
    /// A generation currently being expanded across `think()` calls. `None` means
    /// `frontier` is the next completed generation to expand.
    generation: Option<GenerationWork>,
    /// Plies expanded so far (root seeding = depth 1).
    depth: u8,
    /// Identity of the state this run was seeded from — the [`Mind::reroot`]
    /// fingerprint (the shared [`RootKey`]; compared by value, exact, no hashing).
    root_key: RootKey,
    /// Ply cap the run was seeded under; part of the root identity.
    max_depth: u8,
    /// Frontier nodes expanded so far (the [`Mind::nodes_expanded`] meter; the
    /// beam's *termination* is width × depth, never this count).
    expanded: u32,
}

/// One generation's staged expansion state. Children are scored as parents are
/// processed, but the resulting frontier and backed-up root scores are not
/// published to [`BeamRun`] until every parent in the generation has been consumed.
struct GenerationWork {
    /// Completed frontier from the prior generation, kept in canonical order.
    parents: Vec<BeamNode>,
    /// Next parent index to expand.
    next_parent: usize,
    /// Scored children for the next generation, still in canonical enumeration order.
    nodes: Vec<Option<BeamNode>>,
    /// Lightweight `(score, node-index)` ranking entries for `nodes`.
    ranked: Vec<(i32, u32)>,
    /// Root back-ups including this in-flight generation's scored children. Staged
    /// separately so `best()` stays generation-grain while this work is partial.
    root_best: Vec<i32>,
}

/// A deterministic, batch-shaped, time-sliced beam planner (BEAM.md §2/§4/§5/§6).
pub struct BeamPlanner {
    /// How many nodes survive truncation each generation.
    beam_width: usize,
    /// Whether to speculate past the visible queue over the 7-bag remainder
    /// (BEAM.md §5). On by default; the bench can toggle it.
    speculate: bool,
    /// Transposition pruning before truncation (header pin 5). Off by default:
    /// recorded `new()` baselines stay byte-identical; TP variants are new
    /// registered names.
    transpose: bool,
    /// Optional root pre-filter: given the decision state and its full legal
    /// placement set, returns the subset this run may search (e.g. a learned
    /// policy's top-m — the guided-beam vehicle). `None` = every root, byte-
    /// identical to the recorded baselines. Must be deterministic per state
    /// (re-rooting at the same state assumes the same roots).
    root_filter: Option<RootFilter>,
    /// In-flight search, `None` between decisions. Reset on a new root state.
    run: Option<BeamRun>,
}

/// See [`BeamPlanner::with_root_filter`].
pub type RootFilter = Box<dyn Fn(&SearchState, Vec<Placement>) -> Vec<Placement> + Send + Sync>;

impl BeamPlanner {
    /// A beam planner of the given width, with bag speculation **on** (the default).
    pub fn new(beam_width: usize) -> Self {
        Self {
            beam_width: beam_width.max(1),
            speculate: true,
            transpose: false,
            root_filter: None,
            run: None,
        }
    }

    /// A transposition-pruned beam (header pin 5): equal per-root future states
    /// collapse to their best derivation before truncation, so width buys
    /// distinct futures. Ported from the 2026-06-12 codex-agent worktree
    /// (design verified: per-root max-merge with the bag in the key).
    pub fn transposing(beam_width: usize) -> Self {
        Self {
            transpose: true,
            ..Self::new(beam_width)
        }
    }

    /// Toggle 7-bag speculation past the visible queue (BEAM.md §5). Consuming
    /// builder so a factory can write `BeamPlanner::new(w).with_speculation(false)`.
    pub fn with_speculation(mut self, speculate: bool) -> Self {
        self.speculate = speculate;
        self
    }

    /// Restrict each run's roots through `filter` (the policy-guided beam: a
    /// learned prior picks which ply-1 placements deserve search). A filter
    /// that returns an empty set is ignored for that state (all roots search)
    /// — restriction must never manufacture a resignation.
    pub fn with_root_filter(mut self, filter: RootFilter) -> Self {
        self.root_filter = Some(filter);
        self
    }

    /// The current run's per-root backed-up scores: each ply-1 placement paired
    /// with the best leaf score its subtree has achieved so far (`i32::MIN` =
    /// no scored descendant yet). Empty between decisions or on a topped-out
    /// root. This is the search-improved read of the root decision — the
    /// distribution an expert-iteration data pipeline derives its policy
    /// targets from.
    pub fn root_scores(&self) -> impl Iterator<Item = (&Placement, i32)> {
        self.run
            .iter()
            .flat_map(|run| run.roots.iter().zip(run.root_best.iter().copied()))
    }

    /// Seed a fresh run for `state`: form the ply-1 root children (depth 1), score
    /// them as one batch, and build the initial frontier. A topped-out state (no
    /// legal placement) seeds an *empty* run — the fingerprint still records it,
    /// so re-rooting at the same dead state stays a no-op.
    fn seed(&self, state: &SearchState, eval: &dyn Evaluator, max_depth: u8) -> BeamRun {
        let roots = {
            let all = hold_placements(state);
            match &self.root_filter {
                Some(f) if !all.is_empty() => {
                    let picked = f(state, all.clone());
                    if picked.is_empty() { all } else { picked }
                }
                _ => all,
            }
        };

        // Fork + transition each root child in canonical order, through the shared
        // fork → classify pre-lock → commit helper. Root children score with the
        // decision point's chain — the combo / B2B state before the move — exactly
        // as the per-generation expansion does.
        let root_ctx = EvalContext {
            combo: state.combo,
            b2b: state.b2b,
        };
        let pending: Vec<PendingChild> = roots
            .iter()
            .enumerate()
            .map(|(i, placement)| {
                let (child, lock, t_spin) = commit_child(state, placement);
                PendingChild {
                    state: child,
                    lock,
                    t_spin,
                    ctx: root_ctx,
                    score: None,
                    root_index: i,
                    parent_acc: Reward(0),
                    spec_weight: 1.0,
                }
            })
            .collect();

        let mut run = BeamRun {
            root_best: vec![i32::MIN; roots.len()],
            roots,
            frontier: Vec::new(),
            generation: None,
            depth: 1,
            root_key: RootKey::of(state),
            max_depth,
            expanded: 0,
        };
        Self::score_into_frontier(&mut run, pending, eval, self.beam_width, self.transpose);
        run
    }

    /// Start staging the next generation by moving the completed frontier out of the
    /// run. Root back-ups are copied so partial scores do not leak through
    /// [`Mind::best`] until the generation is fully published.
    fn start_generation(run: &mut BeamRun) {
        run.generation = Some(GenerationWork {
            parents: std::mem::take(&mut run.frontier),
            next_parent: 0,
            nodes: Vec::new(),
            ranked: Vec::new(),
            root_best: run.root_best.clone(),
        });
    }

    /// Expand one parent frontier node into staged, already-scored next-generation
    /// nodes. The caller owns the node meter; this function only performs the work.
    fn expand_parent(
        parent: &BeamNode,
        root_best: &mut [i32],
        nodes: &mut Vec<Option<BeamNode>>,
        ranked: &mut Vec<(i32, u32)>,
        eval: &dyn Evaluator,
        speculate: bool,
    ) {
        let mut pending: Vec<PendingChild> = Vec::new();

        if parent.state.dead {
            // A dead branch is terminal: its DEATH_SCORE back-up already credited its
            // root; it expands to nothing (and never speculates).
            return;
        }
        if parent.state.queue.is_empty() {
            // Past the visible queue: speculate over the bag if enabled, else this
            // node is terminal (no concrete next piece to advance the active). A
            // terminal node contributes no children; its `root_best` was already
            // recorded when it entered the frontier, so the back-up keeps it.
            if speculate {
                Self::expand_speculative(parent, &mut pending, eval);
            }
        } else {
            // Concrete: every placement of the parent's active piece. The trailing
            // `commit_placement` advances the active from the (non-empty) queue.
            // Each child is scored with the PARENT's pre-placement chain (the combo /
            // B2B state before this move, which is what its clear's attack depends on).
            let parent_ctx = EvalContext {
                combo: parent.state.combo,
                b2b: parent.state.b2b,
            };
            for placement in hold_placements(&parent.state) {
                let (child, lock, t_spin) = commit_child(&parent.state, &placement);
                pending.push(PendingChild {
                    state: child,
                    lock,
                    t_spin,
                    ctx: parent_ctx,
                    score: None,
                    root_index: parent.root_index,
                    parent_acc: parent.acc_reward,
                    spec_weight: parent.spec_weight,
                });
            }
        }

        Self::score_pending_into(root_best, pending, eval, nodes, ranked);
    }

    /// Score `pending` children — reusing any score the speculative dedup already
    /// computed and evaluating the rest as one batch through
    /// [`evaluate_leaves`](Evaluator::evaluate_leaves) (BEAM.md §7) — then append
    /// their nodes in canonical order and compact rank entries. `root_best` may be
    /// the live run backup (seeding) or a generation-staged backup (sliced
    /// expansion).
    fn score_pending_into(
        root_best: &mut [i32],
        pending: Vec<PendingChild>,
        eval: &dyn Evaluator,
        nodes: &mut Vec<Option<BeamNode>>,
        ranked: &mut Vec<(i32, u32)>,
    ) {
        if pending.is_empty() {
            return;
        }

        // Evaluate the children the speculative dedup did not already score, as
        // ONE batch through the trait's batch seam — a learned backend fuses the
        // group into a single forward; the default loops `evaluate_cols`, which
        // is exactly the old per-child hot path.
        let unscored: Vec<Leaf<'_>> = pending
            .iter()
            .filter(|p| p.score.is_none())
            .map(|p| Leaf {
                state: &p.state,
                lock: &p.lock,
                t_spin: p.t_spin,
                ctx: p.ctx,
            })
            .collect();
        // `evaluate_leaves` returns an empty Vec on empty input, so the all-scored
        // case yields an empty iterator that the loop below never advances.
        let mut fresh = eval.evaluate_leaves(&unscored).into_iter();

        nodes.reserve(pending.len());
        ranked.reserve(pending.len());
        for p in pending.into_iter() {
            let (value, reward) = p
                .score
                .unwrap_or_else(|| fresh.next().expect("one fresh score per unscored child"));
            // Death is absolute: override the eval (the truncated board it
            // scored is a death remnant, not a position).
            let (value, reward) = if p.state.dead {
                (crate::ai::eval::Value(super::DEATH_SCORE), Reward(0))
            } else {
                (value, reward)
            };
            // Discount this move's reward by the branch's speculative weight; the
            // board Value is kept whole (the resulting board is real regardless). A
            // concrete branch (weight 1.0) keeps the integer reward exactly, with no
            // f32 round-trip.
            let weighted_reward = if p.spec_weight == 1.0 {
                reward
            } else {
                Reward((reward.0 as f32 * p.spec_weight).round() as i32)
            };
            let acc = p.parent_acc + weighted_reward;
            let score = (value + acc).0;
            // `>`: keep the first maximum (canonical order), matching greedy.
            if score > root_best[p.root_index] {
                root_best[p.root_index] = score;
            }
            debug_assert!(u32::try_from(nodes.len()).is_ok());
            ranked.push((score, nodes.len() as u32));
            nodes.push(Some(BeamNode {
                state: p.state,
                acc_reward: acc,
                root_index: p.root_index,
                spec_weight: p.spec_weight,
            }));
        }
    }

    /// Rank/dedup/truncate a completed generation into the next frontier.
    fn ranked_frontier(
        mut nodes: Vec<Option<BeamNode>>,
        mut ranked: Vec<(i32, u32)>,
        beam_width: usize,
        transpose: bool,
    ) -> Vec<BeamNode> {
        // Rank by score descending, ties by canonical index ascending — the exact
        // order the old stable descending node-sort produced (header pin 1).
        ranked.sort_unstable_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));

        // Gather the top `beam_width` survivors in rank order, collapsing
        // transpositions as we go — first-seen wins, i.e. the highest-scoring
        // derivation of an equal `(root, future-state, bag)` (header pin 5). This is
        // identical to the old dedup-then-truncate, but keying stops the moment the
        // frontier is full instead of hashing the whole discarded tail. Each survivor
        // moves out of `nodes` exactly once.
        let mut frontier: Vec<BeamNode> = Vec::with_capacity(beam_width.min(nodes.len()));
        let mut seen = transpose.then(FxHashSet::default);
        for &(_, i) in &ranked {
            let i = i as usize;
            if let Some(seen) = seen.as_mut() {
                let node = nodes[i].as_ref().expect("ranking indexes every node once");
                if !seen.insert((node.root_index, RootKey::of(&node.state), node.state.bag)) {
                    continue;
                }
            }
            frontier.push(nodes[i].take().expect("each survivor index is unique"));
            if frontier.len() == beam_width {
                break;
            }
        }
        frontier
    }

    /// Score a generation's `pending` children and fold the completed generation
    /// into the next frontier. Used by [`seed`](Self::seed), which is intentionally
    /// still one immediate generation so `best()` is valid right after `reroot()`.
    fn score_into_frontier(
        run: &mut BeamRun,
        pending: Vec<PendingChild>,
        eval: &dyn Evaluator,
        beam_width: usize,
        transpose: bool,
    ) {
        let mut nodes: Vec<Option<BeamNode>> = Vec::with_capacity(pending.len());
        let mut ranked: Vec<(i32, u32)> = Vec::with_capacity(pending.len());
        Self::score_pending_into(&mut run.root_best, pending, eval, &mut nodes, &mut ranked);
        run.frontier = Self::ranked_frontier(nodes, ranked, beam_width, transpose);
    }

    /// Speculative expansion of an empty-queue `parent` (BEAM.md §5): each placement
    /// of the parent's active piece is committed **once** — the lock and its clears
    /// are the same whatever the bag deals next — then fanned across the bag-legal
    /// next pieces via the state's speculative deal, which is the only thing a
    /// continuation changes (the spawn still re-checks block-out per piece). For a
    /// [`board_only`](Evaluator::board_only) evaluator the committed board is also
    /// *scored* once here, and the whole fan shares that score; a state-reading
    /// evaluator leaves `score` empty and is batched per child downstream. The child
    /// reward weight is scaled by [`SPEC_DECAY`] so deeper speculative rewards count
    /// for less.
    ///
    /// The hold-aware transition is the same `used_hold` swap as the concrete path
    /// (shared via `apply_placement`); an empty-queue node only offers `used_hold`
    /// placements when hold is occupied (movegen's `hold.or(queue_front)`), so the
    /// swap's empty-hold funding pop never fires here.
    ///
    /// Enumeration stays piece-major — all placements under one next piece, then
    /// the next piece — so ranking order, and therefore every tie-break downstream,
    /// is identical to fanning naively (pinned by
    /// `speculative_share_matches_naive_fan`).
    ///
    /// No RNG, no expectimax average — every bag-legal piece is enumerated and
    /// beam-width truncation prunes the fan-out, keeping the planner deterministic.
    fn expand_speculative(
        parent: &BeamNode,
        pending: &mut Vec<PendingChild>,
        eval: &dyn Evaluator,
    ) {
        let placements = hold_placements(&parent.state);
        let child_weight = parent.spec_weight * SPEC_DECAY;
        let parent_ctx = EvalContext {
            combo: parent.state.combo,
            b2b: parent.state.b2b,
        };

        /// One placement, committed against the parent, before the bag fan.
        struct Committed {
            base: SearchState,
            lock: LockOutcome,
            t_spin: Option<TSpinKind>,
            score: Option<(Value, Reward)>,
        }
        let committed: Vec<Committed> = placements
            .iter()
            .map(|placement| {
                let mut base = parent.state.clone();
                // Classify against the pre-lock board, like the concrete path.
                let t_spin = crate::engine::classify_t_spin(&placement.piece, &base.board);
                let lock = base.apply_placement(placement);
                let score = eval
                    .board_only()
                    .then(|| eval.evaluate_cols(&lock, base.board.view(), t_spin, parent_ctx));
                Committed {
                    base,
                    lock,
                    t_spin,
                    score,
                }
            })
            .collect();

        for next_piece in PieceType::all() {
            if !parent.state.bag.contains(next_piece) {
                continue;
            }
            for c in &committed {
                let mut child = c.base.clone();
                child.deal_speculative(next_piece);
                pending.push(PendingChild {
                    state: child,
                    lock: c.lock.clone(),
                    t_spin: c.t_spin,
                    ctx: parent_ctx,
                    score: c.score,
                    root_index: parent.root_index,
                    parent_acc: parent.acc_reward,
                    spec_weight: child_weight,
                });
            }
        }
    }
}

/// Whether `run` can expand no further: the depth cap is met or the frontier is
/// exhausted (every surviving line is terminal). The back-up `root_best` already
/// holds the best score each ply-1 root ever achieved, so [`Mind::best`] is
/// correct even if a root's descendants were all pruned (BEAM.md §4).
fn exhausted(run: &BeamRun) -> bool {
    run.generation.is_none() && (run.depth >= run.max_depth || run.frontier.is_empty())
}

impl Mind for BeamPlanner {
    fn reroot(&mut self, state: &SearchState, eval: &dyn Evaluator, max_depth: u8) {
        // A different root state — or a different depth cap — is a different
        // search: discard the stale run so a fresh decision never resumes the
        // previous decision's frontier. Seeding is itself generation 1 (== greedy
        // when max_depth == 1, BEAM.md §8).
        let root_key = RootKey::of(state);
        if self
            .run
            .as_ref()
            .is_some_and(|run| run.root_key == root_key && run.max_depth == max_depth)
        {
            return; // already rooted here: the in-flight search continues
        }
        self.run = Some(self.seed(state, eval, max_depth));
    }

    /// **Generation-staged, node-sliced**: spend up to `quantum` parent frontier
    /// nodes, but publish the next frontier only after the whole generation has
    /// been consumed. This keeps generation-level semantics while making the
    /// interactive runner's node quantum meaningful.
    fn think(&mut self, quantum: u32, eval: &dyn Evaluator) -> ThinkProgress {
        let (beam_width, speculate, transpose) = (self.beam_width, self.speculate, self.transpose);
        let Some(run) = self.run.as_mut() else {
            return ThinkProgress::Exhausted; // never rooted: nothing to think about
        };
        if exhausted(run) {
            return ThinkProgress::Exhausted;
        }
        if quantum == 0 {
            return ThinkProgress::Working;
        }

        let mut spent = 0u32;
        while spent < quantum && !exhausted(run) {
            if run.generation.is_none() {
                Self::start_generation(run);
            }

            let mut generation = run.generation.take().expect("generation is started above");
            while spent < quantum && generation.next_parent < generation.parents.len() {
                let parent = &generation.parents[generation.next_parent];
                Self::expand_parent(
                    parent,
                    &mut generation.root_best,
                    &mut generation.nodes,
                    &mut generation.ranked,
                    eval,
                    speculate,
                );
                generation.next_parent += 1;
                run.expanded += 1;
                spent += 1;
            }

            if generation.next_parent == generation.parents.len() {
                run.root_best = generation.root_best;
                run.frontier = Self::ranked_frontier(
                    generation.nodes,
                    generation.ranked,
                    beam_width,
                    transpose,
                );
                run.depth += 1;
            } else {
                run.generation = Some(generation);
                break;
            }
        }

        if exhausted(run) {
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
    use crate::ai::movegen;
    use crate::ai::search::{BestFirstPlanner, SearchBudget};
    use crate::engine::{Board, CellKind, Engine, EngineConfig, InputFrame};

    fn linear() -> LinearEvaluator {
        LinearEvaluator::default()
    }

    /// Drive a mind to its final plan in one blocking call (the direct-drive venue).
    fn drive(
        mind: &mut dyn Mind,
        state: &SearchState,
        eval: &dyn Evaluator,
        budget: SearchBudget,
    ) -> Option<PlacementPlan> {
        crate::ai::search::think_to_completion(mind, state, eval, budget)
    }

    /// The Tetris-well fixture greedy's tests use: a 4-wide board, cols 0-2 filled
    /// four rows high, an empty 1-wide well at col 3 (a vertical I clears a Tetris).
    fn tetris_well_state() -> SearchState {
        let mut board = Board::new(4, 12);
        for y in 0..4 {
            for x in 0..3 {
                board.set(x, y, CellKind::Some(PieceType::O));
            }
        }
        let active = movegen::spawn_piece(PieceType::I, 4, 12);
        SearchState::for_test(board, active, None, std::iter::empty())
    }

    #[test]
    fn beam_is_deterministic() {
        // The same state planned twice (depth 3) yields the identical ply-1 plan.
        let state = engine_snapshot_state(7);
        let mut a = BeamPlanner::new(16);
        let mut b = BeamPlanner::new(16);
        let pa = drive(&mut a, &state, &linear(), SearchBudget::beam(3)).unwrap();
        let pb = drive(&mut b, &state, &linear(), SearchBudget::beam(3)).unwrap();
        assert_eq!(pa.placement.origin(), pb.placement.origin());
        assert_eq!(pa.placement.rotation(), pb.placement.rotation());
        assert_eq!(pa.placement.path, pb.placement.path);
        assert_eq!(pa.score, pb.score);
    }

    #[test]
    fn beam_depth1_equals_greedy_on_tetris_well() {
        // depth-1 beam reproduces greedy's decision on the crafted fixture (the
        // seam-faithful gate, BEAM.md §8).
        let state = tetris_well_state();
        let mut beam = BeamPlanner::new(16);
        let mut bf = BestFirstPlanner::new();
        let bp = drive(&mut beam, &state, &linear(), SearchBudget::beam(1)).unwrap();
        let gp = drive(&mut bf, &state, &linear(), SearchBudget::single_ply()).unwrap();
        assert_eq!(bp.placement.origin(), gp.placement.origin());
        assert_eq!(bp.placement.rotation(), gp.placement.rotation());
        assert_eq!(bp.placement.path, gp.placement.path);
        assert_eq!(bp.score, gp.score);
    }

    #[test]
    fn beam_depth1_equals_greedy_on_engine_snapshot() {
        // And on a real engine snapshot (a non-crafted position with hold + queue).
        let state = engine_snapshot_state(42);
        let mut beam = BeamPlanner::new(16);
        let mut bf = BestFirstPlanner::new();
        let bp = drive(&mut beam, &state, &linear(), SearchBudget::beam(1)).unwrap();
        let gp = drive(&mut bf, &state, &linear(), SearchBudget::single_ply()).unwrap();
        assert_eq!(bp.placement.origin(), gp.placement.origin());
        assert_eq!(bp.placement.rotation(), gp.placement.rotation());
        assert_eq!(bp.placement.path, gp.placement.path);
        assert_eq!(bp.score, gp.score);
    }

    #[test]
    fn beam_returns_none_when_topped() {
        // A board filled to the brim leaves the active piece with no legal resting
        // pose distinct from a top-out; if movegen yields no placements, the beam
        // reports `Done(None)`. Build a board where the active piece overlaps filled
        // cells everywhere it could rest by filling the whole playfield.
        let mut board = Board::new(4, 4);
        for y in 0..4 {
            for x in 0..4 {
                board.set(x, y, CellKind::Some(PieceType::O));
            }
        }
        let active = movegen::spawn_piece(PieceType::I, 4, 4);
        let state = SearchState::for_test(board, active, None, std::iter::empty());

        // Movegen still emits placements if the spawn pose itself rests; to force the
        // empty case we assert on the actual movegen output and only require the beam
        // to mirror it. If placements exist, the beam must return Some; if not, None.
        let placements = hold_placements(&state);
        let mut beam = BeamPlanner::new(16);
        let plan = drive(&mut beam, &state, &linear(), SearchBudget::beam(2));
        assert_eq!(plan.is_some(), !placements.is_empty());
    }

    #[test]
    fn beam_returns_none_on_truly_empty_placements() {
        // A direct check of the topped-out contract: when there is genuinely no
        // placement, `plan` is `Done(None)`. We construct that by stubbing a state
        // whose movegen returns nothing — a fully filled board with the active piece
        // unable to rest anywhere new. We assert via the same empty-placement branch
        // the planner takes, using a board so full that no piece pose is legal.
        let mut board = Board::new(2, 2);
        for y in 0..2 {
            for x in 0..2 {
                board.set(x, y, CellKind::Some(PieceType::O));
            }
        }
        // O on a 2x2 board has nowhere to go; movegen yields no resting poses.
        let active = movegen::spawn_piece(PieceType::O, 2, 2);
        let state = SearchState::for_test(board, active, None, std::iter::empty());
        if hold_placements(&state).is_empty() {
            let mut beam = BeamPlanner::new(8);
            assert!(drive(&mut beam, &state, &linear(), SearchBudget::beam(2)).is_none());
            // The dead root is still fingerprinted: re-rooting at it stays a no-op
            // (no per-call re-seed of a state that can never produce a plan).
            beam.reroot(&state, &linear(), 2);
            assert!(beam.best().is_none());
        }
    }

    #[test]
    fn beam_honors_quantum_and_preserves_final_decision() {
        let state = engine_snapshot_state(11);
        let eval = linear();
        let mut beam = BeamPlanner::new(16);

        beam.reroot(&state, &eval, 3);
        let seeded = beam.best().expect("best is valid right after seeding");

        assert_eq!(
            beam.think(1, &eval),
            ThinkProgress::Working,
            "one parent is not enough to finish the next generation"
        );
        assert_eq!(beam.nodes_expanded(), 1, "beam should honor think(1)");
        let after_one = beam.best().expect("partial generation keeps a valid best");
        assert_eq!(
            (after_one.placement.origin(), after_one.score),
            (seeded.placement.origin(), seeded.score),
            "partial generation scores stay staged until the generation completes"
        );

        let mut wider = BeamPlanner::new(16);
        wider.reroot(&state, &eval, 3);
        assert_eq!(wider.think(16, &eval), ThinkProgress::Working);
        assert!(
            wider.nodes_expanded() > beam.nodes_expanded(),
            "larger quantum should spend more work in one call"
        );

        for _ in 0..10_000 {
            if beam.think(1, &eval) == ThinkProgress::Exhausted {
                break;
            }
        }
        let final_plan = beam.best().expect("final best");
        let one_shot = drive(
            &mut BeamPlanner::new(16),
            &state,
            &eval,
            SearchBudget::beam(3),
        )
        .unwrap();

        assert_eq!(final_plan.placement.origin(), one_shot.placement.origin());
        assert_eq!(
            final_plan.placement.rotation(),
            one_shot.placement.rotation()
        );
        assert_eq!(final_plan.placement.path, one_shot.placement.path);
        assert_eq!(final_plan.score, one_shot.score);

        // Deeper search may refine the choice but never invalidates it: the seeded
        // and final plans are both legal root placements.
        let legal = hold_placements(&state);
        for plan in [&seeded, &final_plan] {
            assert!(
                legal.iter().any(|p| p.origin() == plan.placement.origin()
                    && p.rotation() == plan.placement.rotation()
                    && p.path == plan.placement.path),
                "anytime best is a real root placement"
            );
        }
    }

    #[test]
    fn beam_reasons_through_hold() {
        // Active S (awkward), held I, and a 1-wide well 4 deep only an I clears: a
        // multi-ply beam must choose to hold the I and clear (the same situation
        // greedy's `greedy_uses_hold_when_the_held_piece_is_better` pins, now driven
        // through the beam's hold-aware transition at depth >= 1).
        let mut board = Board::new(4, 12);
        for y in 0..4 {
            for x in 0..3 {
                board.set(x, y, CellKind::Some(PieceType::O));
            }
        }
        let active = movegen::spawn_piece(PieceType::S, 4, 12);
        let state = SearchState::for_test(board, active, Some(PieceType::I), std::iter::empty());

        let mut beam = BeamPlanner::new(16);
        let plan = drive(&mut beam, &state, &linear(), SearchBudget::beam(1)).unwrap();
        assert!(
            plan.uses_hold(),
            "beam should hold to bring in the well-clearing I"
        );
        assert_eq!(plan.placement.piece_type(), PieceType::I);
        assert_eq!(plan.placement.path.first(), Some(&movegen::Move::Hold));
    }

    #[test]
    fn beam_only_returns_legal_movegen_placements() {
        // The chosen ply-1 placement is always one movegen actually produced for the
        // root state (never an illegal/synthesized move). Verify the plan's pose is in
        // the canonical movegen set and its path replays faithfully on the board.
        let state = engine_snapshot_state(3);
        let mut beam = BeamPlanner::new(16);
        let plan = drive(&mut beam, &state, &linear(), SearchBudget::beam(3)).unwrap();

        let legal = hold_placements(&state);
        let matches = legal.iter().any(|p| {
            p.origin() == plan.placement.origin()
                && p.rotation() == plan.placement.rotation()
                && p.used_hold == plan.placement.used_hold
                && p.path == plan.placement.path
        });
        assert!(matches, "beam returned a placement movegen did not produce");

        // The placement, locked into a clone, must not overlap existing cells
        // (it is a real resting pose the engine accepts).
        let mut after = state.board;
        let lock = after.lock_piece(&plan.placement.piece);
        // A legal lock places exactly the piece's four cells (minus any cleared).
        assert!(!lock.cells_locked.is_empty(), "a legal piece was locked");
    }

    #[test]
    fn beam_speculation_is_deterministic() {
        // An empty queue forces the speculation path; the same state run twice must
        // yield the identical plan (no RNG, canonical bag order, BEAM.md §5).
        let mut board = Board::new(6, 12);
        board.set(0, 0, CellKind::Some(PieceType::O));
        board.set(5, 0, CellKind::Some(PieceType::O));
        let active = movegen::spawn_piece(PieceType::T, 6, 12);
        // Empty queue + occupied hold: depth-2 search past the queue speculates.
        let state = SearchState::for_test(board, active, Some(PieceType::L), std::iter::empty());

        let mut a = BeamPlanner::new(12);
        let mut b = BeamPlanner::new(12);
        let pa = drive(&mut a, &state, &linear(), SearchBudget::beam(2)).unwrap();
        let pb = drive(&mut b, &state, &linear(), SearchBudget::beam(2)).unwrap();
        assert_eq!(pa.placement.origin(), pb.placement.origin());
        assert_eq!(pa.placement.rotation(), pb.placement.rotation());
        assert_eq!(pa.placement.path, pb.placement.path);
        assert_eq!(pa.score, pb.score);
    }

    #[test]
    fn beam_speculation_off_matches_concrete_depth1() {
        // With speculation OFF and an empty queue, depth-2 cannot expand past the
        // queue, so the decision collapses to the depth-1 (greedy) choice. This pins
        // that the toggle actually gates speculation.
        let mut board = Board::new(6, 12);
        board.set(0, 0, CellKind::Some(PieceType::O));
        let active = movegen::spawn_piece(PieceType::T, 6, 12);
        let state = SearchState::for_test(board, active, None, std::iter::empty());

        let mut beam = BeamPlanner::new(16).with_speculation(false);
        let mut bf = BestFirstPlanner::new();
        let bp = drive(&mut beam, &state, &linear(), SearchBudget::beam(2)).unwrap();
        let gp = drive(&mut bf, &state, &linear(), SearchBudget::single_ply()).unwrap();
        assert_eq!(bp.placement.origin(), gp.placement.origin());
        assert_eq!(bp.placement.rotation(), gp.placement.rotation());
    }

    /// Delegates to [`LinearEvaluator`] but reports a configurable `board_only`
    /// and counts every `evaluate_cols` call. With `board_only: false` it forces
    /// the naive per-child speculation fan — the oracle the dedup must match.
    struct CountingEval {
        inner: LinearEvaluator,
        board_only: bool,
        calls: std::sync::atomic::AtomicUsize,
    }

    impl CountingEval {
        fn new(board_only: bool) -> Self {
            Self {
                inner: linear(),
                board_only,
                calls: std::sync::atomic::AtomicUsize::new(0),
            }
        }
    }

    impl Evaluator for CountingEval {
        fn evaluate(
            &self,
            lock: &LockOutcome,
            board: &Board,
            t_spin: Option<TSpinKind>,
            ctx: EvalContext,
        ) -> (Value, Reward) {
            self.inner.evaluate(lock, board, t_spin, ctx)
        }

        fn evaluate_cols(
            &self,
            lock: &LockOutcome,
            board: crate::engine::ColumnView,
            t_spin: Option<TSpinKind>,
            ctx: EvalContext,
        ) -> (Value, Reward) {
            self.calls
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            self.inner.evaluate_cols(lock, board, t_spin, ctx)
        }

        fn board_only(&self) -> bool {
            self.board_only
        }
    }

    #[test]
    fn speculative_share_matches_naive_fan() {
        // The dedup shares one board-only evaluation across a placement's whole
        // bag fan; refusing `board_only` forces the naive per-child fan through
        // the same scoring funnel. Same decisions, same scores, same root
        // back-ups — with strictly fewer evaluations — or the dedup is wrong.
        let mut crafted_board = Board::new(6, 12);
        crafted_board.set(0, 0, CellKind::Some(PieceType::O));
        crafted_board.set(5, 0, CellKind::Some(PieceType::O));
        let crafted = SearchState::for_test(
            crafted_board,
            movegen::spawn_piece(PieceType::T, 6, 12),
            Some(PieceType::L),
            std::iter::empty(),
        );
        // (state, depth): crafted speculates at ply 2; the engine snapshots carry
        // a full visible queue, so a deep budget drives expansion past it.
        let cases = [
            (crafted, 3),
            (engine_snapshot_state(7), 9),
            (engine_snapshot_state(42), 9),
        ];
        for (state, depth) in cases {
            let shared = CountingEval::new(true);
            let naive = CountingEval::new(false);
            let mut a = BeamPlanner::new(12);
            let mut b = BeamPlanner::new(12);
            let pa = drive(&mut a, &state, &shared, SearchBudget::beam(depth)).unwrap();
            let pb = drive(&mut b, &state, &naive, SearchBudget::beam(depth)).unwrap();
            assert_eq!(pa.placement.origin(), pb.placement.origin());
            assert_eq!(pa.placement.rotation(), pb.placement.rotation());
            assert_eq!(pa.placement.path, pb.placement.path);
            assert_eq!(pa.score, pb.score);
            let backups = |p: &BeamPlanner| {
                p.root_scores()
                    .map(|(r, s)| (r.origin(), r.rotation(), s))
                    .collect::<Vec<_>>()
            };
            assert_eq!(backups(&a), backups(&b));
            let (na, nb) = (
                shared.calls.load(std::sync::atomic::Ordering::Relaxed),
                naive.calls.load(std::sync::atomic::Ordering::Relaxed),
            );
            assert!(
                na < nb,
                "sharing must evaluate strictly less once speculation fires ({na} vs {nb})"
            );
        }
    }

    /// A `SearchState` from a fresh engine that has spawned its first piece (a real,
    /// non-crafted position carrying hold + a full visible queue).
    fn engine_snapshot_state(seed: u64) -> SearchState {
        let mut engine = Engine::new(EngineConfig::default(), seed);
        engine.step(InputFrame::default());
        let snapshot = engine.snapshot();
        SearchState::from_snapshot(&snapshot).expect("active piece present")
    }
}
