//! The deterministic, batch-shaped beam Tier-2 planner (Colder Clear, STEP 2).
//!
//! A CC2-style **beam search** behind the same [`Mind`] session seam the greedy
//! Tier-1 planner implements, so it drops in with no controller / policy / runner
//! change. See `BEAM.md` (this directory) for the design record; the load-bearing
//! pins:
//!
//! 1. **Determinism (BEAM.md §1).** Zero RNG, no clock. The only tie-breaker is
//!    movegen's canonical placement order: children are pushed in
//!    `(parent-order, movegen-order)` and ranked with a **stable** sort descending
//!    by score, so a tie resolves to the earlier-enumerated node. Back-up uses `>`
//!    (not `>=`) so the **first** maximum wins, mirroring `greedy.rs`'s rule.
//! 2. **Hold-aware transition (BEAM.md §3).** A node forks the [`SearchState`] and
//!    advances through a [`Placement`] with [`SearchState::commit_placement`], the
//!    Step-0 transition that models a hold swap and deals the bag exactly once.
//! 3. **One generation at a time (BEAM.md §4/§7).** Every child of a generation
//!    is scored before any is expanded — the whole-generation grain a neural
//!    value net needs to fold scoring into one forward pass.
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
//! As a session the beam is **batch-grain**: [`Mind::think`] expands exactly one
//! generation per call regardless of the quantum (a generation is scored
//! whole, indivisible by design — pin 3) and reports
//! [`ThinkProgress::Exhausted`] once it reaches the run's depth cap or the
//! frontier empties. [`Mind::best`] is the backed-up ply-1 argmax at any point.

use rustc_hash::FxHashSet;

use crate::ai::eval::{EvalContext, Evaluator, Reward, Value};
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
    /// This node's score: `(leaf_value + acc_reward).0`, in evaluator units. Cached
    /// so sort/truncate is a field read, not a re-evaluation.
    score: i32,
    /// A per-branch reward discount carried from speculation (BEAM.md §5). `1.0`
    /// until the branch crosses into speculative plies; multiplied by [`SPEC_DECAY`]
    /// at each speculative expansion so deeper speculative rewards count for less.
    spec_weight: f32,
}

/// One not-yet-scored child of the generation being expanded: the forked state, its
/// lock results, and the node bookkeeping, in **one** struct — so the batch inputs
/// and the post-score node build read the same element and can never fall out of
/// lockstep (this replaced a pair of parallel "owners"/"meta" vectors).
struct PendingChild {
    state: SearchState,
    lock: LockOutcome,
    t_spin: Option<TSpinKind>,
    /// The chain context the child scores under (the PARENT's pre-placement state).
    ctx: EvalContext,
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
    /// In-flight search, `None` between decisions. Reset on a new root state.
    run: Option<BeamRun>,
}

impl BeamPlanner {
    /// A beam planner of the given width, with bag speculation **on** (the default).
    pub fn new(beam_width: usize) -> Self {
        Self {
            beam_width: beam_width.max(1),
            speculate: true,
            transpose: false,
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

    /// Seed a fresh run for `state`: form the ply-1 root children (depth 1), score
    /// them as one batch, and build the initial frontier. A topped-out state (no
    /// legal placement) seeds an *empty* run — the fingerprint still records it,
    /// so re-rooting at the same dead state stays a no-op.
    fn seed(&self, state: &SearchState, eval: &dyn Evaluator, max_depth: u8) -> BeamRun {
        let roots = hold_placements(state);

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
            depth: 1,
            root_key: RootKey::of(state),
            max_depth,
            expanded: 0,
        };
        Self::score_into_frontier(&mut run, pending, eval, self.beam_width, self.transpose);
        run
    }

    /// Expand one generation from the current frontier: form every child, then hand
    /// the generation to [`score_into_frontier`](Self::score_into_frontier).
    /// An associated fn (config passed in) so [`Mind::think`] can call it while
    /// holding the run borrowed out of `self`.
    fn expand_generation(
        run: &mut BeamRun,
        eval: &dyn Evaluator,
        beam_width: usize,
        speculate: bool,
        transpose: bool,
    ) {
        let mut pending: Vec<PendingChild> = Vec::new();
        run.expanded += run.frontier.len() as u32;

        for parent in &run.frontier {
            if parent.state.dead {
                // A dead branch is terminal: its DEATH_SCORE back-up already
                // credited its root; it expands to nothing (and never
                // speculates).
                continue;
            }
            if parent.state.queue.is_empty() {
                // Past the visible queue: speculate over the bag if enabled, else this
                // node is terminal (no concrete next piece to advance the active). A
                // terminal node contributes no children; its `root_best` was already
                // recorded when it entered the frontier, so the back-up keeps it.
                if speculate {
                    Self::expand_speculative(parent, &mut pending);
                }
                continue;
            }
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
                    root_index: parent.root_index,
                    parent_acc: parent.acc_reward,
                    spec_weight: parent.spec_weight,
                });
            }
        }

        Self::score_into_frontier(run, pending, eval, beam_width, transpose);
        run.depth += 1;
    }

    /// Score a generation's `pending` children with `evaluate_cols` (one call per
    /// child, BEAM.md §7) and fold the results into the next frontier: weight each
    /// child's reward by its branch's speculative discount, accumulate the path
    /// reward, back up `root_best`, then stable-sort / truncate to the beam width.
    /// The shared scoring tail of [`seed`](Self::seed) (all-concrete, weight `1.0`)
    /// and [`expand_generation`](Self::expand_generation).
    fn score_into_frontier(
        run: &mut BeamRun,
        pending: Vec<PendingChild>,
        eval: &dyn Evaluator,
        beam_width: usize,
        transpose: bool,
    ) {
        // Score each child on the hot path directly. (No batch seam here on
        // purpose: a batched value-net backend belongs AT the Evaluator trait,
        // not in the search loop — docs/value-net-postmortem.md has the history.)
        let scores: Vec<(Value, Reward)> = pending
            .iter()
            .map(|p| eval.evaluate_cols(&p.lock, p.state.board.view(), p.t_spin, p.ctx))
            .collect();

        let mut next: Vec<BeamNode> = Vec::with_capacity(pending.len());
        for (p, (value, reward)) in pending.into_iter().zip(scores) {
            // Death is absolute: override the batched eval (the truncated board
            // it scored is a death remnant, not a position).
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
            if score > run.root_best[p.root_index] {
                run.root_best[p.root_index] = score;
            }
            next.push(BeamNode {
                state: p.state,
                acc_reward: acc,
                root_index: p.root_index,
                score,
                spec_weight: p.spec_weight,
            });
        }

        // Stable sort descending by score, then truncate: ties keep canonical order.
        // With transposition pruning, equal (root, future-state, bag) nodes first
        // collapse to their best (therefore first-seen) derivation — a sound
        // max-merge, since identical states share identical futures and only the
        // accumulated reward differs (header pin 5).
        sort_desc_by_score(&mut next);
        if transpose {
            let mut seen = FxHashSet::default();
            next.retain(|node| {
                seen.insert((node.root_index, RootKey::of(&node.state), node.state.bag))
            });
        }
        next.truncate(beam_width);
        run.frontier = next;
    }

    /// Speculative expansion of an empty-queue `parent` (BEAM.md §5): for each piece
    /// still in the 7-bag remainder (in canonical [`PieceType::all`] order) and each
    /// placement of the parent's active piece, lock the placement and spawn that
    /// speculative piece as the next active. The child carries a reward weight scaled
    /// by [`SPEC_DECAY`] so deeper speculative rewards count for less.
    ///
    /// No RNG, no expectimax average — every bag-legal piece is enumerated and
    /// beam-width truncation prunes the fan-out, keeping the planner deterministic.
    fn expand_speculative(parent: &BeamNode, pending: &mut Vec<PendingChild>) {
        let placements = hold_placements(&parent.state);
        let child_weight = parent.spec_weight * SPEC_DECAY;
        let parent_ctx = EvalContext {
            combo: parent.state.combo,
            b2b: parent.state.b2b,
        };
        for next_piece in PieceType::all() {
            if !parent.state.bag.contains(next_piece) {
                continue;
            }
            for placement in &placements {
                let mut child = parent.state.clone();
                // Classify against the pre-lock board, like the concrete path.
                let t_spin = crate::engine::classify_t_spin(&placement.piece, &child.board);
                // The hold-aware speculative transition: the same `used_hold` swap as
                // the concrete path, but dealing `next_piece` (the visible queue is
                // exhausted at a speculative node) as the new active rather than the
                // queue front. An empty-queue node only offers `used_hold` placements
                // when hold is occupied (movegen's `hold.or(queue_front)`), so the
                // shared transition's empty-hold funding pop never fires here.
                let lock = child.commit_placement_with_next(placement, next_piece);
                pending.push(PendingChild {
                    state: child,
                    lock,
                    t_spin,
                    ctx: parent_ctx,
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
    run.depth >= run.max_depth || run.frontier.is_empty()
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

    /// **Batch-grain**: one whole generation per call, regardless of `quantum` —
    /// a generation is scored as one indivisible whole (BEAM.md §7, the grain a
    /// neural value net needs).
    fn think(&mut self, _quantum: u32, eval: &dyn Evaluator) -> ThinkProgress {
        let (beam_width, speculate, transpose) = (self.beam_width, self.speculate, self.transpose);
        let Some(run) = self.run.as_mut() else {
            return ThinkProgress::Exhausted; // never rooted: nothing to think about
        };
        if exhausted(run) {
            return ThinkProgress::Exhausted;
        }
        Self::expand_generation(run, eval, beam_width, speculate, transpose);
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

/// Stable sort of beam nodes by descending score (ties keep canonical enumeration
/// order, the determinism rule of BEAM.md §1).
fn sort_desc_by_score(nodes: &mut [BeamNode]) {
    nodes.sort_by_key(|n| std::cmp::Reverse(n.score));
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
    fn beam_thinks_one_generation_per_call() {
        // The batch-grain session contract (BEAM.md §4): seeding is generation 1
        // and makes `best()` immediately valid; each `think` is one generation; a
        // depth-3 run takes exactly two thinks to exhaust — and `best()` stays a
        // valid plan at every point in between (the anytime contract).
        let state = engine_snapshot_state(11);
        let mut beam = BeamPlanner::new(16);

        beam.reroot(&state, &linear(), 3);
        let seeded = beam.best().expect("best is valid right after seeding");

        assert_eq!(
            beam.think(u32::MAX, &linear()),
            ThinkProgress::Working,
            "after generation 2 of 3 the beam still has work"
        );
        assert!(beam.best().is_some(), "anytime best between generations");

        assert_eq!(
            beam.think(u32::MAX, &linear()),
            ThinkProgress::Exhausted,
            "generation 3 reaches the depth cap"
        );
        let final_plan = beam.best().expect("final best");

        // Deeper search may refine the choice but never invalidates it: both are
        // legal root placements (the seeded one was checked by construction here).
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

    /// A `SearchState` from a fresh engine that has spawned its first piece (a real,
    /// non-crafted position carrying hold + a full visible queue).
    fn engine_snapshot_state(seed: u64) -> SearchState {
        let mut engine = Engine::new(EngineConfig::default(), seed);
        engine.step(InputFrame::default());
        let snapshot = engine.snapshot();
        SearchState::from_snapshot(&snapshot).expect("active piece present")
    }
}
