//! Perfect-clear coverage search over legal seven-bag continuations.
//!
//! A receding-horizon hybrid planner for PC hunting. For each root decision it
//! builds a deterministic sample of legal continuations from the exact bag
//! remainder ([`SearchState::bag`]), searches each continuation independently,
//! and counts how many of them keep at least one reachable perfect clear. Root
//! moves are selected by that **option coverage** — how many futures stay
//! PC-alive — not by one optimistic line, so the choice is robust to which
//! piece actually arrives. When no root clears the configured coverage
//! threshold (or the board is not a PC construction site at all), a
//! transposition-pruned general beam supplies the ordinary-play decision.
//!
//! Line clears stay legal during the scan. The PC-specific pruning uses only
//! *necessary* conditions — cell-count divisibility and a height bound from the
//! rows that can still be cleared ([`pc_feasible`]) — so it can never prune a
//! continuation that actually reaches a PC within the horizon.
//!
//! # Session grain (the [`Mind`] contract)
//!
//! [`Mind::reroot`] seeds the ply-1 roots (scored with the evaluator, like the
//! beam's seeding) and enumerates the scenario sample; the per-scenario
//! searches are spent through [`Mind::think`]. The planner is **batch-grain**
//! like the beam: each `think` completes at least one whole scenario regardless
//! of the quantum (a scenario's verdict is indivisible), and equal total work
//! reaches the identical decision regardless of slicing. [`Mind::best`] is
//! anytime: the fallback beam's plan until the scan completes, then the
//! coverage pick if one cleared the threshold.
//!
//! # Provenance (the 2026-06-12 PC campaign)
//!
//! Ported from the codex-agent worktree's screening campaign and simplified to
//! the two coverage units that carried signal ([`PcCoverageUnit`]). Probed and
//! REJECTED there (4-seed TRAIN opener screen, PPC): a PC-shaped partial-board
//! ranking (0.025), a sampled-mass tiebreak on reveal coverage (0.0625), and a
//! per-reveal robustness threshold (0.0625) — none beat plain reveal coverage
//! (0.075–0.0875), so none survived the port. The control result motivating
//! the planner: a general TP beam with a dominating PC reward stays at
//! 0.0100–0.0125 PPC — reward shaping alone does not find perfect clears.

use rustc_hash::FxHashSet;

use crate::ai::eval::{EvalContext, Evaluator, Reward};
use crate::ai::movegen::Placement;
use crate::ai::search::{
    BeamPlanner, Mind, PlacementPlan, RootKey, ThinkProgress, hold_placements, score_child,
};
use crate::ai::state::{BagState, SearchState};
use crate::engine::PieceType;

/// What one unit of coverage is — the denominator a root's PC-alive count is
/// judged against.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PcCoverageUnit {
    /// Each sampled continuation counts once: coverage = solved scenarios out
    /// of the sample.
    Scenarios,
    /// Continuations aggregate by their **first unknown draw**: a root covers
    /// a reveal when at least one sampled continuation starting with that
    /// piece keeps a PC, and the denominator is the number of distinct legal
    /// next reveals ([`BagState::possible_pieces`]). This judges a move by how
    /// many of the opponent-of-fate's actual next pieces it survives — the
    /// screen winner of the 2026-06-12 campaign.
    Reveals,
}

/// The planner's full configuration — plain data, so a research bot spec can
/// carry it verbatim and a registered arm reproduces from its literal.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PcCoverageConfig {
    /// Most continuations sampled per decision (deterministic FNV-order
    /// subsample of the full enumeration when it is larger).
    pub scenario_cap: usize,
    /// Beam width *per ply-1 root* inside each scenario's PC search.
    pub width_per_root: usize,
    /// Minimum coverage, in percent of the unit denominator, for a root to be
    /// chosen over the fallback beam (`required = ceil(denominator × pct /
    /// 100)`, and never zero covered).
    pub min_coverage_percent: u8,
    /// Width of the transposition-pruned fallback beam that owns the decision
    /// whenever the coverage search abstains.
    pub fallback_width: usize,
    /// How coverage is counted (see [`PcCoverageUnit`]).
    pub unit: PcCoverageUnit,
}

impl PcCoverageConfig {
    /// Clamp degenerate values: zero caps/widths become 1, the percent caps at
    /// 100, and a [`Reveals`](PcCoverageUnit::Reveals) sample is wide enough to
    /// stratify over every possible first draw (a smaller cap could never reach
    /// its own threshold).
    fn normalized(mut self) -> Self {
        self.scenario_cap = self.scenario_cap.max(1);
        if self.unit == PcCoverageUnit::Reveals {
            self.scenario_cap = self.scenario_cap.max(PieceType::LEN);
        }
        self.width_per_root = self.width_per_root.max(1);
        self.min_coverage_percent = self.min_coverage_percent.min(100);
        self.fallback_width = self.fallback_width.max(1);
        self
    }
}

/// One node in a scenario's per-root beam frontier.
#[derive(Clone)]
struct Node {
    state: SearchState,
    acc_reward: Reward,
    root_index: usize,
    score: i32,
}

/// The in-flight coverage scan: the scenario sample plus per-root accumulators,
/// advanced one scenario at a time by [`Mind::think`].
struct Scan {
    state: SearchState,
    roots: Vec<Placement>,
    /// Each root's one-ply evaluator score (`value + reward`), the coverage
    /// tiebreak — among equally covered roots, ordinary play decides.
    root_fallback: Vec<i32>,
    scenarios: Vec<Vec<PieceType>>,
    /// Next scenario to search; `== scenarios.len()` means the scan is done.
    cursor: usize,
    /// The unit actually in effect: [`Reveals`](PcCoverageUnit::Reveals)
    /// degenerates to [`Scenarios`](PcCoverageUnit::Scenarios) when the
    /// revealed queue already covers the horizon (no unknown draw to
    /// stratify over).
    unit: PcCoverageUnit,
    /// Coverage denominator (fixed at seed time): the sample size for
    /// scenarios, the distinct legal next reveals for reveals.
    denominator: usize,
    /// Per-root solved-scenario counts ([`PcCoverageUnit::Scenarios`]).
    scenario_hits: Vec<u32>,
    /// Per-root, per-first-draw "kept a PC" flags ([`PcCoverageUnit::Reveals`]).
    reveal_hits: Vec<[bool; PieceType::LEN]>,
}

/// One root decision's session state. The scan is boxed: one lives per
/// planner, and the indirection keeps the enum (and `Run`) small.
enum ScanState {
    /// Scenarios remain; [`Mind::think`] advances the scan.
    Scanning(Box<Scan>),
    /// Scan complete: the coverage pick, or `None` (fallback beam decides).
    Done(Option<PlacementPlan>),
}

struct Run {
    root_key: RootKey,
    max_depth: u8,
    scan: ScanState,
    /// Node expansions spent on the coverage scan (root seeding + scenario
    /// searches); the fallback beam meters its own.
    nodes: u32,
}

/// Scenario-coverage PC planner with a TP-beam fallback (module docs).
pub struct PcCoveragePlanner {
    config: PcCoverageConfig,
    fallback: BeamPlanner,
    run: Option<Run>,
}

impl PcCoveragePlanner {
    pub fn new(config: PcCoverageConfig) -> Self {
        let config = config.normalized();
        Self {
            config,
            fallback: BeamPlanner::transposing(config.fallback_width),
            run: None,
        }
    }

    /// Seed a scan for `state`, or rule the root out immediately (not a PC
    /// construction site, no legal placement, or no lookahead). Returns the
    /// scan state plus the node expansions the seeding spent.
    fn seed(&self, state: &SearchState, eval: &dyn Evaluator, max_depth: u8) -> (ScanState, u32) {
        if max_depth == 0 || !pc_candidate_state(state) {
            return (ScanState::Done(None), 0);
        }
        let roots = hold_placements(state);
        if roots.is_empty() {
            return (ScanState::Done(None), 0);
        }

        let root_ctx = EvalContext {
            combo: state.combo,
            b2b: state.b2b,
        };
        let root_fallback: Vec<i32> = roots
            .iter()
            .map(|placement| {
                let (_, value, reward) = score_child(state, placement, eval, root_ctx);
                (value + reward).0
            })
            .collect();
        let nodes = roots.len() as u32;

        // `max_depth` placements consume up to `max_depth` queue pieces: one
        // advance per ply past the first, plus one funding pull if an
        // empty-hold swap fires (it can fire at most once per line — hold
        // stays occupied after). Continuations extend the queue to that bound.
        let unknown_draws = usize::from(max_depth).saturating_sub(state.queue.len());
        let unit = if unknown_draws == 0 {
            // Nothing unknown to stratify over: a single (empty) continuation,
            // judged as one scenario.
            PcCoverageUnit::Scenarios
        } else {
            self.config.unit
        };
        let scenarios = match unit {
            PcCoverageUnit::Scenarios => {
                continuation_sample(state.bag, unknown_draws, self.config.scenario_cap)
            }
            PcCoverageUnit::Reveals => {
                continuation_sample_stratified(state.bag, unknown_draws, self.config.scenario_cap)
            }
        };
        let denominator = match unit {
            PcCoverageUnit::Scenarios => scenarios.len(),
            PcCoverageUnit::Reveals => state.bag.possible_pieces().len(),
        };

        let scan = Box::new(Scan {
            state: state.clone(),
            root_fallback,
            scenario_hits: vec![0; roots.len()],
            reveal_hits: vec![[false; PieceType::LEN]; roots.len()],
            roots,
            scenarios,
            cursor: 0,
            unit,
            denominator,
        });
        (ScanState::Scanning(scan), nodes)
    }
}

impl Scan {
    /// Search the scenario at `cursor` and fold its result into the per-root
    /// accumulators. Returns the node expansions spent.
    fn advance(&mut self, eval: &dyn Evaluator, max_depth: u8, width_per_root: usize) -> u32 {
        let continuation = &self.scenarios[self.cursor];
        let first_draw = continuation.first().copied();
        let (solved, spent) = search_scenario(
            &self.state,
            &self.roots,
            continuation,
            eval,
            max_depth,
            width_per_root,
        );
        for (root_index, solved) in solved.into_iter().enumerate() {
            if !solved {
                continue;
            }
            match self.unit {
                PcCoverageUnit::Scenarios => self.scenario_hits[root_index] += 1,
                PcCoverageUnit::Reveals => {
                    // Reveals implies unknown draws (seed-time normalization),
                    // so every continuation has a first piece.
                    let piece = first_draw.expect("Reveals unit implies a first draw");
                    self.reveal_hits[root_index][piece as usize] = true;
                }
            }
        }
        self.cursor += 1;
        spent
    }

    fn done(&self) -> bool {
        self.cursor >= self.scenarios.len()
    }

    /// A root's covered-unit count.
    fn covered(&self, root_index: usize) -> usize {
        match self.unit {
            PcCoverageUnit::Scenarios => self.scenario_hits[root_index] as usize,
            PcCoverageUnit::Reveals => self.reveal_hits[root_index]
                .iter()
                .filter(|&&hit| hit)
                .count(),
        }
    }

    /// The completed scan's pick: the most-covered root at or above the
    /// threshold, evaluator score then canonical order breaking ties (first
    /// maximum wins, the determinism rule every planner here follows).
    fn verdict(&self, min_coverage_percent: u8) -> Option<PlacementPlan> {
        let required = (self.denominator * usize::from(min_coverage_percent)).div_ceil(100);
        let mut best: Option<(usize, i32, usize)> = None;
        for root_index in 0..self.roots.len() {
            let covered = self.covered(root_index);
            // `covered > 0` also guards the degenerate `required == 0` spec:
            // a root that kept no PC anywhere is never a coverage pick.
            if covered == 0 || covered < required {
                continue;
            }
            let candidate = (covered, self.root_fallback[root_index], root_index);
            if best.is_none_or(|current| {
                candidate.0 > current.0 || (candidate.0 == current.0 && candidate.1 > current.1)
            }) {
                best = Some(candidate);
            }
        }
        best.map(|(covered, fallback, root_index)| PlacementPlan {
            placement: self.roots[root_index].clone(),
            // Packed for log readability only (coverage dominates); the
            // selection above is the decision.
            score: (covered as i32)
                .saturating_mul(1_000_000)
                .saturating_add(fallback),
        })
    }
}

impl Mind for PcCoveragePlanner {
    fn reroot(&mut self, state: &SearchState, eval: &dyn Evaluator, max_depth: u8) {
        let root_key = RootKey::of(state);
        if self
            .run
            .as_ref()
            .is_some_and(|run| run.root_key == root_key && run.max_depth == max_depth)
        {
            return; // already rooted here: the in-flight scan continues
        }
        self.fallback.reroot(state, eval, max_depth);
        let (scan, nodes) = self.seed(state, eval, max_depth);
        self.run = Some(Run {
            root_key,
            max_depth,
            scan,
            nodes,
        });
    }

    /// Batch-grain: each call completes at least one whole scenario (then more
    /// until `quantum` expansions are spent); once the scan abstains, the call
    /// stream drives the fallback beam instead.
    fn think(&mut self, quantum: u32, eval: &dyn Evaluator) -> ThinkProgress {
        let width_per_root = self.config.width_per_root;
        let min_coverage_percent = self.config.min_coverage_percent;
        let Some(run) = self.run.as_mut() else {
            return ThinkProgress::Exhausted; // never rooted: nothing to think about
        };
        if let ScanState::Scanning(scan) = &mut run.scan {
            let mut spent = 0u32;
            while spent < quantum && !scan.done() {
                spent = spent.saturating_add(scan.advance(eval, run.max_depth, width_per_root));
            }
            run.nodes = run.nodes.saturating_add(spent);
            if scan.done() {
                run.scan = ScanState::Done(scan.verdict(min_coverage_percent));
            }
        }
        match &run.scan {
            // A coverage pick is final: scenario verdicts cannot change, and
            // the fallback's deeper generations no longer own the decision.
            ScanState::Done(Some(_)) => ThinkProgress::Exhausted,
            ScanState::Done(None) => self.fallback.think(quantum, eval),
            ScanState::Scanning(_) => ThinkProgress::Working,
        }
    }

    fn best(&self) -> Option<PlacementPlan> {
        match self.run.as_ref().map(|run| &run.scan) {
            Some(ScanState::Done(Some(plan))) => Some(plan.clone()),
            // Mid-scan or abstained: the fallback beam's anytime answer (it
            // was seeded at reroot, so this is valid immediately).
            _ => self.fallback.best(),
        }
    }

    fn nodes_expanded(&self) -> u32 {
        self.run.as_ref().map_or(0, |run| run.nodes) + self.fallback.nodes_expanded()
    }
}

/// Search one continuation: a per-root beam (width `width_per_root` each, so a
/// late root is never starved by an early root's fan-out) over `state` with the
/// continuation appended to the queue. Returns which roots reached a perfect
/// clear within `max_depth` plies, plus the node expansions spent.
fn search_scenario(
    state: &SearchState,
    roots: &[Placement],
    continuation: &[PieceType],
    eval: &dyn Evaluator,
    max_depth: u8,
    width_per_root: usize,
) -> (Vec<bool>, u32) {
    let mut base = state.clone();
    // The bag is deliberately NOT advanced: the extended queue covers the whole
    // horizon (seed-time bound), so no search path ever deals from it, and the
    // per-scenario transposition key below need not include it.
    base.queue.extend(continuation.iter().copied());
    let mut solved = vec![false; roots.len()];
    let mut frontier = Vec::new();
    let mut spent = 0u32;
    let ctx = EvalContext {
        combo: base.combo,
        b2b: base.b2b,
    };

    for (root_index, placement) in roots.iter().enumerate() {
        let (child, value, reward) = score_child(&base, placement, eval, ctx);
        spent = spent.saturating_add(1);
        if child.board.is_empty() && !child.dead {
            solved[root_index] = true;
        } else if !child.dead && pc_feasible(&child, max_depth.saturating_sub(1)) {
            frontier.push(Node {
                score: (value + reward).0,
                state: child,
                acc_reward: reward,
                root_index,
            });
        }
    }
    truncate_per_root(&mut frontier, roots.len(), width_per_root);

    for depth in 2..=max_depth {
        if frontier.is_empty() || solved.iter().all(|&done| done) {
            break;
        }
        let remaining = max_depth - depth;
        let mut next = Vec::new();
        for parent in frontier {
            if solved[parent.root_index] || parent.state.dead {
                continue;
            }
            let ctx = EvalContext {
                combo: parent.state.combo,
                b2b: parent.state.b2b,
            };
            for placement in hold_placements(&parent.state) {
                let (child, value, reward) = score_child(&parent.state, &placement, eval, ctx);
                spent = spent.saturating_add(1);
                if child.board.is_empty() && !child.dead {
                    solved[parent.root_index] = true;
                    continue;
                }
                if child.dead || !pc_feasible(&child, remaining) {
                    continue;
                }
                let acc = parent.acc_reward + reward;
                next.push(Node {
                    score: (value + acc).0,
                    state: child,
                    acc_reward: acc,
                    root_index: parent.root_index,
                });
            }
        }
        truncate_per_root(&mut next, roots.len(), width_per_root);
        frontier = next;
    }

    (solved, spent)
}

/// Necessary conditions for a PC within `remaining` more pieces: some piece
/// count makes the total cells a whole number of rows, and the current stack
/// is no taller than the rows that would all clear. Never prunes a reachable
/// PC (both conditions hold on every true PC line).
fn pc_feasible(state: &SearchState, remaining: u8) -> bool {
    if state.board.is_empty() {
        return true;
    }
    let cells: usize = state
        .board
        .columns()
        .iter()
        .map(|column| column.count_ones() as usize)
        .sum();
    let height = state
        .board
        .columns()
        .iter()
        .map(|column| (u64::BITS - column.leading_zeros()) as usize)
        .max()
        .unwrap_or(0);
    let width = state.board.width();

    (1..=usize::from(remaining)).any(|pieces| {
        let cells_available = cells + 4 * pieces;
        cells_available.is_multiple_of(width) && height <= cells_available / width
    })
}

/// PC hunting is a candidate subgoal inside a general policy. Once a failed
/// attempt has produced a large or tall stack, the fallback beam owns recovery;
/// the coverage scan resumes when ordinary play returns to a compact
/// construction zone.
fn pc_candidate_state(state: &SearchState) -> bool {
    let cells: u32 = state.board.columns().iter().map(|c| c.count_ones()).sum();
    let height = state
        .board
        .columns()
        .iter()
        .map(|column| u64::BITS - column.leading_zeros())
        .max()
        .unwrap_or(0);
    cells <= 40 && height <= 6
}

/// Keep the best `width` nodes **per root** (stable order: ties keep canonical
/// enumeration order), deduplicating transposed states within a root. The bag
/// is constant across a scenario (see [`search_scenario`]), so [`RootKey`]
/// alone identifies a future here.
fn truncate_per_root(nodes: &mut Vec<Node>, root_count: usize, width: usize) {
    nodes.sort_by_key(|node| std::cmp::Reverse(node.score));
    let mut seen = FxHashSet::default();
    let mut kept = vec![0usize; root_count];
    nodes.retain(|node| {
        kept[node.root_index] < width
            && seen.insert((node.root_index, RootKey::of(&node.state)))
            && {
                kept[node.root_index] += 1;
                true
            }
    });
}

/// Every legal `draws`-long continuation of `bag`, truncated to `cap` by FNV
/// hash order — a deterministic, seed-free pseudo-random subsample.
fn continuation_sample(bag: BagState, draws: usize, cap: usize) -> Vec<Vec<PieceType>> {
    let mut all = enumerate_continuations(bag, draws);
    if all.len() <= cap {
        return all;
    }
    all.sort_by_key(|sequence| continuation_hash(sequence));
    all.truncate(cap);
    all
}

/// Like [`continuation_sample`], but guaranteeing every possible first draw at
/// least one continuation before the rest of the cap fills in hash order — the
/// sample the [`Reveals`](PcCoverageUnit::Reveals) denominator requires.
fn continuation_sample_stratified(bag: BagState, draws: usize, cap: usize) -> Vec<Vec<PieceType>> {
    let mut all = enumerate_continuations(bag, draws);
    if all.len() <= cap || draws == 0 {
        return all;
    }
    all.sort_by_key(|sequence| continuation_hash(sequence));
    let mut selected = Vec::with_capacity(cap);
    for piece in bag.possible_pieces() {
        if let Some(index) = all.iter().position(|sequence| sequence[0] == piece) {
            selected.push(all.remove(index));
            if selected.len() == cap {
                return selected;
            }
        }
    }
    selected.extend(all.into_iter().take(cap - selected.len()));
    selected
}

/// Every legal `draws`-long piece sequence from `bag` (bag-boundary refills
/// included), in canonical depth-first order. The full enumeration is
/// materialized before sampling truncates it; with the engine's 5-piece
/// preview and the registered depth-10 arms this is ≤ 2,520 sequences — check
/// this bound before raising depth or shrinking the preview.
fn enumerate_continuations(bag: BagState, draws: usize) -> Vec<Vec<PieceType>> {
    fn recurse(
        bag: BagState,
        draws: usize,
        prefix: &mut Vec<PieceType>,
        out: &mut Vec<Vec<PieceType>>,
    ) {
        if draws == 0 {
            out.push(prefix.clone());
            return;
        }
        for piece in bag.possible_pieces() {
            let mut next_bag = bag;
            next_bag.deal(piece);
            prefix.push(piece);
            recurse(next_bag, draws - 1, prefix, out);
            prefix.pop();
        }
    }
    let mut out = Vec::new();
    recurse(bag, draws, &mut Vec::with_capacity(draws), &mut out);
    out
}

/// FNV-1a over the piece sequence — the deterministic shuffle key of the
/// continuation samples (no RNG anywhere in the planner).
fn continuation_hash(sequence: &[PieceType]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for &piece in sequence {
        hash ^= piece as u64 + 1;
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::eval::LinearEvaluator;
    use crate::ai::movegen::spawn_piece;
    use crate::ai::search::{SearchBudget, think_to_completion};
    use crate::engine::{Board, CellKind};

    fn config(unit: PcCoverageUnit) -> PcCoverageConfig {
        PcCoverageConfig {
            scenario_cap: 8,
            width_per_root: 8,
            min_coverage_percent: 100,
            fallback_width: 8,
            unit,
        }
    }

    #[test]
    fn continuation_sample_obeys_the_bag() {
        let mut bag = BagState::full();
        bag.deal(PieceType::I);
        let sample = continuation_sample(bag, 2, 64);
        assert_eq!(sample.len(), 30);
        assert!(sample.iter().all(|s| s[0] != PieceType::I));
        assert!(sample.iter().all(|s| s[0] != s[1]));
    }

    #[test]
    fn stratified_sample_covers_every_first_draw() {
        let mut bag = BagState::full();
        bag.deal(PieceType::I);
        // Cap far below the 6×5×4 enumeration: every legal first draw must
        // still appear (the Reveals denominator counts all six).
        let sample = continuation_sample_stratified(bag, 3, 7);
        assert_eq!(sample.len(), 7);
        for piece in bag.possible_pieces() {
            assert!(
                sample.iter().any(|s| s[0] == piece),
                "first draw {piece:?} missing from the stratified sample"
            );
        }
    }

    #[test]
    fn finds_a_reachable_one_move_pc() {
        // 4-wide board, three columns filled four high: a vertical I in the
        // last column perfect-clears in one move.
        let mut board = Board::new(4, 8);
        for y in 0..4 {
            for x in 0..3 {
                board.set(x, y, CellKind::Some(PieceType::O));
            }
        }
        let active = spawn_piece(PieceType::I, 4, 8);
        let state = SearchState::for_test(board, active, None, [PieceType::T]);
        let mut planner = PcCoveragePlanner::new(config(PcCoverageUnit::Scenarios));
        let plan = think_to_completion(
            &mut planner,
            &state,
            &LinearEvaluator::default(),
            SearchBudget::beam(1),
        )
        .expect("one-move PC should be found");
        let mut child = state.clone();
        child.commit_placement(&plan.placement);
        assert!(child.board.is_empty());
    }

    #[test]
    fn abstains_to_the_fallback_beam_outside_the_construction_zone() {
        // A 7-high stack fails the candidate gate; the decision must be the
        // fallback TP beam's, byte-for-byte.
        let mut board = Board::new(4, 12);
        for y in 0..7 {
            for x in 0..3 {
                board.set(x, y, CellKind::Some(PieceType::O));
            }
        }
        let active = spawn_piece(PieceType::T, 4, 12);
        let queue = [PieceType::I, PieceType::O, PieceType::L];
        let state = SearchState::for_test(board.clone(), active.clone(), None, queue);

        let eval = LinearEvaluator::default();
        let mut planner = PcCoveragePlanner::new(config(PcCoverageUnit::Reveals));
        let pc_plan = think_to_completion(&mut planner, &state, &eval, SearchBudget::beam(3))
            .expect("fallback plan");

        let mut beam = BeamPlanner::transposing(8);
        let beam_plan =
            think_to_completion(&mut beam, &state, &eval, SearchBudget::beam(3)).expect("beam");

        assert_eq!(pc_plan.placement.origin(), beam_plan.placement.origin());
        assert_eq!(pc_plan.placement.rotation(), beam_plan.placement.rotation());
        assert_eq!(pc_plan.placement.path, beam_plan.placement.path);
        assert_eq!(pc_plan.score, beam_plan.score);
    }

    #[test]
    fn think_slicing_never_changes_the_decision() {
        // The anytime contract: quantum-1 slicing reaches the identical
        // decision as one u32::MAX drain, and `best()` is valid (the seeded
        // fallback's plan) at every suspension point in between.
        let mut board = Board::new(4, 10);
        for x in 0..2 {
            board.set(x, 0, CellKind::Some(PieceType::O));
            board.set(x, 1, CellKind::Some(PieceType::O));
        }
        let active = spawn_piece(PieceType::O, 4, 10);
        // Short queue + depth 4 ⇒ unknown draws ⇒ a real multi-scenario scan.
        let state = SearchState::for_test(board, active, None, [PieceType::I]);
        let eval = LinearEvaluator::default();

        let mut drained = PcCoveragePlanner::new(config(PcCoverageUnit::Reveals));
        let one_shot = think_to_completion(&mut drained, &state, &eval, SearchBudget::beam(4))
            .expect("a plan");

        let mut sliced = PcCoveragePlanner::new(config(PcCoverageUnit::Reveals));
        sliced.reroot(&state, &eval, 4);
        assert!(sliced.best().is_some(), "anytime best right after reroot");
        let mut guard = 0;
        while sliced.think(1, &eval) == ThinkProgress::Working {
            assert!(sliced.best().is_some(), "anytime best mid-scan");
            guard += 1;
            assert!(guard < 100_000, "think never exhausted");
        }
        let step_plan = sliced.best().expect("a plan");

        assert_eq!(one_shot.placement.origin(), step_plan.placement.origin());
        assert_eq!(
            one_shot.placement.rotation(),
            step_plan.placement.rotation()
        );
        assert_eq!(one_shot.placement.path, step_plan.placement.path);
        assert_eq!(one_shot.score, step_plan.score);
    }
}
