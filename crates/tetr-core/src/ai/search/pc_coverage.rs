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
//! rows that can still be cleared (`pc_feasible`) — so it can never prune a
//! continuation that actually reaches a PC within the horizon.
//!
//! # Session grain (the [`Mind`] contract)
//!
//! [`Mind::reroot`] seeds the ply-1 roots (scored with the evaluator, like the
//! beam's seeding) and enumerates the scenario sample; the per-scenario
//! searches are spent through [`Mind::think`]. The scan is **node-grain**: the
//! in-flight scenario's frontier and generation cursor live in the session, so
//! `think` is resumable mid-scenario and honors its quantum in `score_child`
//! units, overshooting by at most one parent expansion (the movegen fan-out,
//! ~34 evals). Scenarios are searched in the stratified sample's order — one
//! per possible reveal first — so a truncated scan has judged every reveal
//! once before it refines any. Equal total work reaches the identical decision
//! regardless of slicing.
//!
//! [`Mind::best`] is anytime: mid-scan it returns a **partial verdict** — the
//! coverage accumulated so far, judged against the *full* denominator
//! (conservative: hits only ever grow) — and falls back to the TP beam's plan
//! while nothing has cleared the threshold.
//!
//! # The scan budget ([`PcCoverageConfig::scan_node_budget`])
//!
//! An interactive venue cannot afford the full scan on every piece, so the
//! planner can cap the per-decision scan itself. When the cap trips, `think`
//! finalizes the scan early on the partial accumulators (and still commits a
//! line, below). The cap lives in the planner config — deliberately *not* in
//! the policy-level [`SearchBudget`](crate::ai::search::SearchBudget) — for two
//! reasons: a policy-level cut is invisible to the mind (it could never build
//! its line commitment), and an internal cut always lands on the same step
//! boundary regardless of the venue's quantum, so slicing invariance survives
//! budgeting exactly. `0` means unbounded — the registered research arms'
//! setting, byte-stable with the unbudgeted planner.
//!
//! # Line commitment ([`PcCoverageConfig::commit_lines`])
//!
//! A solved scenario IS a complete PC line; rescanning from scratch on every
//! piece of that line is the planner's dominant waste. When committing is on,
//! each scenario records the **first solving path per (root, first reveal)**;
//! when the verdict picks a root, the planner commits that root's branch
//! table together with the predicted post-placement state. On the next
//! reroot, if the observed state matches the prediction and the newly
//! revealed piece selects a recorded branch, the decision is served from the
//! committed line with **zero search**; each later follow-up confirms one
//! more assumed reveal and serves the next step. Any mismatch — garbage, a
//! surprise reveal, an imperfection substitution, a fallback interlude, the
//! line completing — drops the cache and rescans. Deterministic: the cache is
//! a pure function of past observations, so a fixed seed replays identically.
//! Off (`false`) for the registered research arms, whose recorded runs must
//! stay byte-stable.
//!
//! # Shared-prefix scan (Layer 4)
//!
//! Every scenario appends a different bag continuation to the same revealed
//! queue, so plies that consume only visible queue pieces are searched once
//! in a shared **prefix** phase; the scan forks into per-scenario tails only
//! at the first unknown draw. The prefix depth is
//! `min(queue.len(), horizon − 1)` when unknown draws remain — the ply count
//! before the first continuation piece is dealt. Line-commitment paths stitch
//! the prefix arena and the scenario arena at the branch. Semantics match the
//! per-scenario whole-line search when the prefix is skipped (`prefix_depth
//! == 0`: no unknown draws, or an empty visible queue).
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

use std::collections::VecDeque;

use rustc_hash::{FxHashMap, FxHashSet};

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
    /// The PC line length the scan covers, in plies. This is the *scan's*
    /// depth; the fallback beam's depth is the session's `max_depth` (from
    /// [`SearchBudget::max_depth`](crate::ai::search::SearchBudget)), so an
    /// interactive arm can run a deep scan over a light fallback.
    pub horizon: u8,
    /// Per-decision cap on scan node expansions; `0` = unbounded. When the
    /// cap trips, the scan finalizes early on its partial accumulators (see
    /// the module docs for why this lives here and not in the policy budget).
    pub scan_node_budget: u32,
    /// Record solved lines and serve follow-up decisions from the committed
    /// line while observations match (module docs). Off for research arms.
    pub commit_lines: bool,
    /// Search the visible queue once and fork per scenario at the first
    /// unknown draw (Layer 4). Off for registered research arms whose recorded
    /// runs must stay byte-identical; on for the interactive game arm.
    pub shared_prefix: bool,
}

impl PcCoverageConfig {
    /// Clamp degenerate values: zero caps/widths/horizons become 1, the
    /// percent caps at 100, and a [`Reveals`](PcCoverageUnit::Reveals) sample
    /// is wide enough to stratify over every possible first draw (a smaller
    /// cap could never reach its own threshold).
    fn normalized(mut self) -> Self {
        self.scenario_cap = self.scenario_cap.max(1);
        if self.unit == PcCoverageUnit::Reveals {
            self.scenario_cap = self.scenario_cap.max(PieceType::LEN);
        }
        self.width_per_root = self.width_per_root.max(1);
        self.min_coverage_percent = self.min_coverage_percent.min(100);
        self.fallback_width = self.fallback_width.max(1);
        self.horizon = self.horizon.max(1);
        self
    }
}

/// Sentinel for "no line arena entry" (root children, or recording off).
const NO_LINE: u32 = u32::MAX;

/// One node in a scenario's per-root beam frontier.
#[derive(Clone)]
struct Node {
    state: SearchState,
    acc_reward: Reward,
    root_index: usize,
    score: i32,
    /// This node's entry in the in-flight scenario's line arena (path
    /// reconstruction for the commitment cache); [`NO_LINE`] for the root
    /// children and whenever recording is off.
    line_id: u32,
    /// Prefix-arena path carried into a scenario tail after the shared-prefix
    /// branch; [`NO_LINE`] when the node was seeded at ply 1 (no prefix) or
    /// recording is off.
    prefix_line_id: u32,
}

/// A recorded solving line for one (root, first-reveal) pair.
#[derive(Clone)]
struct SolvedLine {
    /// The solving placements from ply 2 on (ply 1 is the committed root).
    path: Vec<Placement>,
    /// The reveals the line assumed *after* the branch key, confirmed one per
    /// follow-up decision while the line is being served.
    continuation: Vec<PieceType>,
}

/// Key of a recorded line: (root index, the scenario's first unknown draw —
/// `None` when the revealed queue already covered the horizon).
type LineKey = (usize, Option<PieceType>);

/// The in-flight shared-prefix search: the visible-queue plies every scenario
/// agrees on, resumable at the same parent-expansion grain as a scenario tail.
struct PrefixRun {
    /// Per-root "reached a PC during the prefix" flags (folded per scenario
    /// at tail start, keyed by that scenario's first draw).
    solved: Vec<bool>,
    frontier: Vec<Node>,
    next: Vec<Node>,
    parent_cursor: usize,
    /// Generation the frontier holds (1 = the root children).
    depth: u8,
}

/// The in-flight search of one scenario tail: per-root frontier, generation
/// cursor, and (when recording) the path arena. This is what makes the scan
/// resumable mid-scenario — `think` suspends between parent expansions.
struct ScenarioRun {
    /// The scenario's first unknown draw (the reveal-coverage key).
    first_draw: Option<PieceType>,
    /// The scenario's reveals after the first (recorded with a solved line);
    /// empty when recording is off.
    continuation_rest: Vec<PieceType>,
    /// Per-root "reached a PC in this scenario" flags.
    solved: Vec<bool>,
    /// The current generation's nodes (generation `depth`).
    frontier: Vec<Node>,
    /// Children accumulated for the next generation (truncated per root at
    /// the generation boundary).
    next: Vec<Node>,
    /// Next `frontier` index to expand; `== frontier.len()` is the boundary.
    parent_cursor: usize,
    /// Generation the frontier holds (1 = the root children).
    depth: u8,
    /// Per-surviving-child (parent line id, placement) records — the lazy
    /// path store for solved-line reconstruction. Empty when recording is off.
    arena: Vec<(u32, Placement)>,
}

/// The in-flight coverage scan: the scenario sample plus per-root
/// accumulators, advanced one root seeding or one parent expansion per
/// [`Scan::step`].
struct Scan {
    state: SearchState,
    roots: Vec<Placement>,
    /// Each root's one-ply evaluator score (`value + reward`), the coverage
    /// tiebreak — among equally covered roots, ordinary play decides.
    root_fallback: Vec<i32>,
    scenarios: Vec<Vec<PieceType>>,
    /// Next scenario to start; with `current == None` and `cursor ==
    /// scenarios.len()`, the scan is complete.
    cursor: usize,
    /// The scenario currently being searched, if any.
    current: Option<ScenarioRun>,
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
    /// Scan depth in plies (the config's `horizon`).
    horizon: u8,
    /// Per-root beam width (the config's `width_per_root`).
    width_per_root: usize,
    /// Whether solved lines are recorded (the config's `commit_lines`).
    record_lines: bool,
    /// First solving line per (root, first reveal) — the commitment source.
    lines: FxHashMap<LineKey, SolvedLine>,
    /// Plies searched in the shared prefix before per-scenario branching; `0`
    /// when the scan skips straight to whole-scenario search.
    prefix_depth: u8,
    /// The in-flight prefix search, if any.
    prefix: Option<PrefixRun>,
    /// Prefix path arena — kept after the prefix completes for line stitching.
    prefix_arena: Vec<(u32, Placement)>,
    /// Branch frontier after the prefix; `None` until the prefix finishes (or
    /// was skipped).
    prefix_frontier: Option<Vec<Node>>,
    /// Per-root prefix solves — copied into each scenario tail at start.
    prefix_solved: Vec<bool>,
    /// Nodes parked at the first-unknown-draw boundary (visible queue
    /// exhausted); merged into [`prefix_frontier`] when the prefix completes.
    prefix_branch: Vec<Node>,
    /// Length of the speculative tail appended during prefix search (=
    /// unknown draws at seed time); `0` when prefix is skipped.
    tail_len: usize,
    /// A deterministic speculative tail shared during prefix search so queue
    /// bookkeeping matches whole-scenario search until a per-scenario fork.
    canonical_tail: Vec<PieceType>,
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

/// A committed PC line being served across decisions (module docs). Lives on
/// the planner — it must survive the per-decision `Run` replacement.
enum Commitment {
    /// Right after a coverage decision: the root placement was served; the
    /// *next* reveal selects which recorded branch the line follows.
    Branching {
        /// The predicted next decision state (the scan root after the
        /// committed placement), lacking only the not-yet-seen newest reveal.
        predicted: Box<SearchState>,
        /// Recorded solving lines, keyed by first reveal (`None` = the
        /// horizon was fully revealed: a single unconditional line).
        branches: Vec<(Option<PieceType>, SolvedLine)>,
    },
    /// Mid-line: serve `path` front-first while observations keep matching.
    Following {
        predicted: Box<SearchState>,
        path: VecDeque<Placement>,
        /// Assumed reveals still to be confirmed, one per follow-up. Once
        /// empty, further reveals are beyond the line's modeled queue (they
        /// become active only after the line ends) and are accepted as-is.
        continuation: VecDeque<PieceType>,
    },
}

impl Commitment {
    /// Try to serve `observed` from this commitment: validate the prediction,
    /// confirm/select on the newest reveal, and pop the next step. Returns the
    /// placement to play plus the advanced commitment (`None` when the line's
    /// last step was just served). Any mismatch returns `None` — the caller
    /// drops the cache and rescans.
    fn follow(self, observed: &SearchState) -> Option<(Placement, Option<Commitment>)> {
        // The engine reveals exactly one new queue piece per lock, at the
        // back; the prediction was built before that reveal existed.
        let mut trimmed = observed.clone();
        let newest = trimmed.queue.pop()?;
        let (mut path, continuation) = match self {
            Commitment::Branching {
                predicted,
                branches,
            } => {
                if !matches_prediction(&trimmed, &predicted) {
                    return None;
                }
                let line = branches.into_iter().find_map(|(key, line)| {
                    (key.is_none() || key == Some(newest)).then_some(line)
                })?;
                (VecDeque::from(line.path), VecDeque::from(line.continuation))
            }
            Commitment::Following {
                predicted,
                path,
                mut continuation,
            } => {
                if !matches_prediction(&trimmed, &predicted) {
                    return None;
                }
                if let Some(&expected) = continuation.front() {
                    if newest != expected {
                        return None; // fate dealt off-line: the proof is void
                    }
                    continuation.pop_front();
                }
                (path, continuation)
            }
        };
        let placement = path.pop_front()?;
        let next = (!path.is_empty()).then(|| {
            let mut predicted = observed.clone();
            predicted.commit_placement(&placement);
            Commitment::Following {
                predicted: Box::new(predicted),
                path,
                continuation,
            }
        });
        Some((placement, next))
    }
}

/// Whether an observed decision state matches a committed line's prediction.
///
/// Everything the line's validity depends on is compared: board occupancy,
/// piece *types* (active/hold/queue), chain state, and pending garbage. The
/// active piece's **pose** is deliberately not compared: the prediction holds
/// a simulated spawn while the snapshot reports the engine's, and both are
/// fresh spawns of the same type — the served placement is rendered from the
/// observed pose either way. A pose-strict compare would only manufacture
/// misses.
fn matches_prediction(observed: &SearchState, predicted: &SearchState) -> bool {
    observed.board.columns() == predicted.board.columns()
        && observed.active.piece_type() == predicted.active.piece_type()
        && observed.hold == predicted.hold
        && observed.queue == predicted.queue
        && observed.b2b == predicted.b2b
        && observed.combo == predicted.combo
        && observed.pending == predicted.pending
        && !observed.dead
        && !predicted.dead
}

/// Synthetic score for a plan served from a committed line — the decision was
/// made by the scan that recorded the line, not a fresh evaluation.
const COMMITTED_PLAN_SCORE: i32 = 0;

/// Scenario-coverage PC planner with a TP-beam fallback (module docs).
pub struct PcCoveragePlanner {
    config: PcCoverageConfig,
    fallback: BeamPlanner,
    run: Option<Run>,
    /// The committed line currently being served, if any (commit mode only).
    commitment: Option<Commitment>,
}

impl PcCoveragePlanner {
    pub fn new(config: PcCoverageConfig) -> Self {
        let config = config.normalized();
        Self {
            config,
            fallback: BeamPlanner::transposing(config.fallback_width),
            run: None,
            commitment: None,
        }
    }

    /// Seed a scan for `state`, or rule the root out immediately (not a PC
    /// construction site, no PC arithmetically reachable within the horizon,
    /// or no legal placement). Returns the scan state plus the node
    /// expansions the seeding spent.
    fn seed(&self, state: &SearchState, eval: &dyn Evaluator) -> (ScanState, u32) {
        let config = self.config;
        let horizon = config.horizon;
        // The root-level feasibility gate is the same *necessary* condition
        // the per-child pruning applies: when it fails, no scenario can
        // solve, so skipping the scan reaches the identical verdict (None)
        // for free.
        if !pc_candidate_state(state) || !pc_feasible(state, horizon) {
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

        // `horizon` placements consume up to `horizon` queue pieces: one
        // advance per ply past the first, plus one funding pull if an
        // empty-hold swap fires (it can fire at most once per line — hold
        // stays occupied after). Continuations extend the queue to that bound.
        let unknown_draws = usize::from(horizon).saturating_sub(state.queue.len());
        let unit = if unknown_draws == 0 {
            // Nothing unknown to stratify over: a single (empty) continuation,
            // judged as one scenario.
            PcCoverageUnit::Scenarios
        } else {
            config.unit
        };
        let scenarios = match unit {
            PcCoverageUnit::Scenarios => {
                continuation_sample(state.bag, unknown_draws, config.scenario_cap)
            }
            PcCoverageUnit::Reveals => {
                continuation_sample_stratified(state.bag, unknown_draws, config.scenario_cap)
            }
        };
        let denominator = match unit {
            PcCoverageUnit::Scenarios => scenarios.len(),
            PcCoverageUnit::Reveals => state.bag.possible_pieces().len(),
        };
        let prefix_depth = if config.shared_prefix {
            shared_prefix_depth(state, horizon, unknown_draws)
        } else {
            0
        };
        let root_count = roots.len();
        let canonical_tail = if unknown_draws > 0 {
            continuation_sample(state.bag, unknown_draws, 1)
                .into_iter()
                .next()
                .unwrap_or_default()
        } else {
            Vec::new()
        };

        let scan = Box::new(Scan {
            state: state.clone(),
            root_fallback,
            scenario_hits: vec![0; roots.len()],
            reveal_hits: vec![[false; PieceType::LEN]; roots.len()],
            roots,
            scenarios,
            cursor: 0,
            current: None,
            unit,
            denominator,
            horizon,
            width_per_root: config.width_per_root,
            record_lines: config.commit_lines,
            lines: FxHashMap::default(),
            prefix_depth,
            prefix: None,
            prefix_arena: Vec::new(),
            prefix_frontier: None,
            prefix_solved: vec![false; root_count],
            prefix_branch: Vec::new(),
            tail_len: unknown_draws,
            canonical_tail,
        });
        (ScanState::Scanning(scan), nodes)
    }
}

/// Fold a freshly solved (root, scenario) into the coverage accumulators.
fn fold_hit(
    unit: PcCoverageUnit,
    scenario_hits: &mut [u32],
    reveal_hits: &mut [[bool; PieceType::LEN]],
    root_index: usize,
    first_draw: Option<PieceType>,
) {
    match unit {
        PcCoverageUnit::Scenarios => scenario_hits[root_index] += 1,
        PcCoverageUnit::Reveals => {
            // Reveals implies unknown draws (seed-time normalization), so
            // every continuation has a first piece.
            let piece = first_draw.expect("Reveals unit implies a first draw");
            reveal_hits[root_index][piece as usize] = true;
        }
    }
}

/// Walk a path arena from `line_id` up to a root child, returning the recorded
/// placements in root-first order (empty when `line_id` is [`NO_LINE`]).
fn arena_path(arena: &[(u32, Placement)], line_id: u32) -> Vec<Placement> {
    let mut path = Vec::new();
    let mut id = line_id;
    while id != NO_LINE {
        let (parent, placement) = &arena[id as usize];
        path.push(placement.clone());
        id = *parent;
    }
    path.reverse();
    path
}

/// Reconstruct a solving path (ply 2 on) by walking the scenario's line arena
/// from the solving child's parent up to a root child.
fn line_path(arena: &[(u32, Placement)], parent_line_id: u32, last: &Placement) -> Vec<Placement> {
    let mut path = arena_path(arena, parent_line_id);
    path.push(last.clone());
    path
}

/// Like [`line_path`], but prepends the shared-prefix segment when the solving
/// node branched from a prefix frontier.
fn line_path_with_prefix(
    prefix_arena: &[(u32, Placement)],
    prefix_line_id: u32,
    arena: &[(u32, Placement)],
    parent_line_id: u32,
    last: &Placement,
) -> Vec<Placement> {
    let tail = line_path(arena, parent_line_id, last);
    if prefix_line_id == NO_LINE {
        return tail;
    }
    let mut prefix = arena_path(prefix_arena, prefix_line_id);
    prefix.extend(tail);
    prefix
}

/// Plies searched once in the shared prefix before per-scenario branching.
fn shared_prefix_depth(state: &SearchState, horizon: u8, unknown_draws: usize) -> u8 {
    if unknown_draws == 0 {
        return 0;
    }
    state
        .queue
        .len()
        .min(usize::from(horizon.saturating_sub(1))) as u8
}

impl Scan {
    /// Swap the canonical tail for a scenario's continuation (and spawn when
    /// the visible queue was already exhausted at the branch).
    fn attach_scenario_tail(&self, state: &mut SearchState, continuation: &[PieceType]) {
        let vis_len = state.queue.len().saturating_sub(self.tail_len);
        state.queue.truncate(vis_len);
        state.fork_scenario_queue(continuation.iter().copied());
    }

    /// Whether every scenario has been searched (and any prefix is done).
    fn complete(&self) -> bool {
        let prefix_done = self.prefix_depth == 0 || self.prefix_frontier.is_some();
        prefix_done
            && self.prefix.is_none()
            && self.current.is_none()
            && self.cursor >= self.scenarios.len()
    }

    /// Advance the scan by one step — shared prefix, scenario tail seed, or
    /// one parent expansion — and return the node expansions spent.
    fn step(&mut self, eval: &dyn Evaluator) -> u32 {
        if self.prefix_depth > 0 && self.prefix_frontier.is_none() {
            if self.prefix.is_some() {
                return self.expand_prefix(eval);
            }
            return self.start_prefix(eval);
        }
        if self.current.is_some() {
            return self.expand_parent(eval);
        }
        self.start_scenario(eval)
    }

    /// Seed the shared prefix: score every root against the visible queue
    /// only (no continuation appended), truncate per root — generation 1.
    fn start_prefix(&mut self, eval: &dyn Evaluator) -> u32 {
        let ctx = EvalContext {
            combo: self.state.combo,
            b2b: self.state.b2b,
        };
        let canonical_tail = self.canonical_tail.clone();
        let mut frontier = Vec::new();
        let mut spent = 0u32;
        for (root_index, placement) in self.roots.iter().enumerate() {
            let (mut child, value, reward) = score_child(&self.state, placement, eval, ctx);
            spent = spent.saturating_add(1);
            if child.board.is_empty() && !child.dead {
                self.prefix_solved[root_index] = true;
            } else if !child.dead && pc_feasible(&child, self.horizon.saturating_sub(1)) {
                child.queue.extend(canonical_tail.iter().copied());
                frontier.push(Node {
                    score: (value + reward).0,
                    state: child,
                    acc_reward: reward,
                    root_index,
                    line_id: NO_LINE,
                    prefix_line_id: NO_LINE,
                });
            }
        }
        truncate_per_root(&mut frontier, self.roots.len(), self.width_per_root);

        let (branch, active): (Vec<_>, Vec<_>) = frontier
            .into_iter()
            .partition(|node| self.tail_len > 0 && node.state.queue.len() <= self.tail_len);
        self.prefix_branch = branch;

        if self.prefix_depth == 1
            || active.is_empty()
            || self.prefix_solved.iter().all(|&done| done)
        {
            let mut frontier = active;
            frontier.append(&mut self.prefix_branch);
            self.prefix_frontier = Some(frontier);
            self.prefix = None;
        } else {
            self.prefix = Some(PrefixRun {
                solved: self.prefix_solved.clone(),
                frontier: active,
                next: Vec::new(),
                parent_cursor: 0,
                depth: 1,
            });
        }
        spent
    }

    /// Expand one parent of the in-flight prefix — or, at a generation
    /// boundary, roll the frontier and finish when `depth` reaches
    /// [`Scan::prefix_depth`].
    fn expand_prefix(&mut self, eval: &dyn Evaluator) -> u32 {
        let prefix_depth = self.prefix_depth;
        let width_per_root = self.width_per_root;
        let record_lines = self.record_lines;
        let horizon = self.horizon;
        let roots_len = self.roots.len();
        let tail_len = self.tail_len;
        let canonical_tail = self.canonical_tail.clone();
        let run = self.prefix.as_mut().expect("prefix is in flight");

        if run.parent_cursor >= run.frontier.len() {
            truncate_per_root(&mut run.next, roots_len, width_per_root);
            run.frontier = std::mem::take(&mut run.next);
            run.parent_cursor = 0;
            run.depth += 1;
            let (branch, active): (Vec<_>, Vec<_>) = run
                .frontier
                .drain(..)
                .partition(|node| tail_len > 0 && node.state.queue.len() <= tail_len);
            self.prefix_branch.extend(branch);
            run.frontier = active;
            if run.depth >= prefix_depth
                || run.frontier.is_empty()
                || run.solved.iter().all(|&done| done)
            {
                self.prefix_solved.clone_from(&run.solved);
                let mut frontier = std::mem::take(&mut run.frontier);
                frontier.append(&mut self.prefix_branch);
                self.prefix_frontier = Some(frontier);
                self.prefix = None;
            }
            return 0;
        }

        let parent = run.frontier[run.parent_cursor].clone();
        run.parent_cursor += 1;
        if run.solved[parent.root_index] || parent.state.dead {
            return 0;
        }
        if tail_len > 0 && parent.state.queue.len() <= tail_len {
            self.prefix_branch.push(parent);
            return 0;
        }
        let remaining = horizon.saturating_sub(run.depth + 1);
        let ctx = EvalContext {
            combo: parent.state.combo,
            b2b: parent.state.b2b,
        };
        let mut spent = 0u32;
        for placement in hold_placements(&parent.state) {
            let (mut child, value, reward) = score_child(&parent.state, &placement, eval, ctx);
            spent = spent.saturating_add(1);
            if child.board.is_empty() && !child.dead {
                run.solved[parent.root_index] = true;
                self.prefix_solved[parent.root_index] = true;
                continue;
            }
            if child.dead || !pc_feasible(&child, remaining) {
                continue;
            }
            let acc = parent.acc_reward + reward;
            let line_id = if record_lines {
                self.prefix_arena.push((parent.line_id, placement));
                (self.prefix_arena.len() - 1) as u32
            } else {
                NO_LINE
            };
            child.queue.extend(canonical_tail.iter().copied());
            run.next.push(Node {
                score: (value + acc).0,
                state: child,
                acc_reward: acc,
                root_index: parent.root_index,
                line_id,
                prefix_line_id: NO_LINE,
            });
        }
        spent
    }

    /// Seed the scenario at `cursor`: either branch the shared-prefix frontier
    /// with this scenario's continuation, or (when `prefix_depth == 0`) score
    /// every root against the extended queue — generation 1.
    fn start_scenario(&mut self, eval: &dyn Evaluator) -> u32 {
        let continuation = &self.scenarios[self.cursor];
        let first_draw = continuation.first().copied();

        if let Some(branch) = self.prefix_frontier.as_ref() {
            let solved = self.prefix_solved.clone();
            for (root_index, &is_solved) in solved.iter().enumerate() {
                if is_solved {
                    fold_hit(
                        self.unit,
                        &mut self.scenario_hits,
                        &mut self.reveal_hits,
                        root_index,
                        first_draw,
                    );
                }
            }
            if self.horizon <= self.prefix_depth
                || branch.is_empty()
                || solved.iter().all(|&done| done)
            {
                self.cursor += 1;
                return 0;
            }
            let mut frontier = Vec::new();
            for mut node in branch.iter().cloned() {
                if solved[node.root_index] {
                    continue;
                }
                self.attach_scenario_tail(&mut node.state, continuation);
                frontier.push(Node {
                    score: node.score,
                    state: node.state,
                    acc_reward: node.acc_reward,
                    root_index: node.root_index,
                    line_id: NO_LINE,
                    prefix_line_id: node.line_id,
                });
            }
            if frontier.is_empty() {
                self.cursor += 1;
                return 0;
            }
            self.current = Some(ScenarioRun {
                first_draw,
                continuation_rest: if self.record_lines {
                    continuation.get(1..).unwrap_or(&[]).to_vec()
                } else {
                    Vec::new()
                },
                solved,
                frontier,
                next: Vec::new(),
                parent_cursor: 0,
                depth: self.prefix_depth,
                arena: Vec::new(),
            });
            return 0;
        }

        let Self {
            state,
            roots,
            scenarios,
            cursor,
            current,
            unit,
            scenario_hits,
            reveal_hits,
            horizon,
            width_per_root,
            record_lines,
            ..
        } = self;
        let continuation = &scenarios[*cursor];
        let first_draw = continuation.first().copied();
        let mut base = state.clone();
        // The bag is deliberately NOT advanced: the extended queue covers the
        // whole horizon (seed-time bound), so no search path ever deals from
        // it, and the per-scenario transposition key below need not include it.
        base.queue.extend(continuation.iter().copied());
        let ctx = EvalContext {
            combo: base.combo,
            b2b: base.b2b,
        };

        let mut solved = vec![false; roots.len()];
        let mut frontier = Vec::new();
        let mut spent = 0u32;
        for (root_index, placement) in roots.iter().enumerate() {
            let (child, value, reward) = score_child(&base, placement, eval, ctx);
            spent = spent.saturating_add(1);
            if child.board.is_empty() && !child.dead {
                if !solved[root_index] {
                    solved[root_index] = true;
                    // A ply-1 solve leaves no follow-up to commit (the board
                    // is already clean), so no line is recorded.
                    fold_hit(*unit, scenario_hits, reveal_hits, root_index, first_draw);
                }
            } else if !child.dead && pc_feasible(&child, horizon.saturating_sub(1)) {
                frontier.push(Node {
                    score: (value + reward).0,
                    state: child,
                    acc_reward: reward,
                    root_index,
                    line_id: NO_LINE,
                    prefix_line_id: NO_LINE,
                });
            }
        }
        truncate_per_root(&mut frontier, roots.len(), *width_per_root);

        if *horizon == 1 || frontier.is_empty() || solved.iter().all(|&done| done) {
            *cursor += 1; // scenario settled at the roots
        } else {
            *current = Some(ScenarioRun {
                first_draw,
                continuation_rest: if *record_lines {
                    continuation.get(1..).unwrap_or(&[]).to_vec()
                } else {
                    Vec::new()
                },
                solved,
                frontier,
                next: Vec::new(),
                parent_cursor: 0,
                depth: 1,
                arena: Vec::new(),
            });
        }
        spent
    }

    /// Expand one parent of the in-flight scenario into the next generation —
    /// or, at a generation boundary, truncate and roll the frontier over
    /// (finishing the scenario when it is exhausted, solved out, or at the
    /// horizon). Equivalent, step by step, to the original whole-scenario
    /// search: same expansion order, same truncation points, same hits.
    fn expand_parent(&mut self, eval: &dyn Evaluator) -> u32 {
        let Self {
            roots,
            cursor,
            current,
            unit,
            scenario_hits,
            reveal_hits,
            horizon,
            width_per_root,
            record_lines,
            lines,
            prefix_arena,
            ..
        } = self;
        let run = current.as_mut().expect("a scenario is in flight");

        if run.parent_cursor >= run.frontier.len() {
            // Generation boundary: settle `next` into the new frontier.
            truncate_per_root(&mut run.next, roots.len(), *width_per_root);
            run.frontier = std::mem::take(&mut run.next);
            run.parent_cursor = 0;
            run.depth += 1;
            if run.depth >= *horizon
                || run.frontier.is_empty()
                || run.solved.iter().all(|&done| done)
            {
                *cursor += 1;
                *current = None;
            }
            return 0;
        }

        let parent = run.frontier[run.parent_cursor].clone();
        run.parent_cursor += 1;
        if run.solved[parent.root_index] || parent.state.dead {
            return 0; // this root's verdict for the scenario is already in
        }
        let remaining = *horizon - (run.depth + 1);
        let ctx = EvalContext {
            combo: parent.state.combo,
            b2b: parent.state.b2b,
        };
        let mut spent = 0u32;
        for placement in hold_placements(&parent.state) {
            let (child, value, reward) = score_child(&parent.state, &placement, eval, ctx);
            spent = spent.saturating_add(1);
            if child.board.is_empty() && !child.dead {
                if !run.solved[parent.root_index] {
                    run.solved[parent.root_index] = true;
                    fold_hit(
                        *unit,
                        scenario_hits,
                        reveal_hits,
                        parent.root_index,
                        run.first_draw,
                    );
                    if *record_lines {
                        lines
                            .entry((parent.root_index, run.first_draw))
                            .or_insert_with(|| SolvedLine {
                                path: line_path_with_prefix(
                                    prefix_arena,
                                    parent.prefix_line_id,
                                    &run.arena,
                                    parent.line_id,
                                    &placement,
                                ),
                                continuation: run.continuation_rest.clone(),
                            });
                    }
                }
                continue;
            }
            if child.dead || !pc_feasible(&child, remaining) {
                continue;
            }
            let acc = parent.acc_reward + reward;
            let line_id = if *record_lines {
                run.arena.push((parent.line_id, placement));
                (run.arena.len() - 1) as u32
            } else {
                NO_LINE
            };
            run.next.push(Node {
                score: (value + acc).0,
                state: child,
                acc_reward: acc,
                root_index: parent.root_index,
                line_id,
                prefix_line_id: parent.prefix_line_id,
            });
        }
        spent
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

    /// The scan's pick on the accumulators so far: the most-covered root at
    /// or above the threshold, evaluator score then canonical order breaking
    /// ties (first maximum wins, the determinism rule every planner here
    /// follows). On a completed scan this is the final verdict; mid-scan it
    /// is the (conservative) partial verdict — hits only ever grow, and the
    /// denominator is always the full one.
    fn verdict(&self, min_coverage_percent: u8) -> Option<PlacementPlan> {
        self.verdict_indexed(min_coverage_percent)
            .map(|(plan, _)| plan)
    }

    /// [`verdict`](Self::verdict) plus the picked root's index (the line
    /// commitment needs to know whose branches to commit).
    fn verdict_indexed(&self, min_coverage_percent: u8) -> Option<(PlacementPlan, usize)> {
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
        best.map(|(covered, fallback, root_index)| {
            (
                PlacementPlan {
                    placement: self.roots[root_index].clone(),
                    // Packed for log readability only (coverage dominates);
                    // the selection above is the decision.
                    score: (covered as i32)
                        .saturating_mul(1_000_000)
                        .saturating_add(fallback),
                },
                root_index,
            )
        })
    }

    /// Build the commitment for the verdict's `root_index` from the recorded
    /// lines: the predicted post-placement state plus the root's branch
    /// table. `None` when nothing followable was recorded (e.g. every solve
    /// was at ply 1). Drains the line store — the scan is finished.
    fn take_commitment(&mut self, root_index: usize) -> Option<Commitment> {
        let lines = std::mem::take(&mut self.lines);
        let mut branches: Vec<(Option<PieceType>, SolvedLine)> = lines
            .into_iter()
            .filter(|((root, _), line)| *root == root_index && !line.path.is_empty())
            .map(|((_, key), line)| (key, line))
            .collect();
        if branches.is_empty() {
            return None;
        }
        // The hash map's iteration order is arbitrary; restore the canonical
        // piece order (selection is by key equality, this is for inspection).
        branches.sort_by_key(|(key, _)| key.map_or(usize::MAX, |piece| piece as usize));
        let mut predicted = self.state.clone();
        predicted.commit_placement(&self.roots[root_index]);
        Some(Commitment::Branching {
            predicted: Box::new(predicted),
            branches,
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
        // A genuinely new decision: first offer it to the committed line.
        // `take` makes the attempt one-shot — any miss drops the cache.
        let followed = self
            .commitment
            .take()
            .and_then(|commitment| commitment.follow(state));
        self.fallback.reroot(state, eval, max_depth);
        if let Some((placement, next)) = followed {
            self.commitment = next;
            self.run = Some(Run {
                root_key,
                max_depth,
                scan: ScanState::Done(Some(PlacementPlan {
                    placement,
                    score: COMMITTED_PLAN_SCORE,
                })),
                nodes: 0,
            });
            return; // served from the line: zero search this decision
        }
        let (scan, nodes) = self.seed(state, eval);
        self.run = Some(Run {
            root_key,
            max_depth,
            scan,
            nodes,
        });
    }

    /// Node-grain: spends `quantum` in `score_child` units (overshooting by
    /// at most one parent expansion), suspending mid-scenario; once the scan
    /// settles — exhausted or budget-cut — the call stream drives the
    /// fallback beam instead.
    fn think(&mut self, quantum: u32, eval: &dyn Evaluator) -> ThinkProgress {
        let config = self.config;
        let Some(run) = self.run.as_mut() else {
            return ThinkProgress::Exhausted; // never rooted: nothing to think about
        };
        if let ScanState::Scanning(scan) = &mut run.scan {
            let budget = config.scan_node_budget;
            let mut spent = 0u32;
            while spent < quantum
                && !scan.complete()
                && (budget == 0 || run.nodes.saturating_add(spent) < budget)
            {
                spent = spent.saturating_add(scan.step(eval));
            }
            run.nodes = run.nodes.saturating_add(spent);
            let budget_cut = budget != 0 && run.nodes >= budget;
            if scan.complete() || budget_cut {
                // Finalize — on the full accumulators or the budget-cut
                // partials (same code path, so the cut is step-aligned and
                // slicing-invariant either way).
                let verdict = scan.verdict_indexed(config.min_coverage_percent);
                self.commitment = match &verdict {
                    Some((_, root_index)) if config.commit_lines => {
                        scan.take_commitment(*root_index)
                    }
                    _ => None,
                };
                run.scan = ScanState::Done(verdict.map(|(plan, _)| plan));
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
            // Mid-scan: the partial verdict if something already cleared the
            // (full-denominator) threshold, else the fallback beam's anytime
            // answer — it was seeded at reroot, so this is valid immediately.
            Some(ScanState::Scanning(scan)) => scan
                .verdict(self.config.min_coverage_percent)
                .or_else(|| self.fallback.best()),
            _ => self.fallback.best(),
        }
    }

    fn nodes_expanded(&self) -> u32 {
        self.run.as_ref().map_or(0, |run| run.nodes) + self.fallback.nodes_expanded()
    }
}

/// Total occupied cells and stack height (topmost occupied row + 1) of `state`'s
/// board, both read straight off the column bitboard — the two metrics the PC
/// feasibility and candidate gates share.
fn board_cells_and_height(state: &SearchState) -> (usize, usize) {
    let columns = state.board.columns();
    let cells = columns
        .iter()
        .map(|column| column.count_ones() as usize)
        .sum();
    let height = columns
        .iter()
        .map(|column| (u64::BITS - column.leading_zeros()) as usize)
        .max()
        .unwrap_or(0);
    (cells, height)
}

/// Necessary conditions for a PC within `remaining` more pieces: some piece
/// count makes the total cells a whole number of rows, and the current stack
/// is no taller than the rows that would all clear. Never prunes a reachable
/// PC (both conditions hold on every true PC line).
fn pc_feasible(state: &SearchState, remaining: u8) -> bool {
    if state.board.is_empty() {
        return true;
    }
    let (cells, height) = board_cells_and_height(state);
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
    let (cells, height) = board_cells_and_height(state);
    cells <= 40 && height <= 6
}

/// Keep the best `width` nodes **per root** (stable order: ties keep canonical
/// enumeration order), deduplicating transposed states within a root. The bag
/// is constant across a scenario (see [`Scan::start_scenario`]), so
/// [`RootKey`] alone identifies a future here.
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
/// least one continuation — placed **first**, one per reveal — before the rest
/// of the cap fills in hash order. Both the [`Reveals`](PcCoverageUnit::Reveals)
/// denominator and the budget-cut scan rely on that order: a truncated scan
/// has judged every reveal once before any is refined.
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
/// preview and the registered horizon-10 arms this is ≤ 2,520 sequences —
/// check this bound before raising the horizon or shrinking the preview.
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

    fn config(unit: PcCoverageUnit, horizon: u8) -> PcCoverageConfig {
        PcCoverageConfig {
            scenario_cap: 8,
            width_per_root: 8,
            min_coverage_percent: 100,
            fallback_width: 8,
            unit,
            horizon,
            scan_node_budget: 0,
            commit_lines: false,
            shared_prefix: false,
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
        // last column perfect-clears in one move. The horizon (1) is the
        // config's, NOT the budget's depth (3) — this also pins the
        // horizon/fallback-depth decoupling.
        let mut board = Board::new(4, 8);
        for y in 0..4 {
            for x in 0..3 {
                board.set(x, y, CellKind::Some(PieceType::O));
            }
        }
        let active = spawn_piece(PieceType::I, 4, 8);
        let state = SearchState::for_test(board, active, None, [PieceType::T]);
        let mut planner = PcCoveragePlanner::new(config(PcCoverageUnit::Scenarios, 1));
        let plan = think_to_completion(
            &mut planner,
            &state,
            &LinearEvaluator::default(),
            SearchBudget::beam(3),
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
        let mut planner = PcCoveragePlanner::new(config(PcCoverageUnit::Reveals, 3));
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
        // fallback's plan, or a partial verdict) at every suspension point in
        // between. Quantum-1 is the adversarial grain: every single step is a
        // suspension point.
        let mut board = Board::new(4, 10);
        for x in 0..2 {
            board.set(x, 0, CellKind::Some(PieceType::O));
            board.set(x, 1, CellKind::Some(PieceType::O));
        }
        let active = spawn_piece(PieceType::O, 4, 10);
        // Short queue + horizon 4 ⇒ unknown draws ⇒ a real multi-scenario scan.
        let state = SearchState::for_test(board, active, None, [PieceType::I]);
        let eval = LinearEvaluator::default();

        let mut drained = PcCoveragePlanner::new(config(PcCoverageUnit::Reveals, 4));
        let one_shot = think_to_completion(&mut drained, &state, &eval, SearchBudget::beam(4))
            .expect("a plan");

        let mut sliced = PcCoveragePlanner::new(config(PcCoverageUnit::Reveals, 4));
        sliced.reroot(&state, &eval, 4);
        assert!(sliced.best().is_some(), "anytime best right after reroot");
        let mut guard = 0;
        while sliced.think(1, &eval) == ThinkProgress::Working {
            assert!(sliced.best().is_some(), "anytime best mid-scan");
            guard += 1;
            assert!(guard < 1_000_000, "think never exhausted");
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

    #[test]
    fn scan_budget_cut_is_slicing_invariant() {
        // A budgeted scan finalizes on its partial accumulators at the SAME
        // step boundary whether thinking is drained in one call or sliced at
        // quantum 1 — the internal-budget design's whole point. (And the
        // decision is still valid: partial verdict or the fallback's plan.)
        let mut board = Board::new(4, 10);
        for x in 0..2 {
            board.set(x, 0, CellKind::Some(PieceType::O));
            board.set(x, 1, CellKind::Some(PieceType::O));
        }
        let active = spawn_piece(PieceType::O, 4, 10);
        let state = SearchState::for_test(board, active, None, [PieceType::I]);
        let eval = LinearEvaluator::default();
        let budgeted = PcCoverageConfig {
            scan_node_budget: 120, // cuts well inside the multi-scenario scan
            ..config(PcCoverageUnit::Reveals, 4)
        };

        let mut drained = PcCoveragePlanner::new(budgeted);
        let one_shot = think_to_completion(&mut drained, &state, &eval, SearchBudget::beam(4))
            .expect("a plan");

        let mut sliced = PcCoveragePlanner::new(budgeted);
        sliced.reroot(&state, &eval, 4);
        let mut guard = 0;
        while sliced.think(1, &eval) == ThinkProgress::Working {
            guard += 1;
            assert!(guard < 1_000_000, "think never exhausted");
        }
        let step_plan = sliced.best().expect("a plan");

        assert_eq!(one_shot.placement.origin(), step_plan.placement.origin());
        assert_eq!(one_shot.placement.path, step_plan.placement.path);
        assert_eq!(one_shot.score, step_plan.score);
        assert_eq!(
            drained.nodes_expanded(),
            sliced.nodes_expanded(),
            "the cut must land on the same step boundary under any slicing"
        );
    }

    #[test]
    fn a_committed_line_serves_the_follow_up_with_zero_search() {
        // A two-move PC: 4-wide board with columns 0-1 filled four high;
        // vertical I pieces in columns 2 and 3 clear it. The first decision
        // scans and commits the line; the follow-up decision must be served
        // from the commitment instantly — `think` exhausts on its very first
        // quantum-1 call, impossible for a fresh scan — and must complete
        // the perfect clear.
        let mut board = Board::new(4, 8);
        for y in 0..4 {
            for x in 0..2 {
                board.set(x, y, CellKind::Some(PieceType::O));
            }
        }
        let active = spawn_piece(PieceType::I, 4, 8);
        let state = SearchState::for_test(board, active, None, [PieceType::I]);
        let eval = LinearEvaluator::default();
        let committing = PcCoverageConfig {
            commit_lines: true,
            min_coverage_percent: 100,
            ..config(PcCoverageUnit::Reveals, 2)
        };

        let mut planner = PcCoveragePlanner::new(committing);
        let first = think_to_completion(&mut planner, &state, &eval, SearchBudget::beam(2))
            .expect("the opening move of the 2-move PC");

        // Reality follows the plan: play the move, the engine deals a piece.
        let mut next_state = state.clone();
        next_state.commit_placement(&first.placement);
        assert!(
            !next_state.board.is_empty(),
            "the PC needs the second move still"
        );
        next_state.queue.push(PieceType::T); // the newest reveal

        planner.reroot(&next_state, &eval, 2);
        assert_eq!(
            planner.think(1, &eval),
            ThinkProgress::Exhausted,
            "the committed line must serve the follow-up with zero search"
        );
        let second = planner.best().expect("the committed step");
        let mut done = next_state.clone();
        done.commit_placement(&second.placement);
        assert!(
            done.board.is_empty(),
            "the committed step must complete the perfect clear"
        );
    }

    #[test]
    fn a_mismatched_observation_drops_the_commitment_and_rescans() {
        // Same setup as above, but reality diverges (a garbage row appears
        // under the bot): the commitment must be dropped — the planner
        // rescans instead of serving a now-invalid step.
        let mut board = Board::new(4, 8);
        for y in 0..4 {
            for x in 0..2 {
                board.set(x, y, CellKind::Some(PieceType::O));
            }
        }
        let active = spawn_piece(PieceType::I, 4, 8);
        let state = SearchState::for_test(board, active, None, [PieceType::I]);
        let eval = LinearEvaluator::default();
        let committing = PcCoverageConfig {
            commit_lines: true,
            ..config(PcCoverageUnit::Reveals, 2)
        };

        let mut planner = PcCoveragePlanner::new(committing);
        think_to_completion(&mut planner, &state, &eval, SearchBudget::beam(2))
            .expect("the opening move");

        // A diverged reality: the planned placement never happened (the board
        // is unchanged), yet a new piece arrived.
        let mut diverged = state.clone();
        diverged.queue.push(PieceType::T);

        planner.reroot(&diverged, &eval, 2);
        assert_eq!(
            planner.think(1, &eval),
            ThinkProgress::Working,
            "a mismatch must fall through to a fresh scan, not serve the line"
        );
    }

    #[test]
    fn the_shipping_config_serves_a_prefix_stitched_line() {
        // The exact in-game / pc-watch-v1 flag combo — commit_lines AND
        // shared_prefix ON together (Reveals, min_coverage 25) — which no other
        // test exercises. With a visible queue and an unknown draw the scan runs
        // a real prefix phase (prefix_depth = min(queue, horizon-1) = 1), so the
        // committed line is stitched across the prefix arena and the scenario
        // arena (`line_path_with_prefix`). The follow-up must be served with
        // zero search and complete the perfect clear.
        let mut board = Board::new(4, 8);
        for y in 0..4 {
            for x in 0..2 {
                board.set(x, y, CellKind::Some(PieceType::O));
            }
        }
        let active = spawn_piece(PieceType::I, 4, 8);
        let state = SearchState::for_test(board, active, None, [PieceType::I]);
        let eval = LinearEvaluator::default();
        let shipping = PcCoverageConfig {
            commit_lines: true,
            shared_prefix: true,
            min_coverage_percent: 25,
            ..config(PcCoverageUnit::Reveals, 2)
        };

        let mut planner = PcCoveragePlanner::new(shipping);
        let first = think_to_completion(&mut planner, &state, &eval, SearchBudget::beam(2))
            .expect("the opening move of the 2-move PC");

        let mut next_state = state.clone();
        next_state.commit_placement(&first.placement);
        assert!(
            !next_state.board.is_empty(),
            "the PC needs the second move still"
        );
        next_state.queue.push(PieceType::T); // the newest reveal

        planner.reroot(&next_state, &eval, 2);
        assert_eq!(
            planner.think(1, &eval),
            ThinkProgress::Exhausted,
            "the prefix-stitched committed line must serve the follow-up with zero search"
        );
        let second = planner.best().expect("the committed step");
        let mut done = next_state.clone();
        done.commit_placement(&second.placement);
        assert!(
            done.board.is_empty(),
            "the committed step must complete the perfect clear"
        );
    }
}
