//! The AI model registry: the catalog of "brains" a bot seat can run.
//!
//! Each `ModelEntry` names a bot and knows how to build a fresh `AiController`
//! for it. The shipped catalog spans the linear DT-20 evaluator and the ported
//! Cold Clear 2 attack evaluator, on greedy / beam / best-first search.
//! **Adding a model is one entry in [`ModelRegistry::default`].**
//!
//! The setup screens (`crate::screens`) render [`labels`](ModelRegistry::labels)
//! and write the selection; the session's seat spawner builds a controller per
//! bot seat when a session starts. Difficulty is the shared `beatable()` handicap
//! for every entry: only the *model* (the planner + board evaluator) differs, so
//! picks compare like-for-like: greedy vs beam, linear eval vs ported CC2 eval.

use bevy::prelude::*;

use std::time::Duration;

use crate::ai::{
    AiController, BeamPlanner, Cc2Evaluator, Cc2Weights, DEFAULT_AI_SEED, Evaluator, Handicap,
    LinearEvaluator, Mind, PcCoverageConfig, PcCoveragePlanner, PcCoverageUnit, SearchBudget,
};

/// Beam settings for the in-game Tier-2 bots. Depth 2 is smooth per piece (a few ms
/// in release) and already ~+26% over greedy on the marathon benchmark; width 16
/// matches the bench. The headless `bench-marathon` explores deeper (depth 3 ≈ +33%).
const BEAM_WIDTH: usize = 16;
const BEAM_DEPTH: u8 = 2;

/// The PC Hunter's scan horizon: the PC line length the coverage scan covers.
/// Lines complete around lock 10 (a 10-piece line fills 40 cells).
const PC_HORIZON: u8 = 10;

/// Per-decision cap on the PC Hunter's coverage scan, in evaluator-call
/// units. At the measured release rate (~600 evals/ms) this bounds the
/// worst-case decision near 100 ms of compute — spread at [`PC_QUANTUM`] per
/// frame, ≈0.5 s of wall clock, the budgeted "thinking pause" a committed
/// line then amortizes over its whole length (follow-ups are zero-search).
const PC_SCAN_NODE_BUDGET: u32 = 60_000;

/// Fallback TP-beam depth for the PC Hunter when the scan abstains —
/// decoupled from the scan's [`PC_HORIZON`] (depth 10 cost the catalog norm
/// ×10 for ordinary-play moves the beam decides anyway).
const PC_FALLBACK_DEPTH: u8 = 3;

/// Per-poll node quantum for the PC Hunter's sliced venue. The shared
/// default (32) is sized for best-first node *expansions*; the PC scan
/// meters single evaluator calls, ~100× cheaper each, so its frame slice is
/// correspondingly larger: ≈3 ms at the measured ~600 evals/ms — same
/// convention as `ATTACK_NODE_BUDGET`, a configured count, never a clock.
const PC_QUANTUM: u32 = 2_000;

/// One selectable AI model: a short display name (sized for a menu row), a
/// one-line detail blurb (the picker's focus-driven description pane), and a
/// factory for a fresh controller.
///
/// The factory is `Send + Sync` (it only *builds* a controller); the produced
/// [`AiController`] is `Send`-but-not-`Sync` and lives in the non-send
/// `SessionBots`.
struct ModelEntry {
    /// Short name — must fit a 220 px menu row (pinned by `labels_fit_a_menu_row`).
    label: String,
    /// One-line description shown for the focused row; wraps in the detail pane.
    detail: String,
    build: Box<dyn Fn() -> AiController + Send + Sync>,
}

impl ModelEntry {
    fn new(
        label: impl Into<String>,
        detail: impl Into<String>,
        build: impl Fn() -> AiController + Send + Sync + 'static,
    ) -> Self {
        Self {
            label: label.into(),
            detail: detail.into(),
            build: Box::new(build),
        }
    }
}

/// The catalog of AI models plus the current selection. Inserted by the
/// [`GamePlugin`](crate::GamePlugin); read by the setup screens and the
/// session's seat spawner.
///
/// Invariant: `entries` is non-empty ([`Default`] always populates it) and
/// `selected` is always in bounds (it starts at 0 and [`select`](Self::select)
/// bounds-checks) — so the accessors index directly instead of carrying dead
/// fallback arms.
#[derive(Resource)]
pub struct ModelRegistry {
    entries: Vec<ModelEntry>,
    selected: usize,
}

impl ModelRegistry {
    /// Display labels in registry order — what the picker renders, one row each.
    pub fn labels(&self) -> Vec<String> {
        self.entries.iter().map(|e| e.label.clone()).collect()
    }

    /// Select model `index` (out-of-range indices are ignored).
    pub fn select(&mut self, index: usize) {
        if index < self.entries.len() {
            self.selected = index;
        }
    }

    /// The currently selected index (the picker opens focused here).
    pub fn selected_index(&self) -> usize {
        self.selected
    }

    /// The selected model's label (for the log line / HUD).
    pub fn selected_label(&self) -> &str {
        &self.entries[self.selected].label
    }

    /// The one-line description of model `index` (the picker's detail pane).
    /// Out-of-range reads as empty rather than panicking — the picker's focus
    /// cursor is bounded by `labels().len()`, but a text pane never needs to.
    pub fn detail(&self, index: usize) -> &str {
        self.entries.get(index).map_or("", |e| e.detail.as_str())
    }

    /// Build a fresh [`AiController`] for the selected model.
    pub fn selected_controller(&self) -> AiController {
        (self.entries[self.selected].build)()
    }

    /// The label of model `index` (out of range reads empty — same contract as
    /// [`detail`](Self::detail)). Versus seat HUDs and pickers read this.
    pub fn label(&self, index: usize) -> &str {
        self.entries.get(index).map_or("", |e| e.label.as_str())
    }

    /// Build a fresh [`AiController`] for model `index`, if it exists. The
    /// versus mode builds *two* controllers (possibly the same model twice), so
    /// it addresses the catalog directly instead of going through the single
    /// `selected` cursor the Watch-AI flow uses.
    pub fn build(&self, index: usize) -> Option<AiController> {
        self.entries.get(index).map(|e| (e.build)())
    }

    /// Number of models in the catalog (the picker's cycle length).
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Standard companion to [`len`](Self::len) (the catalog is never empty,
    /// but clippy rightly insists the pair exists together).
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

/// Wire a mind + evaluator into a fresh controller — the core's
/// [`AiController::interactive`] convention (default handicap + AI seed +
/// cooperative venue), so an entry differs only by the (mind, evaluator,
/// budget) triple it names and the game can never fork the operating
/// conventions from the core's.
fn search_model(
    mind: Box<dyn Mind>,
    eval: Box<dyn Evaluator>,
    budget: SearchBudget,
) -> AiController {
    AiController::interactive(mind, eval, budget)
}

impl Default for ModelRegistry {
    fn default() -> Self {
        let mut entries = Vec::new();

        // Always available: the shipped linear DT-20 / SURVIVAL evaluator (greedy).
        entries.push(ModelEntry::new(
            "Greedy DT-20",
            "The baseline: one-piece greedy search over the linear DT-20 board \
             evaluator — the original shipped opponent.",
            AiController::beatable,
        ));

        // The Tier-2 beam: deterministic multi-ply search over the SAME linear eval.
        // It reads `LinearEvaluator::default()`, so `weights.rs` tuning flows in free.
        entries.push(ModelEntry::new(
            "Beam DT-20",
            "Deterministic multi-ply beam search over the same linear evaluator — \
             the Tier-2 architecture jump over greedy.",
            || {
                search_model(
                    Box::new(BeamPlanner::new(BEAM_WIDTH)),
                    Box::new(LinearEvaluator::default()),
                    SearchBudget::beam(BEAM_DEPTH),
                )
            },
        ));

        // Cold Clear 2's evaluator, ported (`Cc2Evaluator`) on the same beam — watch
        // the bot we benchmarked against play here, tetris-well / T-spin style and all.
        entries.push(ModelEntry::new(
            "Beam CC2",
            "Cold Clear 2's board evaluator, ported verbatim, on our beam — the \
             benchmark rival's style on this engine.",
            || {
                search_model(
                    Box::new(BeamPlanner::new(BEAM_WIDTH)),
                    Box::new(Cc2Evaluator::default()),
                    SearchBudget::beam(BEAM_DEPTH),
                )
            },
        ));

        // The APP-sprint attack bot: CC2's evaluator with the APP-climbed board
        // weights on a deeper beam. Depth 3 keeps the per-piece search watchable
        // in-browser; the headless bench runs it deeper.
        entries.push(ModelEntry::new(
            "Beam CC2 Attack",
            "CC2's evaluator with board weights climbed for attack-per-piece, on a \
             deeper beam — concentrated B2B Tetris and T-spin offense.",
            || {
                search_model(
                    Box::new(BeamPlanner::new(BEAM_WIDTH)),
                    Box::new(Cc2Evaluator::new(Cc2Weights::attack_tuned())),
                    SearchBudget::beam(3), // deeper than the default 2 — attack tuning + combos need lookahead
                )
            },
        ));

        // The strongest model — the shared `AiController::attack` factory (one home
        // for the operating point; the wasm embed builds the same brain).
        entries.push(ModelEntry::new(
            "Best-First Attack",
            "The strongest model: best-first graph search with transposition over \
             the tuned attack evaluator. Also the brain of the web embed.",
            || AiController::attack(Handicap::default(), DEFAULT_AI_SEED),
        ));

        // The perfect-clear hunter: the research crate's coverage planner at
        // the registered `pc-watch-v1` operating point (reveal coverage, 14
        // scenarios, width 2; horizon-10 scan over a depth-3 fallback). The
        // pace machinery: the scan is node-grain and budgeted at 60k evals
        // (≈0.5 s worst case, sliced at PC_QUANTUM ≈ 3 ms/frame — no frame
        // hitches), and committed PC lines serve follow-up pieces with zero
        // search, so the full scan runs about once per line, not per piece.
        // Two deliberate deviations from the catalog's shared conventions,
        // both about the model's character: imperfection 0 (one misplaced
        // piece kills a ten-piece PC line — the shared 0.12 would erase what
        // this entry exists to show, and would desync the committed line)
        // while the human-feel reaction stays; and the custom venue quantum
        // below.
        entries.push(ModelEntry::new(
            "PC Hunter",
            "Perfect-clear builder: covers the bag's possible continuations and \
             keeps boards PC-alive; plays precisely (no imperfection), with a \
             general attack beam as its fallback.",
            || {
                AiController::interactive_with(
                    Box::new(PcCoveragePlanner::new(PcCoverageConfig {
                        scenario_cap: 14,
                        width_per_root: 2,
                        min_coverage_percent: 25,
                        fallback_width: 32,
                        unit: PcCoverageUnit::Reveals,
                        horizon: PC_HORIZON,
                        scan_node_budget: PC_SCAN_NODE_BUDGET,
                        commit_lines: true,
                        shared_prefix: true,
                    })),
                    Box::new(Cc2Evaluator::new(Cc2Weights::attack_tuned())),
                    SearchBudget::beam(PC_FALLBACK_DEPTH),
                    Handicap {
                        reaction: Duration::from_millis(200),
                        imperfection: 0.0,
                    },
                    PC_QUANTUM,
                )
            },
        ));

        Self {
            entries,
            selected: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn select_clamps_to_the_catalog() {
        let mut reg = ModelRegistry::default();
        let last = reg.labels().len() - 1;
        reg.select(last);
        assert_eq!(reg.selected, last);
        reg.select(usize::MAX); // out of range: ignored, selection unchanged
        assert_eq!(reg.selected, last);
    }

    #[test]
    fn selected_label_tracks_selection() {
        let mut reg = ModelRegistry::default();
        for (i, label) in reg.labels().into_iter().enumerate() {
            reg.select(i);
            assert_eq!(reg.selected_label(), label);
            assert_eq!(reg.selected_index(), i);
        }
    }

    #[test]
    fn labels_fit_a_menu_row() {
        // The picker renders labels in a 320 px row, and the pixel font runs ~15 px
        // per glyph at the button size — so 17 characters (~255 px + padding) is the
        // budget. Longer names wrap onto a second line inside the fixed-height row
        // (the original overflow bug); descriptions belong in the detail pane.
        let reg = ModelRegistry::default();
        for label in reg.labels() {
            assert!(
                label.len() <= 17,
                "label {label:?} ({} chars) would wrap in the 320px menu row",
                label.len()
            );
        }
    }

    #[test]
    fn every_entry_has_a_detail() {
        let reg = ModelRegistry::default();
        for i in 0..reg.labels().len() {
            assert!(
                !reg.detail(i).is_empty(),
                "entry {i} is missing its detail blurb"
            );
        }
        assert_eq!(reg.detail(usize::MAX), "", "out of range reads empty");
    }

    #[test]
    fn every_entry_builds_a_controller() {
        // Each factory must construct without panicking — the registry-level smoke
        // test that a catalog edit cannot ship an unbuildable model.
        let mut reg = ModelRegistry::default();
        for i in 0..reg.labels().len() {
            reg.select(i);
            let _ = reg.selected_controller();
        }
    }
}
