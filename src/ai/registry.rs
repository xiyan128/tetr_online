//! The AI model registry: the catalog of "brains" a bot seat can run.
//!
//! Each `ModelEntry` names a bot and knows how to build a fresh `AiController`
//! for it. The shipped catalog spans the linear DT-20 evaluator and the ported
//! Cold Clear 2 attack evaluator, on greedy / beam / best-first search.
//! **Adding a model is one entry in [`ModelRegistry::default`].**
//!
//! The setup screens (`crate::screens`) render per-seat rows via
//! [`len`](ModelRegistry::len) / [`label`](ModelRegistry::label); the session's
//! seat spawner builds a controller via [`build`](ModelRegistry::build) per bot
//! seat when a session starts. Difficulty is the shared `beatable()` handicap
//! for every entry: only the *model* (the planner + board evaluator) differs, so
//! picks compare like-for-like: greedy vs beam, linear eval vs ported CC2 eval.

use bevy::prelude::*;

use std::time::Duration;

use crate::ai::{
    AiController, BeamPlanner, Cc2Evaluator, Cc2Weights, DEFAULT_AI_SEED, Evaluator, Handicap,
    LinearEvaluator, Mind, MonotonicClock, PcCoverageConfig, PcCoveragePlanner, PcCoverageUnit,
    SearchBudget,
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
/// Invariant: `entries` is non-empty ([`Default`] always populates it), so the
/// index accessors ([`label`](Self::label) / [`detail`](Self::detail) /
/// [`build`](Self::build)) treat any in-range index as present.
#[derive(Resource)]
pub struct ModelRegistry {
    entries: Vec<ModelEntry>,
}

impl ModelRegistry {
    /// The one-line description of model `index` (the picker's detail pane).
    /// Out-of-range reads as empty rather than panicking — a seat row's focus
    /// cursor is bounded by [`len`](Self::len), but a text pane never needs to.
    pub fn detail(&self, index: usize) -> &str {
        self.entries.get(index).map_or("", |e| e.detail.as_str())
    }

    /// The label of model `index` (out of range reads empty — same contract as
    /// [`detail`](Self::detail)). Seat HUDs and pickers read this.
    pub fn label(&self, index: usize) -> &str {
        self.entries.get(index).map_or("", |e| e.label.as_str())
    }

    /// Build a fresh [`AiController`] for model `index`, if it exists. Each bot
    /// seat addresses the catalog by index (a versus match builds *two*, possibly
    /// the same model twice), so the catalog has no single "selected" cursor.
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
/// time-budgeted cooperative venue), so an entry differs only by the (mind,
/// evaluator, budget) triple it names and the game can never fork the operating
/// conventions from the core's. The host supplies the venue's [`FrameClock`] — the
/// core stays clock-free.
fn search_model(
    mind: Box<dyn Mind>,
    eval: Box<dyn Evaluator>,
    budget: SearchBudget,
) -> AiController {
    AiController::interactive(mind, eval, budget, Box::new(FrameClock::new()))
}

/// The host clock for the time-budgeted venue. The engine-agnostic core defines the
/// [`MonotonicClock`] contract but reads no platform clock itself; Bevy's
/// `platform::time::Instant` is web-time-backed, so this one source works on native
/// and wasm alike.
struct FrameClock {
    start: bevy::platform::time::Instant,
}

impl FrameClock {
    fn new() -> Self {
        Self {
            start: bevy::platform::time::Instant::now(),
        }
    }
}

impl MonotonicClock for FrameClock {
    fn elapsed(&self) -> Duration {
        self.start.elapsed()
    }
}

/// A transposition-pruned beam over the attack-tuned CC2 evaluator at `(width, depth)` — the
/// shared shape of every in-game "attack" bot (the champion family). One home for the
/// construction so a tweak (eval weights, speculation) lands once instead of per entry.
fn tp_attack_model(width: usize, depth: u8) -> AiController {
    search_model(
        Box::new(BeamPlanner::transposing(width)),
        Box::new(Cc2Evaluator::new(Cc2Weights::attack_tuned())),
        SearchBudget::beam(depth),
    )
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

        // The APP champion's brain, watchable: the transposition-pruned attack
        // beam — the mechanism behind our strongest headless bot (TP dedup lets
        // width buy DISTINCT futures). Width 32 / depth 4 keeps each generation a
        // frame-safe chunk in-browser; the headless champion runs w128 / depth 9.
        entries.push(ModelEntry::new(
            "Beam Attack TP",
            "Transposition-pruned beam over the attack-tuned CC2 evaluator — the \
             search mechanism behind our strongest headless APP bot, at a \
             browser-watchable width and depth.",
            || tp_attack_model(32, 4),
        ));

        // The literal session champion: TP-beam width 128 / depth 9 over the
        // attack-tuned CC2 evaluator — 0.8225 APP on the held-out marathon (the
        // research arm `probe-tp128d9`). Deliberately heavy: the beam is
        // batch-grain (one whole generation per frame), so at this width it
        // deliberates a beat per move — here to WATCH the strongest bot think,
        // not for a snappy opponent.
        entries.push(ModelEntry::new(
            "APP Champion",
            "The strongest headless bot, verbatim: transposition-pruned beam \
             (width 128, depth 9) over the attack-tuned CC2 evaluator, 0.8225 \
             APP held-out. Thinks a beat per move — built to watch, not race.",
            || tp_attack_model(128, 9),
        ));

        // The deeper champion from the depth-cap study (docs/research-directions.md):
        // TP-beam width 128 / DEPTH 12. Depth past the d9 grid wall buys a little
        // head-to-head survival (it sees further and plays safer) but trades a touch of
        // raw attack for it — APP 0.81 vs the champion's 0.83. The heaviest model: thinks
        // an even longer beat per move. Here to watch the deepest search reason.
        entries.push(ModelEntry::new(
            "Deep Champion",
            "A beat deeper than the APP champion: transposition-pruned beam at width 128, \
             depth 12. Slightly safer head-to-head, slightly less raw attack — the \
             heaviest model, built to watch think.",
            || tp_attack_model(128, 12),
        ));

        // The efficient narrow-deep config from the scaling study: a small survival-width
        // floor (16) + depth to the knee (12) — roughly 1/5 the champion's search, so it
        // plays SNAPPILY (no champion-style deliberation pause) while still searching deep.
        // Less raw attack than the wide bots (APP ~0.69), but the responsive "go narrow,
        // go deep" pick — a strong opponent you don't wait on.
        entries.push(ModelEntry::new(
            "Efficient Deep",
            "Narrow but deep (width 16, depth 12) — the scaling study's efficient corner. \
             Searches deep at a fraction of the champion's cost, so it plays snappily; \
             less raw attack than the wide bots.",
            || tp_attack_model(16, 12),
        ));

        // The deployed learned evaluator (tetr-valuenet): a single-board value
        // net composed with the CC2 per-move reward. Weights are committed under
        // models/conv_rb1 and added only if the dir loads — a stripped checkout,
        // or the wasm target (no filesystem), simply omits the entry, exactly
        // like any other optional model. `ValueNet` is cheap-ish to clone, so
        // the factory forks a fresh `DeployNet` (its own scratch) per game.
        if let Ok(value_net) = tetr_valuenet::ValueNet::load("models/conv_rb1") {
            entries.push(ModelEntry::new(
                "Learned NNUE",
                "A learned board value net (16×32×32 conv, self-play \
                 replay-buffer trained) composed with the CC2 attack reward — \
                 the strongest stable learned evaluator. Pure-Rust, no GPU.",
                move || {
                    search_model(
                        Box::new(BeamPlanner::new(BEAM_WIDTH)),
                        Box::new(tetr_valuenet::DeployNet::new(value_net.clone())),
                        SearchBudget::beam(BEAM_DEPTH),
                    )
                },
            ));
        }

        Self { entries }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn labels_fit_a_menu_row() {
        // The picker renders labels in a 320 px row, and the pixel font runs ~15 px
        // per glyph at the button size — so 17 characters (~255 px + padding) is the
        // budget. Longer names wrap onto a second line inside the fixed-height row
        // (the original overflow bug); descriptions belong in the detail pane.
        let reg = ModelRegistry::default();
        for i in 0..reg.len() {
            let label = reg.label(i);
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
        for i in 0..reg.len() {
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
        let reg = ModelRegistry::default();
        for i in 0..reg.len() {
            assert!(reg.build(i).is_some(), "entry {i} failed to build");
        }
        assert!(
            reg.build(usize::MAX).is_none(),
            "out of range builds nothing"
        );
    }
}
