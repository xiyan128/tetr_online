//! The AI model registry: the catalog of "brains" the Watch-AI sandbox can run.
//!
//! Each [`ModelEntry`] names a bot and knows how to build a fresh [`AiController`]
//! for it. The shipped catalog spans the linear DT-20 evaluator and the ported
//! Cold Clear 2 attack evaluator, on greedy / beam / best-first search.
//! **Adding a model is one entry in [`ModelRegistry::default`].**
//!
//! The picker screen ([`crate::screens`]) renders [`labels`](ModelRegistry::labels)
//! and writes the selection; the sandbox ([`crate::ai::sandbox`]) builds
//! [`selected_controller`](ModelRegistry::selected_controller) when a Watch-AI
//! session starts. Difficulty is the shared `beatable()` handicap for every entry —
//! only the *model* (the planner + board evaluator) differs, so picks compare
//! like-for-like: greedy vs beam, linear eval vs ported CC2 eval.

use bevy::prelude::*;

use crate::ai::{
    AiController, BeamPlanner, BestFirstPlanner, Cc2Evaluator, Cc2Weights, Handicap,
    LinearEvaluator, Policy, SearchBudget, SearchPolicy, DEFAULT_AI_SEED,
};

/// Beam settings for the in-game Tier-2 bots. Depth 2 is smooth per piece (a few ms
/// in release) and already ~+26% over greedy on the marathon benchmark; width 16
/// matches the bench. The headless `bench-marathon` explores deeper (depth 3 ≈ +33%).
const BEAM_WIDTH: usize = 16;
const BEAM_DEPTH: u8 = 2;

/// Cold Clear 2 board weights warm-climbed for attack-per-piece (`cc2-app-climb`,
/// APP fitness) — the attack profile shared by the beam and best-first attack models.
const ATTACK_BOARD_PARAMS: [f32; Cc2Weights::BOARD_PARAM_COUNT] = [
    -0.003_447_473,
    -1.5,
    -0.2,
    -0.362_030_36,
    -1.5,
    -5.0,
    0.347_263_3,
    0.1,
    1.5,
    4.465_080_7,
    4.0,
];

/// Best-first node budget for the in-game attack model. At ~120-160 ms/piece (measured)
/// it blocks the main thread noticeably, but the AI's reaction delay masks most of it;
/// the headless bench runs it far higher, where quality scales with budget (tuned-attack
/// clean APP 0.655→0.680, faucet1/2 0.325→0.455 from budget 150→400, but at 2-3x the ms).
const ATTACK_BF_BUDGET: u32 = 150;

/// One selectable AI model: a display name + a factory for a fresh controller.
///
/// The factory is `Send + Sync` (it only *builds* a controller); the produced
/// [`AiController`] is `Send`-but-not-`Sync` and lives in the non-send `AiPlayer`.
struct ModelEntry {
    label: String,
    build: Box<dyn Fn() -> AiController + Send + Sync>,
}

impl ModelEntry {
    fn new(
        label: impl Into<String>,
        build: impl Fn() -> AiController + Send + Sync + 'static,
    ) -> Self {
        Self {
            label: label.into(),
            build: Box::new(build),
        }
    }
}

/// The catalog of AI models plus the current selection. Inserted by the
/// [`GamePlugin`](crate::GamePlugin); read by the model-select screen and the
/// sandbox.
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

    /// The selected model's label (for the log line / HUD).
    pub fn selected_label(&self) -> &str {
        self.entries
            .get(self.selected)
            .map_or("?", |e| e.label.as_str())
    }

    /// Build a fresh [`AiController`] for the selected model.
    pub fn selected_controller(&self) -> AiController {
        match self.entries.get(self.selected) {
            Some(entry) => (entry.build)(),
            None => AiController::beatable(),
        }
    }
}

impl Default for ModelRegistry {
    fn default() -> Self {
        let mut entries = Vec::new();

        // Always available: the shipped linear DT-20 / SURVIVAL evaluator (greedy).
        entries.push(ModelEntry::new("Linear - DT-20 (greedy)", || {
            AiController::beatable()
        }));

        // The Tier-2 beam: deterministic multi-ply search over the SAME linear eval.
        // ~+26-33% over greedy on the marathon bench — the architecture leapfrog. It
        // reads `LinearEvaluator::default()`, so any `weights.rs` tuning flows in free.
        entries.push(ModelEntry::new("Beam - DT-20 (Tier-2)", || {
            let h = Handicap::default();
            let policy = SearchPolicy::new(
                Box::new(BeamPlanner::new(BEAM_WIDTH)),
                Box::new(LinearEvaluator::default()),
                SearchBudget::beam(BEAM_DEPTH),
                h.imperfection,
                DEFAULT_AI_SEED,
            );
            AiController::with_policy(Box::new(policy) as Box<dyn Policy>, h.reaction)
        }));

        // Cold Clear 2's evaluator, ported (`Cc2Evaluator`) on the same beam — watch
        // the bot we benchmarked against play here. On the fair native arena our
        // DT-20 beats it at both downstacking and versus (it is depth-hungry, tuned
        // for CC2's deep MCTS); this shows its tetris-well / T-spin-seeking style.
        entries.push(ModelEntry::new("Beam - CC2 eval (ported)", || {
            let h = Handicap::default();
            let policy = SearchPolicy::new(
                Box::new(BeamPlanner::new(BEAM_WIDTH)),
                Box::new(Cc2Evaluator::default()),
                SearchBudget::beam(BEAM_DEPTH),
                h.imperfection,
                DEFAULT_AI_SEED,
            );
            AiController::with_policy(Box::new(policy) as Box<dyn Policy>, h.reaction)
        }));

        // The APP-sprint attack bot: Cold Clear 2's evaluator with board weights
        // **warm-climbed for attack-per-piece** (`cc2-app-climb`), on a deeper beam,
        // with the engine's combo tracking active. In benchmarks it plays concentrated
        // B2B-Tetris / T-spin attack plus combo-aware digging (clean APP ~0.67, cheese
        // ~0.68 at depth 6 — far above the shipped linear bot's ~0.2). Depth 3 keeps
        // the per-piece search watchable in-browser; the headless bench runs it deeper.
        entries.push(ModelEntry::new("Beam - CC2 Attack (tuned)", || {
            let weights = Cc2Weights::DEFAULT.with_board_params(&ATTACK_BOARD_PARAMS);
            let h = Handicap::default();
            let policy = SearchPolicy::new(
                Box::new(BeamPlanner::new(BEAM_WIDTH)),
                Box::new(Cc2Evaluator::new(weights)),
                SearchBudget::beam(3), // deeper than the default 2 — attack tuning + combos need lookahead
                h.imperfection,
                DEFAULT_AI_SEED,
            );
            AiController::with_policy(Box::new(policy) as Box<dyn Policy>, h.reaction)
        }));

        // The **best-first** search over the same tuned CC2 attack eval — a graph
        // search with a per-root transposition table that pursues deep attack lines the
        // beam's fixed-width truncation prunes. In benches it beats the beam per node and
        // scales smoothly with `NODE_BUDGET`; here the budget is kept modest so the
        // (blocking) per-piece search stays watchable in-browser.
        entries.push(ModelEntry::new("Search - CC2 Attack (best-first)", || {
            let weights = Cc2Weights::DEFAULT.with_board_params(&ATTACK_BOARD_PARAMS);
            let h = Handicap::default();
            let policy = SearchPolicy::new(
                Box::new(BestFirstPlanner::new(ATTACK_BF_BUDGET)),
                Box::new(Cc2Evaluator::new(weights)),
                SearchBudget::beam(6), // best-first is depth-capped by the visible queue, not width
                h.imperfection,
                DEFAULT_AI_SEED,
            );
            AiController::with_policy(Box::new(policy) as Box<dyn Policy>, h.reaction)
        }));

        Self {
            entries,
            selected: 0,
        }
    }
}
