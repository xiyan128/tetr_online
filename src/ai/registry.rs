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
    AiController, BeamPlanner, BestFirstPlanner, Cc2Evaluator, Cc2Weights, Evaluator, Handicap,
    LinearEvaluator, Planner, SearchBudget, SearchPolicy, DEFAULT_AI_SEED,
};

/// Beam settings for the in-game Tier-2 bots. Depth 2 is smooth per piece (a few ms
/// in release) and already ~+26% over greedy on the marathon benchmark; width 16
/// matches the bench. The headless `bench-marathon` explores deeper (depth 3 ≈ +33%).
const BEAM_WIDTH: usize = 16;
const BEAM_DEPTH: u8 = 2;

/// Best-first node budget for the in-game attack model — the same operating point the
/// research harness benchmarks (~25 ms/piece in release after the bitboard strike, well
/// inside a watchable cadence). The headless bench runs it far higher, where quality
/// scales with budget at proportional latency.
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

    /// The selected model's label (for the log line / HUD).
    pub fn selected_label(&self) -> &str {
        &self.entries[self.selected].label
    }

    /// Build a fresh [`AiController`] for the selected model.
    pub fn selected_controller(&self) -> AiController {
        (self.entries[self.selected].build)()
    }
}

/// Wire a planner + evaluator into a fresh controller under the shared default
/// handicap — the one construction every Tier-2 entry shares, so an entry differs
/// only by the (planner, evaluator, budget) triple it names.
fn search_model(
    planner: Box<dyn Planner>,
    eval: Box<dyn Evaluator>,
    budget: SearchBudget,
) -> AiController {
    let h = Handicap::default();
    let policy = SearchPolicy::new(planner, eval, budget, h.imperfection, DEFAULT_AI_SEED);
    AiController::with_policy(Box::new(policy), h.reaction)
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
            search_model(
                Box::new(BeamPlanner::new(BEAM_WIDTH)),
                Box::new(LinearEvaluator::default()),
                SearchBudget::beam(BEAM_DEPTH),
            )
        }));

        // Cold Clear 2's evaluator, ported (`Cc2Evaluator`) on the same beam — watch
        // the bot we benchmarked against play here. On the fair native arena our
        // DT-20 beats it at both downstacking and versus (it is depth-hungry, tuned
        // for CC2's deep MCTS); this shows its tetris-well / T-spin-seeking style.
        entries.push(ModelEntry::new("Beam - CC2 eval (ported)", || {
            search_model(
                Box::new(BeamPlanner::new(BEAM_WIDTH)),
                Box::new(Cc2Evaluator::default()),
                SearchBudget::beam(BEAM_DEPTH),
            )
        }));

        // The APP-sprint attack bot: Cold Clear 2's evaluator with the APP-climbed
        // board weights (`Cc2Weights::attack_tuned`), on a deeper beam, with the
        // engine's combo tracking active — concentrated B2B-Tetris / T-spin attack
        // plus combo-aware digging. Depth 3 keeps the per-piece search watchable
        // in-browser; the headless bench runs it deeper.
        entries.push(ModelEntry::new("Beam - CC2 Attack (tuned)", || {
            search_model(
                Box::new(BeamPlanner::new(BEAM_WIDTH)),
                Box::new(Cc2Evaluator::new(Cc2Weights::attack_tuned())),
                SearchBudget::beam(3), // deeper than the default 2 — attack tuning + combos need lookahead
            )
        }));

        // The **best-first** search over the same tuned CC2 attack eval — a graph
        // search with a per-root transposition table that pursues deep attack lines
        // the beam's fixed-width truncation prunes. In benches it beats the beam per
        // node and scales smoothly with the node budget.
        entries.push(ModelEntry::new("Search - CC2 Attack (best-first)", || {
            search_model(
                Box::new(BestFirstPlanner::new()),
                Box::new(Cc2Evaluator::new(Cc2Weights::attack_tuned())),
                // Depth-capped by the visible queue, not width; the node budget is
                // the quality/latency dial.
                SearchBudget::best_first(ATTACK_BF_BUDGET, 6),
            )
        }));

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
        }
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
