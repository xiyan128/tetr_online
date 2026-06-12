//! Bot construction for the research suites — one home for the conventions.
//!
//! [`BotSpec`] is the canonical recipe: **search × eval × sight**, built at
//! full strength (imperfection 0, no reaction delay: these suites measure
//! *policy quality*, not the in-game handicap), seeded policy RNG, blocking
//! venue. Any two bots built here are apples-to-apples by construction, and an
//! experiment's arms are data:
//!
//! ```no_run
//! use tetr_core::ai::eval::Cc2Weights;
//! use tetr_research::bots::BotSpec;
//!
//! let aware = BotSpec::beam(16, 2).cc2(Cc2Weights::attack_tuned());
//! let blind = aware.blind(); // same brain, pending queue hidden
//! let bot = aware.controller(7);
//! ```
//!
//! There is exactly one way to build a research bot: compose a spec. (The
//! pre-spec factory functions are gone — this crate carries no compatibility
//! surface; recorded run records cite settings, which a spec expresses
//! completely. A new evaluator gets a new [`EvalSpec`] arm, not a bypass.)
//!
//! # The bot registry
//!
//! Instances are NAMED ([`bots`]): an experiment binding references bots
//! purely by name, and a name is registered exactly once — so a climbed
//! candidate added here is immediately raceable, panelable, and
//! benchmarkable everywhere with no per-command plumbing. Like experiment
//! names, bot names with recorded runs are immutable: new weights, new name.

use std::time::Duration;

use tetr_core::ai::eval::{Cc2Evaluator, Cc2Weights, Evaluator, LinearEvaluator, Weights};
use tetr_core::ai::{
    AiController, BeamPlanner, BestFirstPlanner, Handicap, Policy, SearchBudget, SearchPolicy,
};
use tetr_core::player::PlayerController;

use crate::versus::BlindToGarbage;

/// The search algorithm and its budget.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SearchSpec {
    /// The shipped greedy baseline (one-piece search over the default linear
    /// evaluator). NOTE: greedy ignores the spec's `eval` — it exists to name
    /// the historical baseline exactly, not to compose.
    Greedy,
    /// Deterministic beam search: fixed `width` per generation, `depth` plies.
    /// Two recorded invariants every beam baseline rests on: `depth == 1`
    /// reproduces the greedy decision exactly (the seam-faithful gate), and
    /// bag speculation past the visible queue is ON (the `BeamPlanner`
    /// default) — every recorded beam number includes it.
    Beam { width: usize, depth: u8 },
    /// Best-first graph search with transposition: `budget` node expansions
    /// per decision, lookahead capped at `depth` plies.
    BestFirst { budget: u32, depth: u8 },
}

/// The board evaluator and its weights.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum EvalSpec {
    /// The linear DT-20 / SURVIVAL evaluator.
    Linear(Weights),
    /// The ported Cold Clear 2 evaluator.
    Cc2(Cc2Weights),
}

impl EvalSpec {
    fn build(self) -> Box<dyn Evaluator> {
        match self {
            EvalSpec::Linear(weights) => Box::new(LinearEvaluator::new(weights)),
            EvalSpec::Cc2(weights) => Box::new(Cc2Evaluator::new(weights)),
        }
    }
}

/// A declarative bot recipe: search × eval × sight. `Copy`, so an experiment's
/// arms can be passed around and varied as plain data.
#[derive(Clone, Copy, Debug)]
pub struct BotSpec {
    pub search: SearchSpec,
    pub eval: EvalSpec,
    /// Hide the pending-garbage queue from the bot ([`BlindToGarbage`]) — the
    /// blind arm of awareness experiments, and currently the stronger versus
    /// configuration (the mispricing record).
    pub blind: bool,
}

impl BotSpec {
    /// The shipped greedy baseline (see [`SearchSpec::Greedy`]).
    pub fn greedy() -> Self {
        Self {
            search: SearchSpec::Greedy,
            eval: EvalSpec::Linear(Weights::default()),
            blind: false,
        }
    }

    /// A beam bot over the default linear evaluator.
    pub fn beam(width: usize, depth: u8) -> Self {
        Self {
            search: SearchSpec::Beam { width, depth },
            eval: EvalSpec::Linear(Weights::default()),
            blind: false,
        }
    }

    /// A best-first bot over the default linear evaluator.
    pub fn best_first(budget: u32, depth: u8) -> Self {
        Self {
            search: SearchSpec::BestFirst { budget, depth },
            eval: EvalSpec::Linear(Weights::default()),
            blind: false,
        }
    }

    /// Swap the evaluator to CC2 with `weights`.
    pub fn cc2(mut self, weights: Cc2Weights) -> Self {
        self.eval = EvalSpec::Cc2(weights);
        self
    }

    /// Swap the evaluator to linear with explicit `weights`.
    pub fn linear(mut self, weights: Weights) -> Self {
        self.eval = EvalSpec::Linear(weights);
        self
    }

    /// Hide the pending-garbage queue from this bot.
    pub fn blind(mut self) -> Self {
        self.blind = true;
        self
    }

    /// Build a fresh controller for this spec (the policy RNG seeded by `seed`).
    pub fn controller(&self, seed: u64) -> Box<dyn PlayerController> {
        let inner: Box<dyn PlayerController> = match self.search {
            SearchSpec::Greedy => {
                // Greedy is the shipped baseline construction and cannot take a
                // custom evaluator; a spec that pairs it with one is a wrong-arm
                // experiment that would silently record lying run headers — fail
                // loudly instead (the review's footgun finding).
                assert!(
                    self.eval == EvalSpec::Linear(Weights::default()),
                    "SearchSpec::Greedy ignores custom evaluators — compose beam()/best_first() instead"
                );
                Box::new(AiController::new(Handicap::perfect(), seed))
            }
            SearchSpec::Beam { width, depth } => full_strength(
                Box::new(BeamPlanner::new(width)),
                self.eval.build(),
                SearchBudget::beam(depth),
                seed,
            ),
            SearchSpec::BestFirst { budget, depth } => full_strength(
                Box::new(BestFirstPlanner::new()),
                self.eval.build(),
                SearchBudget::best_first(budget, depth),
                seed,
            ),
        };
        if self.blind {
            Box::new(BlindToGarbage(inner))
        } else {
            inner
        }
    }

    /// This spec as a harness factory — what `play_versus` / `evaluate_*` take.
    pub fn factory(self) -> impl Fn(u64) -> Box<dyn PlayerController> + Send + Sync + 'static {
        move |seed| self.controller(seed)
    }
}

/// The one place the full-strength convention lives: imperfection 0 and no
/// reaction delay (suites measure pure policy quality), blocking venue.
fn full_strength(
    planner: Box<dyn tetr_core::ai::Mind>,
    eval: Box<dyn Evaluator>,
    budget: SearchBudget,
    seed: u64,
) -> Box<dyn PlayerController> {
    let policy = SearchPolicy::new(planner, eval, budget, 0.0, seed);
    Box::new(AiController::with_policy(
        Box::new(policy) as Box<dyn Policy>,
        Duration::ZERO,
    ))
}

/// Reward = exactly `λ ×` attack sent (the engine's guideline table, chain-exact
/// via `EvalContext`) on top of the attack-tuned board Value: CC2's shaped clear
/// tables are zeroed so the search optimizes the APP objective itself within its
/// horizon. `wasted_t` / `has_back_to_back` stay at CC2's values — they are setup
/// priors encoding beyond-horizon value a single placement's attack can't see.
fn attack_true(lambda: f32) -> Cc2Weights {
    Cc2Weights {
        attack: lambda,
        normal_clears: [0.0; 5],
        mini_spin_clears: [0.0; 3],
        spin_clears: [0.0; 4],
        back_to_back_clear: 0.0,
        combo_attack: 0.0,
        perfect_clear: 0.0,
        perfect_clear_override: false,
        ..Cc2Weights::attack_tuned()
    }
}

/// The climb's v3 accept (see the climb command's RUN RECORD v3) — judged and
/// REJECTED by the race run record; registered so the records stay runnable.
pub const V3_CANDIDATE: [f32; Cc2Weights::BOARD_PARAM_COUNT] = [
    -0.003_662_888_2,
    -1.573_386_2,
    -0.195_788_15,
    -0.349_775_85,
    -1.538_758_6,
    -5.149_458,
    0.357_563_6,
    0.096_651_86,
    1.550_793,
    4.478_138_4,
    3.782_923,
];

/// The bot registry: every named instance, as code. Names are kebab-case and
/// permanent once a run cites them.
pub fn bots() -> Vec<(&'static str, BotSpec)> {
    vec![
        ("greedy", BotSpec::greedy()),
        ("dt20", BotSpec::beam(16, 2)),
        ("cc2-default", BotSpec::beam(16, 2).cc2(Cc2Weights::DEFAULT)),
        (
            "attack-tuned",
            BotSpec::beam(16, 2).cc2(Cc2Weights::attack_tuned()),
        ),
        (
            "cc2-default-blind",
            BotSpec::beam(16, 2).cc2(Cc2Weights::DEFAULT).blind(),
        ),
        (
            "attack-tuned-blind",
            BotSpec::beam(16, 2).cc2(Cc2Weights::attack_tuned()).blind(),
        ),
        (
            "bf-192",
            BotSpec::best_first(192, 6).cc2(Cc2Weights::DEFAULT),
        ),
        (
            "v3-candidate",
            BotSpec::beam(16, 2).cc2(Cc2Weights::attack_tuned().with_board_params(&V3_CANDIDATE)),
        ),
        (
            // Depth-3 candidate: same attack-tuned eval, one ply deeper —
            // the "deeper search" lever from the v3 epilogue.
            "attack-tuned-d3",
            BotSpec::beam(16, 3).cc2(Cc2Weights::attack_tuned()),
        ),
        // The APP depth/width ladder (post-bitboard re-map: the old d6w32 ≈ 0.67
        // plateau was measured pre-perf-strike, and its d8 point was width-16).
        (
            "attack-tuned-d4",
            BotSpec::beam(16, 4).cc2(Cc2Weights::attack_tuned()),
        ),
        (
            "attack-tuned-d6",
            BotSpec::beam(16, 6).cc2(Cc2Weights::attack_tuned()),
        ),
        (
            "attack-tuned-d6w32",
            BotSpec::beam(32, 6).cc2(Cc2Weights::attack_tuned()),
        ),
        (
            "attack-tuned-d8w32",
            BotSpec::beam(32, 8).cc2(Cc2Weights::attack_tuned()),
        ),
        // Engine-true attack reward (λ = 1), shaped tables zeroed — the
        // objective-in-the-search hypothesis, at matched configs for clean A/Bs.
        // RESULT 2026-06-12: LOSES at both configs (0.434 vs 0.572 @ d3, 0.618
        // vs 0.721 @ d6w32) — CC2's shaping carries beyond-horizon value.
        ("attack-true-d3", BotSpec::beam(16, 3).cc2(attack_true(1.0))),
        (
            "attack-true-d6w32",
            BotSpec::beam(32, 6).cc2(attack_true(1.0)),
        ),
        // --- probe-* : single-lever APP hypotheses at the d6w32 incumbent ----
        // The exploratory tier (screened on TRAIN marathon; most will lose —
        // they are the lab notebook, immutable like every cited name).
        // Mixture: engine-true attack ADDED to the intact shaped tables, so
        // chain continuation (combo staircase, B2B +1) is valued at true scale.
        (
            "probe-mix05-d6w32",
            BotSpec::beam(32, 6).cc2(Cc2Weights {
                attack: 0.5,
                ..Cc2Weights::attack_tuned()
            }),
        ),
        (
            "probe-mix1-d6w32",
            BotSpec::beam(32, 6).cc2(Cc2Weights {
                attack: 1.0,
                ..Cc2Weights::attack_tuned()
            }),
        ),
        // Combo emphasis beyond CC2's 1.5 (the engine resumes real combos now).
        (
            "probe-combo4-d6w32",
            BotSpec::beam(32, 6).cc2(Cc2Weights {
                combo_attack: 4.0,
                ..Cc2Weights::attack_tuned()
            }),
        ),
        // Deeper tetris well (the recorded offense↔survival trade-off knob).
        (
            "probe-well1-d6w32",
            BotSpec::beam(32, 6).cc2(Cc2Weights {
                tetris_well_depth: 1.0,
                ..Cc2Weights::attack_tuned()
            }),
        ),
        // Perfect-clear hunter (PC = +10 attack, the table's biggest prize).
        (
            "probe-pc40-d6w32",
            BotSpec::beam(32, 6).cc2(Cc2Weights {
                perfect_clear: 40.0,
                ..Cc2Weights::attack_tuned()
            }),
        ),
        // Spin emphasis: clear rewards and slot Value both doubled.
        (
            "probe-spin2x-d6w32",
            BotSpec::beam(32, 6).cc2(Cc2Weights {
                spin_clears: [0.0, 2.0, 8.0, 12.0],
                tslot: [0.2, 3.0, 8.93, 8.0],
                ..Cc2Weights::attack_tuned()
            }),
        ),
        // Best-first at research budgets (never re-tested post-bitboard; the
        // recorded result: beats the beam per node and scales with budget).
        (
            "probe-bf1k-d8",
            BotSpec::best_first(1000, 8).cc2(Cc2Weights::attack_tuned()),
        ),
        (
            "probe-bf2k-d8",
            BotSpec::best_first(2000, 8).cc2(Cc2Weights::attack_tuned()),
        ),
        // Toy-sized twins for the smoke gate (seconds, not minutes).
        (
            "attack-tuned-tiny",
            BotSpec::beam(4, 1).cc2(Cc2Weights::attack_tuned()),
        ),
        ("dt20-tiny", BotSpec::beam(4, 1)),
    ]
}

/// A registered bot: its name travels with the spec so reports can speak
/// names instead of weight dumps (the registry and receipt hold the rest).
#[derive(Clone, Copy, Debug)]
pub struct Bot {
    pub name: &'static str,
    pub spec: BotSpec,
}

/// Look a registered bot up by name.
pub fn find(name: &str) -> Option<Bot> {
    bots()
        .into_iter()
        .find(|(n, _)| *n == name)
        .map(|(name, spec)| Bot { name, spec })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Same spec, same seed => the identical game (the determinism every suite
    /// rests on, witnessed at the construction seam).
    #[test]
    fn same_spec_same_game() {
        let spec = BotSpec::beam(8, 2).cc2(Cc2Weights::attack_tuned());
        let outcome = |make: &dyn Fn(u64) -> Box<dyn PlayerController>| {
            let o = crate::versus::play_versus(make, make, 7, 30);
            (o.plies, o.attack_a, o.attack_b, o.a_topped, o.b_topped)
        };
        assert_eq!(outcome(&spec.factory()), outcome(&spec.factory()));
    }

    /// `.blind()` wraps the same brain: with nothing queued the play is
    /// identical to the sighted spec (the wrapper only strips pending).
    #[test]
    fn blind_spec_plays_identically_with_empty_queue() {
        let spec = BotSpec::beam(8, 2);
        let o1 = crate::marathon::play_marathon_capped(&spec.factory(), 3, 50_000, 30);
        let o2 = crate::marathon::play_marathon_capped(&spec.blind().factory(), 3, 50_000, 30);
        assert_eq!(
            (o1.score, o1.pieces, o1.lines),
            (o2.score, o2.pieces, o2.lines)
        );
    }
}
