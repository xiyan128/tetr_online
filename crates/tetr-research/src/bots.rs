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
