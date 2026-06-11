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
//! The free functions below predate the spec and remain as one-line shims —
//! they are the names the recorded baselines were measured under. New
//! experiments should compose a [`BotSpec`].

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
            SearchSpec::Beam { width, depth } => {
                full_strength(Box::new(BeamPlanner::new(width)), self.eval.build(), {
                    SearchBudget::beam(depth)
                })(seed)
            }
            SearchSpec::BestFirst { budget, depth } => full_strength(
                Box::new(BestFirstPlanner::new()),
                self.eval.build(),
                SearchBudget::best_first(budget, depth),
            )(seed),
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
/// Returns a one-shot builder so [`BotSpec::controller`] and the legacy shims
/// construct identically.
fn full_strength(
    planner: Box<dyn tetr_core::ai::Mind>,
    eval: Box<dyn Evaluator>,
    budget: SearchBudget,
) -> impl FnOnce(u64) -> Box<dyn PlayerController> {
    move |seed| {
        let policy = SearchPolicy::new(planner, eval, budget, 0.0, seed);
        Box::new(AiController::with_policy(
            Box::new(policy) as Box<dyn Policy>,
            Duration::ZERO,
        ))
    }
}

// --- Legacy factory shims ------------------------------------------------------
// The names the recorded baselines were measured under. Each is a one-line
// composition of `BotSpec` (or the raw `full_strength` core for the
// `Box<dyn Evaluator>` forms); new experiments should use the spec directly.

/// The current shipped bot: greedy search over the linear DT-20 / SURVIVAL
/// evaluator, at full strength. This is the baseline.
pub fn baseline_bot(seed: u64) -> Box<dyn PlayerController> {
    BotSpec::greedy().controller(seed)
}

/// A beam bot over an arbitrary evaluator (the `Box<dyn Evaluator>` escape
/// hatch for one-off evals that have no [`EvalSpec`] arm yet).
pub fn beam_bot(
    seed: u64,
    beam_width: usize,
    max_depth: u8,
    eval: Box<dyn Evaluator>,
) -> Box<dyn PlayerController> {
    full_strength(
        Box::new(BeamPlanner::new(beam_width)),
        eval,
        SearchBudget::beam(max_depth),
    )(seed)
}

/// A best-first bot over an arbitrary evaluator (escape hatch, like [`beam_bot`]).
pub fn bestfirst_bot(
    seed: u64,
    node_budget: u32,
    max_depth: u8,
    eval: Box<dyn Evaluator>,
) -> Box<dyn PlayerController> {
    full_strength(
        Box::new(BestFirstPlanner::new()),
        eval,
        SearchBudget::best_first(node_budget, max_depth),
    )(seed)
}

/// Best-first over CC2's evaluator with custom weights.
pub fn bestfirst_cc2_weights_bot(
    seed: u64,
    node_budget: u32,
    max_depth: u8,
    weights: Cc2Weights,
) -> Box<dyn PlayerController> {
    BotSpec::best_first(node_budget, max_depth)
        .cc2(weights)
        .controller(seed)
}

/// Best-first over the linear evaluator with explicit weights.
pub fn bestfirst_weights_bot(
    seed: u64,
    node_budget: u32,
    max_depth: u8,
    weights: Weights,
) -> Box<dyn PlayerController> {
    BotSpec::best_first(node_budget, max_depth)
        .linear(weights)
        .controller(seed)
}

/// The Tier-2 beam bot over the default linear evaluator (differs from
/// [`baseline_bot`] in only the planner, so a head-to-head isolates search).
pub fn beam_linear_bot(seed: u64, beam_width: usize, max_depth: u8) -> Box<dyn PlayerController> {
    BotSpec::beam(beam_width, max_depth).controller(seed)
}

/// Cold Clear 2's evaluator (default weights) on our beam — the eval
/// head-to-head against [`beam_linear_bot`], and the baseline to hillclimb past.
pub fn beam_cc2_bot(seed: u64, beam_width: usize, max_depth: u8) -> Box<dyn PlayerController> {
    BotSpec::beam(beam_width, max_depth)
        .cc2(Cc2Weights::DEFAULT)
        .controller(seed)
}

/// A beam bot over explicit linear weights.
pub fn beam_weights_bot(
    seed: u64,
    beam_width: usize,
    max_depth: u8,
    weights: Weights,
) -> Box<dyn PlayerController> {
    BotSpec::beam(beam_width, max_depth)
        .linear(weights)
        .controller(seed)
}

/// A beam bot over CC2's evaluator with custom weights — the hillclimb's
/// candidate factory.
pub fn beam_cc2_weights_bot(
    seed: u64,
    beam_width: usize,
    max_depth: u8,
    weights: Cc2Weights,
) -> Box<dyn PlayerController> {
    BotSpec::beam(beam_width, max_depth)
        .cc2(weights)
        .controller(seed)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The spec and the legacy shims must construct identical bots: same spec,
    /// same seed ⇒ identical play. (Construction equality is unobservable
    /// directly; a short deterministic game is the witness.)
    #[test]
    fn spec_and_shim_build_the_same_bot() {
        let weights = Cc2Weights::attack_tuned();
        let outcome = |make: &dyn Fn(u64) -> Box<dyn PlayerController>| {
            let o = crate::versus::play_versus(make, make, 7, 30);
            (o.plies, o.attack_a, o.attack_b, o.a_topped, o.b_topped)
        };
        let spec = BotSpec::beam(8, 2).cc2(weights);
        let via_spec = outcome(&spec.factory());
        let via_shim = outcome(&|s| beam_cc2_weights_bot(s, 8, 2, weights));
        assert_eq!(via_spec, via_shim);
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
