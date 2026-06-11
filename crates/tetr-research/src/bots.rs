//! Bot construction for the research suites.
//!
//! Every contender funnels through the same conventions — full strength
//! (imperfection 0, no reaction delay: these suites measure *policy quality*,
//! not the in-game handicap), seeded policy RNG, blocking venue — so any two
//! bots built here are apples-to-apples by construction.

use std::time::Duration;

use tetr_core::ai::eval::{Cc2Evaluator, Cc2Weights, Evaluator, LinearEvaluator, Weights};
use tetr_core::ai::{
    AiController, BeamPlanner, BestFirstPlanner, Handicap, Policy, SearchBudget, SearchPolicy,
};
use tetr_core::player::PlayerController;

/// The current shipped bot: greedy search over the linear DT-20 / SURVIVAL
/// evaluator, at full strength (`Handicap::perfect()`). This is the baseline.
pub fn baseline_bot(seed: u64) -> Box<dyn PlayerController> {
    Box::new(AiController::new(Handicap::perfect(), seed))
}

/// Core beam-bot constructor: a deterministic [`BeamPlanner`] over `eval` at full
/// strength (imperfection 0, no reaction delay — measures pure policy quality). Every
/// beam contender funnels through here, so the planner / budget / strength convention
/// lives in one place and head-to-heads stay apples-to-apples. Adding a new contender
/// is one line: `beam_bot(seed, w, d, Box::new(MyEvaluator::new(..)))`.
pub fn beam_bot(
    seed: u64,
    beam_width: usize,
    max_depth: u8,
    eval: Box<dyn Evaluator>,
) -> Box<dyn PlayerController> {
    let policy = SearchPolicy::new(
        Box::new(BeamPlanner::new(beam_width)),
        eval,
        SearchBudget::beam(max_depth),
        0.0, // no imperfection — measure policy quality
        seed,
    );
    Box::new(AiController::with_policy(
        Box::new(policy) as Box<dyn Policy>,
        Duration::ZERO,
    ))
}

/// Core **best-first-search** bot: a [`BestFirstPlanner`] over `eval` at full strength
/// (imperfection 0, no reaction delay). The best-first analogue of [`beam_bot`] with
/// the SAME eval/strength convention, so a head-to-head isolates the **search
/// algorithm** — the beam's fixed-width generations vs best-first's node-budgeted
/// graph search with transposition. `node_budget` is total expansions per decision;
/// `max_depth` caps lookahead plies.
pub fn bestfirst_bot(
    seed: u64,
    node_budget: u32,
    max_depth: u8,
    eval: Box<dyn Evaluator>,
) -> Box<dyn PlayerController> {
    let policy = SearchPolicy::new(
        Box::new(BestFirstPlanner::new()),
        eval,
        SearchBudget::best_first(node_budget, max_depth),
        0.0, // no imperfection — measure policy quality
        seed,
    );
    Box::new(AiController::with_policy(
        Box::new(policy) as Box<dyn Policy>,
        Duration::ZERO,
    ))
}

/// A best-first bot over CC2's evaluator with custom [`Cc2Weights`] — the search-
/// algorithm counterpart of [`beam_cc2_weights_bot`], for an apples-to-apples
/// best-first-vs-beam comparison at a fixed eval.
pub fn bestfirst_cc2_weights_bot(
    seed: u64,
    node_budget: u32,
    max_depth: u8,
    weights: Cc2Weights,
) -> Box<dyn PlayerController> {
    bestfirst_bot(
        seed,
        node_budget,
        max_depth,
        Box::new(Cc2Evaluator::new(weights)),
    )
}

/// A best-first bot over the linear evaluator with explicit [`Weights`] — the
/// counterpart of [`beam_weights_bot`]. Pairs best-first's deep-line search with the
/// `near_full_rows` combo feature, to test whether it can find the clean-board combo
/// cascade the beam's fixed-width truncation prunes.
pub fn bestfirst_weights_bot(
    seed: u64,
    node_budget: u32,
    max_depth: u8,
    weights: Weights,
) -> Box<dyn PlayerController> {
    bestfirst_bot(
        seed,
        node_budget,
        max_depth,
        Box::new(LinearEvaluator::new(weights)),
    )
}

/// The Tier-2 beam bot: a deterministic `BeamPlanner` over the **same** linear
/// DT-20 / SURVIVAL evaluator the baseline uses, at full strength (imperfection 0,
/// no reaction delay). It differs from [`baseline_bot`] in *only* the planner
/// (greedy → beam), so a head-to-head isolates the search depth's effect on
/// score/sec. `beam_width` controls truncation; `max_depth` the lookahead plies
/// (`max_depth == 1` reproduces the greedy decision exactly — the seam-faithful
/// gate). Bag speculation past the visible queue is on (the `BeamPlanner` default).
pub fn beam_linear_bot(seed: u64, beam_width: usize, max_depth: u8) -> Box<dyn PlayerController> {
    beam_bot(
        seed,
        beam_width,
        max_depth,
        Box::new(LinearEvaluator::default()),
    )
}

/// **Cold Clear 2's evaluator, ported** ([`Cc2Evaluator`]) on our beam — CC2's
/// *evaluation function* playing on our engine and search. Identical planner,
/// budget, and strength to [`beam_linear_bot`]; only the evaluator differs, so a
/// head-to-head isolates eval quality. Crucially this plays the **fair** versus
/// harness on our engine with real garbage — the comparison the TBP bridge could
/// not give (CC2 has no garbage message). This is the baseline to hillclimb past.
pub fn beam_cc2_bot(seed: u64, beam_width: usize, max_depth: u8) -> Box<dyn PlayerController> {
    beam_bot(
        seed,
        beam_width,
        max_depth,
        Box::new(Cc2Evaluator::default()),
    )
}

/// A beam bot over an explicit linear [`Weights`] set — lets a head-to-head vary
/// the board features and/or the reward profile on the same planner/strength (e.g.
/// DT-20 board + Cold-Clear *concentrated-attack* reward vs the shipped SURVIVAL
/// reward that cashes every clear).
pub fn beam_weights_bot(
    seed: u64,
    beam_width: usize,
    max_depth: u8,
    weights: Weights,
) -> Box<dyn PlayerController> {
    beam_bot(
        seed,
        beam_width,
        max_depth,
        Box::new(LinearEvaluator::new(weights)),
    )
}

/// Like [`beam_cc2_bot`] but with **custom** CC2 weights — the hillclimb's
/// candidate factory. Only the evaluator's weights differ.
pub fn beam_cc2_weights_bot(
    seed: u64,
    beam_width: usize,
    max_depth: u8,
    weights: Cc2Weights,
) -> Box<dyn PlayerController> {
    beam_bot(
        seed,
        beam_width,
        max_depth,
        Box::new(Cc2Evaluator::new(weights)),
    )
}
