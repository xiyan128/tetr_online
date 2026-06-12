//! Behavior + APP suite report for a bot across the standard garbage
//! scenarios. APP (attack per piece) is the primary strike metric; also
//! reports DS/P, survival, attack/line (concentration vs combo-spam), and the
//! clear-type behavior histogram.

use serde_json::json;

use tetr_core::ai::Cc2Weights;
use tetr_core::ai::eval::{BoardWeights, RewardWeights, Weights};

use crate::behavior::{ScenarioReport, evaluate_scenario, standard_suite};
use crate::bots::BotSpec;
use crate::commands::{Beam, BoardParams, Runtime};
use crate::ledger::RunLedger;
use crate::seeds::seed_set;

#[derive(Clone, Copy, Debug, serde::Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum Bot {
    /// Beam over the shipped linear DT-20 / SURVIVAL weights.
    Dt20,
    /// Beam over the ported CC2 evaluator at its defaults.
    Cc2,
    /// Beam over CC2 with the spec's `cc2_params`.
    Cc2custom,
    /// Beam over linear weights from `board_params` / `reward_params`.
    Lincustom,
    /// Best-first over CC2 defaults (beam twin at `node_budget`).
    Bf,
    /// Best-first over CC2 with `cc2_params`.
    Bfcustom,
    /// Best-first over custom linear weights (can deep search build the
    /// cascade the beam's truncation prunes?).
    Bflin,
}

#[derive(Clone, Copy, Debug, serde::Serialize)]
pub struct Spec {
    /// Seed count per scenario.
    pub seeds: usize,
    pub beam: Beam,
    /// Best-first total node expansions (the `bf*` arms).
    pub node_budget: u32,
    /// The arm under test.
    pub bot: Bot,
    /// Custom linear board weights (`lincustom`/`bflin`; None = shipped).
    pub board_params: Option<[f32; 10]>,
    /// Custom linear reward weights (`lincustom`/`bflin`; None = shipped).
    pub reward_params: Option<[f32; 11]>,
    /// Custom CC2 board params (`cc2custom`/`bfcustom`; None = defaults).
    pub cc2_params: Option<BoardParams>,
}

impl Spec {
    pub fn bot(bot: Bot) -> Self {
        Self {
            seeds: 24,
            beam: Beam::default(),
            node_budget: 4000,
            bot,
            board_params: None,
            reward_params: None,
            cc2_params: None,
        }
    }
}

/// Custom linear [`Weights`] for validating an `app-climb` result at any
/// depth/width; each group falls back to its shipped default when absent.
fn linear_custom_weights(spec: &Spec) -> Weights {
    let bp = spec.board_params.unwrap_or(BoardWeights::DT20.params());
    let rp = spec
        .reward_params
        .unwrap_or(RewardWeights::SURVIVAL.params());
    Weights {
        board: BoardWeights::from_params(&bp),
        reward: RewardWeights::from_params(&rp),
    }
}

/// CC2 weights with the spec's board params, for validating a
/// `cc2-app-climb` result on the full behavior suite.
fn cc2_custom_weights(spec: &Spec) -> Cc2Weights {
    let params = spec
        .cc2_params
        .unwrap_or(Cc2Weights::DEFAULT.board_params());
    Cc2Weights::DEFAULT.with_board_params(&params)
}

fn print_report(r: &ScenarioReport) {
    eprintln!(
        "\n[{}] survival {:.0}% | APP {:.3} | DS/P {:.2} | atk/line {:.2} | pieces {:.0} | garbage_recv {:.1} | {:.1} ms/piece",
        r.scenario.label(),
        r.survival_rate * 100.0,
        r.mean_app,
        r.mean_dsp,
        r.mean_attack_per_line,
        r.mean_pieces,
        r.mean_garbage_received,
        r.mean_ms_per_piece,
    );
    let t = &r.totals;
    eprintln!(
        "    clears: S{} D{} T{} Quad{} | TSmini{} TSS{} TSD{} TST{} | B2B{} comboClears{} maxCombo{} PC{}",
        t.singles,
        t.doubles,
        t.triples,
        t.tetrises,
        t.tspin_mini,
        t.tspin_single,
        t.tspin_double,
        t.tspin_triple,
        t.b2b_clears,
        t.combo_clears,
        t.max_combo,
        t.perfect_clears,
    );
    println!("APP[{}] {:.3}", r.scenario.label(), r.mean_app);
}

pub fn run(spec: &Spec, _rt: &Runtime, ledger: &mut RunLedger) -> std::io::Result<()> {
    let seeds = seed_set(spec.seeds);
    let Beam { width, depth } = spec.beam;
    let node_budget = spec.node_budget;

    eprintln!(
        "Behavior + APP suite | bot={:?} beam(depth={depth}, width={width}) | {} seeds",
        spec.bot,
        seeds.len()
    );

    // The best-first arms are the search-algorithm counterparts of their beam
    // twins — same eval, beam vs best-first.
    let bot = match spec.bot {
        Bot::Cc2 => BotSpec::beam(width, depth).cc2(Cc2Weights::DEFAULT),
        Bot::Cc2custom => BotSpec::beam(width, depth).cc2(cc2_custom_weights(spec)),
        Bot::Lincustom => BotSpec::beam(width, depth).linear(linear_custom_weights(spec)),
        Bot::Bf => BotSpec::best_first(node_budget, depth).cc2(Cc2Weights::DEFAULT),
        Bot::Bfcustom => BotSpec::best_first(node_budget, depth).cc2(cc2_custom_weights(spec)),
        Bot::Bflin => BotSpec::best_first(node_budget, depth).linear(linear_custom_weights(spec)),
        Bot::Dt20 => BotSpec::beam(width, depth),
    };

    let mut summaries = Vec::new();
    for scenario in standard_suite() {
        let report = evaluate_scenario(&bot.factory(), &seeds, scenario);
        for (&seed, outcome) in seeds.iter().zip(&report.outcomes) {
            ledger.append_outcome(&json!({
                "seed": seed,
                "scenario": scenario,
                "outcome": outcome,
            }))?;
        }
        summaries.push(json!({
            "scenario": scenario,
            "games": report.games,
            "survival_rate": report.survival_rate,
            "mean_app": report.mean_app,
            "mean_dsp": report.mean_dsp,
            "mean_attack_per_line": report.mean_attack_per_line,
            "mean_pieces": report.mean_pieces,
            "mean_garbage_received": report.mean_garbage_received,
            "mean_ms_per_piece": report.mean_ms_per_piece,
            "totals": report.totals,
        }));
        print_report(&report);
    }
    ledger.write_summary(json!({
        "exit_reason": "complete",
        "scenarios": summaries,
    }))?;
    Ok(())
}
