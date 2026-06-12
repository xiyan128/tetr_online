//! Behavior + APP suite report for a bot across the standard garbage scenarios.
//! APP (attack per piece) is the primary strike metric; also reports DS/P, survival,
//! attack/line (concentration vs combo-spam), and the clear-type behavior histogram.
//!
//! Run: `SEEDS=24 BEAM_DEPTH=2 cargo run --release -p tetr-research --bin behavior`

use tetr_core::ai::Cc2Weights;
use tetr_core::ai::eval::{BoardWeights, RewardWeights, Weights};
use tetr_research::behavior::{ScenarioReport, evaluate_scenario, standard_suite};
use tetr_research::bots::BotSpec;
use tetr_research::cli::{env_choice, env_f32_array, env_usize};
use tetr_research::ledger::RunLedger;
use tetr_research::seeds::seed_set;

/// Custom linear [`Weights`] from `BOARD_PARAMS` (10) + `REWARD_PARAMS` (11) — for
/// validating an `app-climb` result at any depth/width. Each group uses its shipped
/// default only when the corresponding environment variable is unset.
fn linear_custom_weights() -> Weights {
    let bp = env_f32_array("BOARD_PARAMS", BoardWeights::DT20.params());
    let rp = env_f32_array("REWARD_PARAMS", RewardWeights::SURVIVAL.params());
    let board = BoardWeights::from_params(&bp);
    let reward = RewardWeights::from_params(&rp);
    Weights { board, reward }
}

/// Parse `CC2_PARAMS` (11 comma-separated `board_params` floats) into a `Cc2Weights`,
/// for validating a `cc2-app-climb` result on the full behavior suite. Uses CC2's
/// defaults only when the variable is unset.
fn cc2_custom_weights() -> Cc2Weights {
    let params = env_f32_array("CC2_PARAMS", Cc2Weights::DEFAULT.board_params());
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

fn main() -> std::io::Result<()> {
    let seeds = seed_set(env_usize("SEEDS", 24));
    let depth = env_usize("BEAM_DEPTH", 2) as u8;
    let width = env_usize("BEAM_WIDTH", 16);
    let node_budget = env_usize("NODE_BUDGET", 4000) as u32; // best-first total expansions
    let bot = env_choice(
        "BOT",
        "dt20",
        &[
            "dt20",
            "cc2",
            "cc2custom",
            "lincustom",
            "bf",
            "bfcustom",
            "bflin",
        ],
    );

    eprintln!(
        "Behavior + APP suite | bot={bot} beam(depth={depth}, width={width}) | {} seeds",
        seeds.len()
    );

    // The arm under test, as a spec (see BOT env). The best-first arms are the
    // search-algorithm counterparts of their beam twins — same eval, beam vs
    // best-first; "bflin" pairs deep search with the near_full_rows combo
    // feature (can best-first build the cascade the beam's truncation prunes?).
    let spec = match bot.as_str() {
        "cc2" => BotSpec::beam(width, depth).cc2(Cc2Weights::DEFAULT),
        "cc2custom" => BotSpec::beam(width, depth).cc2(cc2_custom_weights()),
        "lincustom" => BotSpec::beam(width, depth).linear(linear_custom_weights()),
        "bf" => BotSpec::best_first(node_budget, depth).cc2(Cc2Weights::DEFAULT),
        "bfcustom" => BotSpec::best_first(node_budget, depth).cc2(cc2_custom_weights()),
        "bflin" => BotSpec::best_first(node_budget, depth).linear(linear_custom_weights()),
        "dt20" => BotSpec::beam(width, depth),
        _ => unreachable!("env_choice returned an unregistered value"),
    };

    let mut ledger = RunLedger::create(
        "behavior",
        serde_json::json!({
            "bot": bot,
            "bot_spec": format!("{spec:?}"),
            "seeds": seeds,
            "suite": standard_suite(),
        }),
    )?;
    let mut summaries = Vec::new();

    for scenario in standard_suite() {
        let report = evaluate_scenario(&spec.factory(), &seeds, scenario);
        for (&seed, outcome) in seeds.iter().zip(&report.outcomes) {
            ledger.append_outcome(&serde_json::json!({
                "seed": seed,
                "scenario": scenario,
                "outcome": outcome,
            }))?;
        }
        summaries.push(serde_json::json!({
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
    ledger.write_summary(serde_json::json!({
        "exit_reason": "complete",
        "scenarios": summaries,
    }))?;
    Ok(())
}
