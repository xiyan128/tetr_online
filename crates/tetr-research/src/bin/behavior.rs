//! Behavior + APP suite report for a bot across the standard garbage scenarios.
//! APP (attack per piece) is the primary strike metric; also reports DS/P, survival,
//! attack/line (concentration vs combo-spam), and the clear-type behavior histogram.
//!
//! Run: `SEEDS=24 BEAM_DEPTH=2 cargo run --release -p tetr-research --bin behavior`

use tetr_core::ai::eval::{BoardWeights, RewardWeights, Weights};
use tetr_core::ai::Cc2Weights;
use tetr_research::behavior::{evaluate_scenario, standard_suite, ScenarioReport};
use tetr_research::cli::env_usize;
use tetr_research::{
    beam_cc2_bot, beam_cc2_weights_bot, beam_linear_bot, beam_weights_bot,
    bestfirst_cc2_weights_bot, bestfirst_weights_bot, seed_set,
};

/// Parse a comma-separated `f32` list from env var `key` (empty if unset/malformed).
fn parse_f32_list(key: &str) -> Vec<f32> {
    std::env::var(key)
        .ok()
        .map(|s| s.split(',').filter_map(|t| t.trim().parse().ok()).collect())
        .unwrap_or_default()
}

/// Custom linear [`Weights`] from `BOARD_PARAMS` (10) + `REWARD_PARAMS` (11) — for
/// validating an `app-climb` result at any depth/width. Each group falls back to its
/// shipped default when the env list is absent or the wrong length.
fn linear_custom_weights() -> Weights {
    let bp = parse_f32_list("BOARD_PARAMS");
    let rp = parse_f32_list("REWARD_PARAMS");
    let board = <[f32; BoardWeights::PARAM_COUNT]>::try_from(bp.as_slice())
        .map_or(BoardWeights::DT20, |a| BoardWeights::from_params(&a));
    let reward = <[f32; RewardWeights::PARAM_COUNT]>::try_from(rp.as_slice())
        .map_or(RewardWeights::SURVIVAL, |a| RewardWeights::from_params(&a));
    Weights { board, reward }
}

/// Parse `CC2_PARAMS` (11 comma-separated `board_params` floats) into a `Cc2Weights`,
/// for validating a `cc2-app-climb` result on the full behavior suite. Falls back to
/// CC2's defaults when unset or malformed.
fn cc2_custom_weights() -> Cc2Weights {
    let parsed: Vec<f32> = std::env::var("CC2_PARAMS")
        .ok()
        .map(|s| s.split(',').filter_map(|t| t.trim().parse().ok()).collect())
        .unwrap_or_default();
    if parsed.len() == Cc2Weights::BOARD_PARAM_COUNT {
        let mut p = [0f32; Cc2Weights::BOARD_PARAM_COUNT];
        p.copy_from_slice(&parsed);
        Cc2Weights::DEFAULT.with_board_params(&p)
    } else {
        Cc2Weights::DEFAULT
    }
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
        t.singles, t.doubles, t.triples, t.tetrises,
        t.tspin_mini, t.tspin_single, t.tspin_double, t.tspin_triple,
        t.b2b_clears, t.combo_clears, t.max_combo, t.perfect_clears,
    );
    println!("APP[{}] {:.3}", r.scenario.label(), r.mean_app);
}

fn main() {
    let seeds = seed_set(env_usize("SEEDS", 24));
    let depth = env_usize("BEAM_DEPTH", 2) as u8;
    let width = env_usize("BEAM_WIDTH", 16);
    let node_budget = env_usize("NODE_BUDGET", 4000) as u32; // best-first total expansions
    let bot = std::env::var("BOT").unwrap_or_else(|_| "dt20".to_string());

    eprintln!(
        "Behavior + APP suite | bot={bot} beam(depth={depth}, width={width}) | {} seeds",
        seeds.len()
    );

    for scenario in standard_suite() {
        let r = match bot.as_str() {
            "cc2" => evaluate_scenario(&|s| beam_cc2_bot(s, width, depth), &seeds, scenario),
            "cc2custom" => {
                let w = cc2_custom_weights();
                evaluate_scenario(&|s| beam_cc2_weights_bot(s, width, depth, w), &seeds, scenario)
            }
            "lincustom" => {
                let w = linear_custom_weights();
                evaluate_scenario(&|s| beam_weights_bot(s, width, depth, w), &seeds, scenario)
            }
            // Best-first search over CC2's eval (DEFAULT weights), depth-capped. The
            // search-algorithm counterpart of BOT=cc2 — same eval, beam vs best-first.
            "bf" => evaluate_scenario(
                &|s| bestfirst_cc2_weights_bot(s, node_budget, depth, Cc2Weights::DEFAULT),
                &seeds,
                scenario,
            ),
            // Best-first over the warm-climbed CC2 weights (`CC2_PARAMS`) — counterpart
            // of BOT=cc2custom; best-first vs beam at the tuned attack eval.
            "bfcustom" => {
                let w = cc2_custom_weights();
                evaluate_scenario(
                    &|s| bestfirst_cc2_weights_bot(s, node_budget, depth, w),
                    &seeds,
                    scenario,
                )
            }
            // Best-first over the linear eval with custom `BOARD_PARAMS`/`REWARD_PARAMS`
            // (incl. `near_full_rows`) — the combo-synergy test: can deep best-first
            // search build the clean-board combo cascade the beam's truncation prunes?
            "bflin" => {
                let w = linear_custom_weights();
                evaluate_scenario(
                    &|s| bestfirst_weights_bot(s, node_budget, depth, w),
                    &seeds,
                    scenario,
                )
            }
            _ => evaluate_scenario(&|s| beam_linear_bot(s, width, depth), &seeds, scenario),
        };
        print_report(&r);
    }
}
