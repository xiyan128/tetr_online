//! Native Cold Clear 2 head-to-head: CC2's **ported** evaluator
//! ([`tetr_core::ai::Cc2Evaluator`]) vs our DT-20 evaluator, both on the SAME beam,
//! engine, and garbage rules. Fair by construction — no TBP, no re-sync, both bots
//! play real mutual garbage on our engine. This is the comparison the TBP bridge
//! could not give (CC2 has no garbage message), and the baseline we hillclimb past.
//!
//! RUN RECORD 2026-06-12 UTC — `20260612-032001-cc2-native-2680`
//! Defaults: 12 seeds, beam depth 2 / width 16, 160 plies, 9 garbage rows,
//! 100-piece censoring cap. CC2-eval won 9–3 (`0.75`); mean net attack was
//! 46.8 vs 39.8. Downstack censored pieces were CC2 `16.50` (clear rate `1.00`)
//! vs DT-20 `13.92` (clear rate `1.00`). (Env-var-era invocation; the knobs
//! map 1:1 onto [`Spec`] fields.)

use serde_json::json;

use tetr_core::ai::eval::Cc2Weights;

use crate::bots::BotSpec;
use crate::commands::{Beam, Runtime};
use crate::downstack::evaluate_downstack;
use crate::ledger::RunLedger;
use crate::seeds::seed_set;
use crate::versus::evaluate_versus;

#[derive(Clone, Copy, Debug, serde::Serialize)]
pub struct Spec {
    pub seeds: usize,
    pub beam: Beam,
    /// Versus ply cap.
    pub max_plies: u32,
    /// Downstack cheese height.
    pub garbage_rows: u32,
    /// Downstack censoring cap.
    pub max_pieces: u32,
}

impl Default for Spec {
    fn default() -> Self {
        Self {
            seeds: 12,
            beam: Beam::default(),
            max_plies: 160,
            garbage_rows: 9,
            max_pieces: 100,
        }
    }
}

pub fn run(spec: &Spec, _rt: &Runtime, ledger: &mut RunLedger) -> std::io::Result<()> {
    let seeds = seed_set(spec.seeds);
    let Beam { width, depth } = spec.beam;
    let (plies, rows, cap) = (spec.max_plies, spec.garbage_rows, spec.max_pieces);
    let cc2 = BotSpec::beam(width, depth).cc2(Cc2Weights::DEFAULT);
    let dt20 = BotSpec::beam(width, depth);

    eprintln!(
        "Native CC2-eval vs DT-20-eval — both on beam(depth={depth}, width={width}); {} seeds",
        seeds.len()
    );

    // --- Versus (fair, our engine, mutual garbage): A = CC2-eval, B = DT-20 ---
    let vs = evaluate_versus(&cc2.factory(), &dt20.factory(), &seeds, plies);
    for outcome in &vs.outcomes {
        ledger.append_outcome(&json!({ "suite": "versus", "outcome": outcome }))?;
    }
    println!("versus_cc2eval_win_rate {:.2}", vs.a_win_rate());
    eprintln!(
        "VERSUS  CC2-eval(A) vs DT20(B) | CC2 {} / DT20 {} / draw {} | mean attack CC2 {:.1} DT20 {:.1} | {plies} plies",
        vs.a_wins, vs.b_wins, vs.draws, vs.mean_attack_a, vs.mean_attack_b
    );

    // --- Downstack: defense (pieces, lower=better) + offense (attack, higher) ---
    let cc2_ds = evaluate_downstack(&cc2.factory(), &seeds, rows, cap);
    let dt_ds = evaluate_downstack(&dt20.factory(), &seeds, rows, cap);
    for outcome in &cc2_ds.outcomes {
        ledger
            .append_outcome(&json!({ "suite": "downstack", "arm": "cc2", "outcome": outcome }))?;
    }
    for outcome in &dt_ds.outcomes {
        ledger
            .append_outcome(&json!({ "suite": "downstack", "arm": "dt20", "outcome": outcome }))?;
    }
    eprintln!(
        "DOWNSTACK {rows} rows | CC2-eval: {:.2} pieces  {:.1} attack  {:.0}% clear  ||  DT20: {:.2} pieces  {:.1} attack  {:.0}% clear",
        cc2_ds.mean_pieces_to_clear,
        cc2_ds.mean_attack,
        cc2_ds.clear_rate * 100.0,
        dt_ds.mean_pieces_to_clear,
        dt_ds.mean_attack,
        dt_ds.clear_rate * 100.0,
    );
    println!(
        "downstack_cc2_pieces_censored {:.2}",
        cc2_ds.mean_pieces_censored
    );
    println!("downstack_cc2_clear_rate {:.2}", cc2_ds.clear_rate);
    println!(
        "downstack_dt20_pieces_censored {:.2}",
        dt_ds.mean_pieces_censored
    );
    println!("downstack_dt20_clear_rate {:.2}", dt_ds.clear_rate);
    ledger.write_summary(json!({
        "exit_reason": "complete",
        "versus": {
            "games": vs.games,
            "cc2_wins": vs.a_wins,
            "dt20_wins": vs.b_wins,
            "draws": vs.draws,
            "cc2_win_rate": vs.a_win_rate(),
            "mean_attack_cc2": vs.mean_attack_a,
            "mean_attack_dt20": vs.mean_attack_b,
        },
        "downstack": {
            "cc2": {
                "mean_pieces_censored": cc2_ds.mean_pieces_censored,
                "mean_pieces_to_clear": cc2_ds.mean_pieces_to_clear,
                "clear_rate": cc2_ds.clear_rate,
                "mean_attack": cc2_ds.mean_attack,
            },
            "dt20": {
                "mean_pieces_censored": dt_ds.mean_pieces_censored,
                "mean_pieces_to_clear": dt_ds.mean_pieces_to_clear,
                "clear_rate": dt_ds.clear_rate,
                "mean_attack": dt_ds.mean_attack,
            },
        },
    }))?;
    Ok(())
}
