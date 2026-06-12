//! Fast single-config metrics for the `/autoresearch` loop.
//!
//! Runs ONE beam config over a few seeds and prints machine-readable headline
//! metrics for autoresearch to parse — the fast counterpart to `marathon`'s
//! full multi-config sweep. The eval that autoresearch tunes
//! (`crates/tetr-core/src/ai/eval/weights.rs`) is shared by this beam, so a
//! weight change shows up directly here. Guard metrics such as downstack
//! clear rate stay on stdout beside their optimization target; human context
//! stays on stderr. Run with `--release`.

use serde_json::json;

use crate::bots::BotSpec;
use crate::commands::{Beam, Runtime};
use crate::downstack::evaluate_downstack;
use crate::ledger::RunLedger;
use crate::marathon::{DEFAULT_MAX_FRAMES, evaluate_capped};
use crate::seeds::seed_set;
use crate::versus::evaluate_versus;

#[derive(Clone, Copy, Debug, serde::Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum Suite {
    /// Capped marathon: score/sec + APP (the original iteration metric).
    Marathon,
    /// Seeded cheese: censored pieces-to-clear + clear rate (lower = better).
    Downstack,
    /// Head-to-head vs the greedy baseline (mutual garbage).
    Versus,
}

#[derive(Clone, Copy, Debug, serde::Serialize)]
pub struct Spec {
    pub suite: Suite,
    /// Seed count (more = less noise, slower loop).
    pub seeds: usize,
    pub beam: Beam,
    /// Marathon iteration cap / downstack censoring cap — part of the metric
    /// definition (the censored mean is only comparable at one cap).
    pub max_pieces: u32,
    /// Downstack cheese height.
    pub garbage_rows: u32,
    /// Versus ply cap.
    pub max_plies: u32,
}

impl Spec {
    pub fn marathon() -> Self {
        Self {
            suite: Suite::Marathon,
            seeds: 6,
            beam: Beam::default(),
            max_pieces: 150,
            garbage_rows: 9,
            max_plies: 120,
        }
    }

    pub fn downstack() -> Self {
        Self {
            suite: Suite::Downstack,
            max_pieces: 100,
            ..Self::marathon()
        }
    }

    pub fn versus() -> Self {
        Self {
            suite: Suite::Versus,
            ..Self::marathon()
        }
    }
}

pub fn run(spec: &Spec, _rt: &Runtime, ledger: &mut RunLedger) -> std::io::Result<()> {
    let seeds = seed_set(spec.seeds);
    let Beam { width, depth } = spec.beam;
    let beam = BotSpec::beam(width, depth);

    match spec.suite {
        Suite::Downstack => {
            let ds =
                evaluate_downstack(&beam.factory(), &seeds, spec.garbage_rows, spec.max_pieces);
            for outcome in &ds.outcomes {
                ledger.append_outcome(outcome)?;
            }
            ledger.write_summary(json!({
                "exit_reason": "complete",
                "games": ds.games,
                "max_pieces": ds.max_pieces,
                "mean_pieces_censored": ds.mean_pieces_censored,
                "mean_pieces_to_clear": ds.mean_pieces_to_clear,
                "mean_attack": ds.mean_attack,
                "clear_rate": ds.clear_rate,
            }))?;
            println!("downstack_pieces_censored {:.2}", ds.mean_pieces_censored);
            println!("downstack_clear_rate {:.2}", ds.clear_rate);
            eprintln!(
                "beam depth={depth} width={width} | {} seeds | {} garbage rows | clear_rate={:.0}% mean_pieces_to_clear={:.2}",
                seeds.len(),
                spec.garbage_rows,
                ds.clear_rate * 100.0,
                ds.mean_pieces_to_clear,
            );
        }
        Suite::Versus => {
            // Validation pairing is this beam (A) vs the greedy baseline (B)
            // — A should dominate. The complete "beat CC2" metric once CC2 is
            // wired as opponent B.
            let stats = evaluate_versus(
                &beam.factory(),
                &BotSpec::greedy().factory(),
                &seeds,
                spec.max_plies,
            );
            for outcome in &stats.outcomes {
                ledger.append_outcome(outcome)?;
            }
            ledger.write_summary(json!({
                "exit_reason": "complete",
                "games": stats.games,
                "a_wins": stats.a_wins,
                "b_wins": stats.b_wins,
                "draws": stats.draws,
                "a_win_rate": stats.a_win_rate(),
                "mean_attack_a": stats.mean_attack_a,
                "mean_attack_b": stats.mean_attack_b,
            }))?;
            println!("versus_a_win_rate {:.2}", stats.a_win_rate());
            eprintln!(
                "A(beam d={depth} w={width}) vs B(greedy baseline) | A {} / B {} / draw {} | attack A {:.1} B {:.1} | {} seeds, {} plies",
                stats.a_wins,
                stats.b_wins,
                stats.draws,
                stats.mean_attack_a,
                stats.mean_attack_b,
                seeds.len(),
                spec.max_plies,
            );
        }
        Suite::Marathon => {
            // Piece cap keeps each iteration fast (full marathon ~930 pieces
            // is too slow for a tight loop); score/sec stays a faithful
            // early-game scoring-rate proxy.
            let stats =
                evaluate_capped(&beam.factory(), &seeds, DEFAULT_MAX_FRAMES, spec.max_pieces);
            for outcome in &stats.outcomes {
                ledger.append_outcome(outcome)?;
            }
            ledger.write_summary(json!({
                "exit_reason": "complete",
                "games": stats.games,
                "mean_score_per_second": stats.mean_score_per_second,
                "mean_attack_per_piece": stats.mean_attack_per_piece,
                "mean_score": stats.mean_score,
                "mean_level": stats.mean_level,
                "mean_pieces": stats.mean_pieces,
                "completion_rate": stats.completion_rate,
                "topout_rate": stats.topout_rate,
                "mean_attack": stats.mean_attack,
            }))?;
            // Machine-readable metrics (one per line) for the autoresearch
            // loop to parse. `attack_per_piece` (APP) is the versus /
            // Cold-Clear-2 metric; score_per_second is the marathon proxy.
            println!("score_per_second {:.2}", stats.mean_score_per_second);
            println!("attack_per_piece {:.4}", stats.mean_attack_per_piece);
            eprintln!(
                "beam depth={depth} width={width} cap={} | {} seeds | APP={:.4} attack/game={:.1} | score={:.0} level={:.2} pieces={:.0} completion={:.0}%",
                spec.max_pieces,
                seeds.len(),
                stats.mean_attack_per_piece,
                stats.mean_attack,
                stats.mean_score,
                stats.mean_level,
                stats.mean_pieces,
                stats.completion_rate * 100.0,
            );
        }
    }
    Ok(())
}
