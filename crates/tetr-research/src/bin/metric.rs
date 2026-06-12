//! Fast single-config metric for the `/autoresearch` loop.
//!
//! Runs ONE beam config over a few seeds and prints machine-readable headline
//! metrics for autoresearch to parse — the
//! fast counterpart to `bench-marathon`'s full multi-config sweep. The eval that
//! autoresearch tunes (`crates/tetr-core/src/ai/eval/weights.rs`) is shared by this
//! beam, so a weight change shows up directly here. Guard metrics such as downstack
//! clear rate stay on stdout beside their optimization target. Human context remains
//! on stderr. Run with `--release`.
//!
//! Env: `BENCH_SEEDS` (default 6), `BEAM_DEPTH` (default 2), `BEAM_WIDTH` (default 16).

use tetr_research::bots::BotSpec;
use tetr_research::cli::{env_flag, env_usize};
use tetr_research::downstack::evaluate_downstack;
use tetr_research::ledger::RunLedger;
use tetr_research::marathon::{DEFAULT_MAX_FRAMES, evaluate_capped};
use tetr_research::seeds::seed_set;
use tetr_research::versus::evaluate_versus;

fn main() -> std::io::Result<()> {
    let seeds = seed_set(env_usize("BENCH_SEEDS", 6));
    let depth = env_usize("BEAM_DEPTH", 2) as u8;
    let width = env_usize("BEAM_WIDTH", 16);
    let beam = BotSpec::beam(width, depth);

    // Downstack (cheese) mode: the non-gameable digging metric. Lower = better.
    if env_flag("DOWNSTACK") {
        let rows = env_usize("GARBAGE_ROWS", 9) as u32;
        let cap = env_usize("MAX_PIECES", 100) as u32;
        let mut ledger = RunLedger::create(
            "metric",
            serde_json::json!({
                "mode": "downstack",
                "bot": { "search": "beam", "depth": depth, "width": width },
                "seeds": seeds,
                "garbage_rows": rows,
                "max_pieces": cap,
            }),
        )?;
        let ds = evaluate_downstack(&beam.factory(), &seeds, rows, cap);
        for outcome in &ds.outcomes {
            ledger.append_outcome(outcome)?;
        }
        ledger.write_summary(serde_json::json!({
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
            "beam depth={depth} width={width} | {} seeds | {rows} garbage rows | clear_rate={:.0}% mean_pieces_to_clear={:.2}",
            seeds.len(),
            ds.clear_rate * 100.0,
            ds.mean_pieces_to_clear,
        );
        return Ok(());
    }

    // Versus mode: full head-to-head (mutual garbage). Validation pairing is this
    // beam (A) vs the greedy baseline (B) — A should dominate. The complete
    // "beat CC2" metric once CC2 is wired as opponent B.
    if env_flag("VERSUS") {
        let plies = env_usize("MAX_PLIES", 120) as u32;
        let mut ledger = RunLedger::create(
            "metric",
            serde_json::json!({
                "mode": "versus",
                "arm_a": { "search": "beam", "depth": depth, "width": width },
                "arm_b": { "search": "greedy" },
                "seeds": seeds,
                "max_plies": plies,
            }),
        )?;
        let stats = evaluate_versus(&beam.factory(), &BotSpec::greedy().factory(), &seeds, plies);
        for outcome in &stats.outcomes {
            ledger.append_outcome(outcome)?;
        }
        ledger.write_summary(serde_json::json!({
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
            "A(beam d={depth} w={width}) vs B(greedy baseline) | A {} / B {} / draw {} | attack A {:.1} B {:.1} | {} seeds, {plies} plies",
            stats.a_wins,
            stats.b_wins,
            stats.draws,
            stats.mean_attack_a,
            stats.mean_attack_b,
            seeds.len(),
        );
        return Ok(());
    }

    // Piece cap keeps each iteration fast (full marathon ~930 pieces is too slow for
    // a tight loop); score/sec stays a faithful early-game scoring-rate proxy.
    let max_pieces = env_usize("MAX_PIECES", 150) as u32;
    let mut ledger = RunLedger::create(
        "metric",
        serde_json::json!({
            "mode": "marathon",
            "bot": { "search": "beam", "depth": depth, "width": width },
            "seeds": seeds,
            "max_frames": DEFAULT_MAX_FRAMES,
            "max_pieces": max_pieces,
        }),
    )?;
    let stats = evaluate_capped(&beam.factory(), &seeds, DEFAULT_MAX_FRAMES, max_pieces);
    for outcome in &stats.outcomes {
        ledger.append_outcome(outcome)?;
    }
    ledger.write_summary(serde_json::json!({
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

    // Machine-readable metrics (one per line) for the autoresearch loop to parse.
    // `attack_per_piece` (APP) is the versus / Cold-Clear-2 metric; score_per_second
    // is the marathon proxy.
    println!("score_per_second {:.2}", stats.mean_score_per_second);
    println!("attack_per_piece {:.4}", stats.mean_attack_per_piece);
    // Human context on stderr (kept off the parsed stdout lines).
    eprintln!(
        "beam depth={depth} width={width} cap={max_pieces} | {} seeds | APP={:.4} attack/game={:.1} | score={:.0} level={:.2} pieces={:.0} completion={:.0}%",
        seeds.len(),
        stats.mean_attack_per_piece,
        stats.mean_attack,
        stats.mean_score,
        stats.mean_level,
        stats.mean_pieces,
        stats.completion_rate * 100.0,
    );
    Ok(())
}
