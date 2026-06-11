//! Fast single-config metric for the `/autoresearch` loop.
//!
//! Runs ONE beam config over a few seeds and prints a single machine-readable
//! number (`score_per_second <x>`) for autoresearch to parse and maximize — the
//! fast counterpart to `bench-marathon`'s full multi-config sweep. The eval that
//! autoresearch tunes (`crates/tetr-core/src/ai/eval/weights.rs`) is shared by this
//! beam, so a weight change shows up directly here. Run with `--release`.
//!
//! Env: `BENCH_SEEDS` (default 6), `BEAM_DEPTH` (default 2), `BEAM_WIDTH` (default 16).

use tetr_research::bots::BotSpec;
use tetr_research::cli::env_usize;
use tetr_research::downstack::evaluate_downstack;
use tetr_research::marathon::{evaluate_capped, DEFAULT_MAX_FRAMES};
use tetr_research::seeds::seed_set;
use tetr_research::versus::evaluate_versus;

fn main() {
    let seeds = seed_set(env_usize("BENCH_SEEDS", 6));
    let depth = env_usize("BEAM_DEPTH", 2) as u8;
    let width = env_usize("BEAM_WIDTH", 16);
    let beam = BotSpec::beam(width, depth);

    // Downstack (cheese) mode: the non-gameable digging metric. Lower = better.
    if std::env::var("DOWNSTACK").is_ok() {
        let rows = env_usize("GARBAGE_ROWS", 9) as u32;
        let cap = env_usize("MAX_PIECES", 100) as u32;
        let ds = evaluate_downstack(&beam.factory(), &seeds, rows, cap);
        println!("downstack_pieces_to_clear {:.2}", ds.mean_pieces_to_clear);
        eprintln!(
            "beam depth={depth} width={width} | {} seeds | {rows} garbage rows | clear_rate={:.0}% mean_pieces_to_clear={:.2}",
            seeds.len(),
            ds.clear_rate * 100.0,
            ds.mean_pieces_to_clear,
        );
        return;
    }

    // Versus mode: full head-to-head (mutual garbage). Validation pairing is this
    // beam (A) vs the greedy baseline (B) — A should dominate. The complete
    // "beat CC2" metric once CC2 is wired as opponent B.
    if std::env::var("VERSUS").is_ok() {
        let plies = env_usize("MAX_PLIES", 120) as u32;
        let stats = evaluate_versus(&beam.factory(), &BotSpec::greedy().factory(), &seeds, plies);
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
        return;
    }

    // Piece cap keeps each iteration fast (full marathon ~930 pieces is too slow for
    // a tight loop); score/sec stays a faithful early-game scoring-rate proxy.
    let max_pieces = env_usize("MAX_PIECES", 150) as u32;
    let stats = evaluate_capped(&beam.factory(), &seeds, DEFAULT_MAX_FRAMES, max_pieces);

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
}
