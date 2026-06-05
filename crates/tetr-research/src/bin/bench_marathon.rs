//! `bench-marathon` — measure Marathon scoring speed for the baseline (and, if a
//! trained model is present, the tetr-nn value net), over a deterministic seed set.
//!
//! This is the metric `/autoresearch` hill-climbs: **mean score per simulated
//! second**. Run:
//!
//! ```text
//! cargo run --release -p tetr-research --bin bench-marathon
//! ```
//!
//! The NN comparison is included automatically when
//! `crates/tetr-nn/assets/value_net.safetensors` exists (train it with
//! `training/train_value_net.py`).

use std::path::PathBuf;

use tetr_research::{
    baseline_bot, beam_linear_bot, beam_nn_bot, evaluate, nn_bot, seed_set, MarathonStats,
    DEFAULT_MAX_FRAMES,
};

/// Beam width used for the beam comparison runs.
const BEAM_WIDTH: usize = 16;

/// Default seed count (full validation run). Override with `BENCH_SEEDS=<n>` for
/// fast iteration — score/sec is a per-game mean, so fewer seeds stays comparable
/// (just noisier). This is the knob that turns the hour-long sweep into seconds.
const DEFAULT_NUM_SEEDS: usize = 24;

fn model_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../tetr-nn/assets/value_net.safetensors")
}

fn print_stats(label: &str, s: &MarathonStats) {
    println!("\n== {label} ==  ({} games)", s.games);
    println!("  score/sec (METRIC) : {:.2}", s.mean_score_per_second);
    println!("  mean score         : {:.0}", s.mean_score);
    println!("  mean level         : {:.2}", s.mean_level);
    println!("  mean pieces        : {:.0}", s.mean_pieces);
    println!("  completion rate    : {:.0}%", s.completion_rate * 100.0);
    println!("  top-out rate       : {:.0}%", s.topout_rate * 100.0);
}

/// Print the score/sec delta of `s` against `baseline`, with a BEATS/below verdict.
fn print_delta(label: &str, s: &MarathonStats, baseline: &MarathonStats) {
    let delta = s.mean_score_per_second - baseline.mean_score_per_second;
    let pct = if baseline.mean_score_per_second > 0.0 {
        100.0 * delta / baseline.mean_score_per_second
    } else {
        0.0
    };
    let verdict = if delta > 0.0 {
        "BEATS baseline"
    } else if delta == 0.0 {
        "ties baseline"
    } else {
        "below baseline"
    };
    println!(
        "\n{label} vs baseline score/sec: {delta:+.2} ({pct:+.1}%) — {verdict}"
    );
}

fn main() {
    let num_seeds = std::env::var("BENCH_SEEDS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(DEFAULT_NUM_SEEDS);
    let seeds = seed_set(num_seeds);
    println!(
        "Marathon scoring-speed benchmark — {} seeds, perfect handicap, deterministic",
        seeds.len()
    );

    let baseline = evaluate(&|s| baseline_bot(s), &seeds, DEFAULT_MAX_FRAMES);
    print_stats("baseline: greedy (linear DT20 / SURVIVAL)", &baseline);

    // --- BeamPlanner head-to-head (same linear eval, perfect handicap) -----------
    // depth-1 beam must reproduce the greedy decisions, so its score/sec ties the
    // baseline within noise — the seam-faithful gate before depth rises.
    let beam1 = evaluate(
        &|s| beam_linear_bot(s, BEAM_WIDTH, 1),
        &seeds,
        DEFAULT_MAX_FRAMES,
    );
    print_stats("beam @depth1 (== greedy check)", &beam1);
    print_delta("beam @depth1", &beam1, &baseline);

    let beam2 = evaluate(
        &|s| beam_linear_bot(s, BEAM_WIDTH, 2),
        &seeds,
        DEFAULT_MAX_FRAMES,
    );
    print_stats("beam @depth2 (linear DT20 / SURVIVAL)", &beam2);
    print_delta("beam @depth2", &beam2, &baseline);

    let beam3 = evaluate(
        &|s| beam_linear_bot(s, BEAM_WIDTH, 3),
        &seeds,
        DEFAULT_MAX_FRAMES,
    );
    print_stats("beam @depth3 (linear DT20 / SURVIVAL)", &beam3);
    print_delta("beam @depth3", &beam3, &baseline);

    let path = model_path();
    match std::fs::read(&path) {
        Ok(bytes) => {
            let nn = evaluate(&|s| nn_bot(&bytes, s), &seeds, DEFAULT_MAX_FRAMES);
            print_stats("greedy + tetr-nn value net (CPU)", &nn);
            print_delta("NN (greedy)", &nn, &baseline);

            // STEP 3: the NN dropped into the beam (pure backend swap — same planner
            // as `beam3`, only the evaluator changes). The beam batches each
            // generation's children through `evaluate_batch`, so the net runs one
            // forward pass per generation.
            let beam_nn = evaluate(
                &|s| beam_nn_bot(&bytes, s, BEAM_WIDTH, 3),
                &seeds,
                DEFAULT_MAX_FRAMES,
            );
            print_stats("beam @depth3 + tetr-nn value net (CPU)", &beam_nn);
            print_delta("beam (NN) @depth3", &beam_nn, &baseline);

            // --- 3-way headline: greedy(linear) vs beam(linear) vs beam(NN) ----------
            println!("\n== 3-WAY score/sec comparison ==");
            println!(
                "  greedy (linear DT20 / SURVIVAL) : {:.2}",
                baseline.mean_score_per_second
            );
            println!(
                "  beam   @depth3 (linear)         : {:.2}",
                beam3.mean_score_per_second
            );
            println!(
                "  beam   @depth3 (tetr-nn)        : {:.2}",
                beam_nn.mean_score_per_second
            );
        }
        Err(_) => {
            println!(
                "\n(no model at {} — baseline only. Train one with training/train_value_net.py)",
                path.display()
            );
        }
    }
}
