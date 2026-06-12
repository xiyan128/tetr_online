//! Marathon scoring-speed sweep: the greedy baseline vs the multi-ply beam
//! (same linear DT-20 eval) over a deterministic seed set. The headline is
//! **mean score per simulated second**; depth-1 beam must reproduce the
//! greedy decisions (the seam-faithful gate before depth rises).

use serde_json::json;

use crate::bots::BotSpec;
use crate::commands::Runtime;
use crate::ledger::RunLedger;
use crate::marathon::{DEFAULT_MAX_FRAMES, MarathonStats, evaluate};
use crate::seeds::seed_set;

/// Beam width used for the beam comparison runs.
const BEAM_WIDTH: usize = 16;

#[derive(Clone, Copy, Debug, serde::Serialize)]
pub struct Spec {
    /// Seed count — 24 is the full validation run; fewer stays comparable
    /// (score/sec is a per-game mean), just noisier. The knob that turns the
    /// hour-long sweep into seconds.
    pub seeds: usize,
}

impl Default for Spec {
    fn default() -> Self {
        Self { seeds: 24 }
    }
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
    println!("\n{label} vs baseline score/sec: {delta:+.2} ({pct:+.1}%) — {verdict}");
}

fn append_stats(ledger: &mut RunLedger, arm: &str, stats: &MarathonStats) -> std::io::Result<()> {
    for outcome in &stats.outcomes {
        ledger.append_outcome(&json!({ "arm": arm, "outcome": outcome }))?;
    }
    Ok(())
}

fn summary(stats: &MarathonStats) -> serde_json::Value {
    json!({
        "games": stats.games,
        "mean_score_per_second": stats.mean_score_per_second,
        "mean_score": stats.mean_score,
        "mean_level": stats.mean_level,
        "mean_pieces": stats.mean_pieces,
        "completion_rate": stats.completion_rate,
        "topout_rate": stats.topout_rate,
        "mean_attack_per_piece": stats.mean_attack_per_piece,
        "mean_attack": stats.mean_attack,
    })
}

pub fn run(spec: &Spec, _rt: &Runtime, ledger: &mut RunLedger) -> std::io::Result<()> {
    let seeds = seed_set(spec.seeds);
    println!(
        "Marathon scoring-speed benchmark — {} seeds, perfect handicap, deterministic",
        seeds.len()
    );

    let baseline = evaluate(&BotSpec::greedy().factory(), &seeds, DEFAULT_MAX_FRAMES);
    append_stats(ledger, "greedy", &baseline)?;
    print_stats("baseline: greedy (linear DT20 / SURVIVAL)", &baseline);

    // Depth-1 beam must reproduce the greedy decisions, so its score/sec ties
    // the baseline within noise — the seam-faithful gate before depth rises.
    let beam1 = evaluate(
        &BotSpec::beam(BEAM_WIDTH, 1).factory(),
        &seeds,
        DEFAULT_MAX_FRAMES,
    );
    append_stats(ledger, "beam-depth1", &beam1)?;
    print_stats("beam @depth1 (== greedy check)", &beam1);
    print_delta("beam @depth1", &beam1, &baseline);

    let beam2 = evaluate(
        &BotSpec::beam(BEAM_WIDTH, 2).factory(),
        &seeds,
        DEFAULT_MAX_FRAMES,
    );
    append_stats(ledger, "beam-depth2", &beam2)?;
    print_stats("beam @depth2 (linear DT20 / SURVIVAL)", &beam2);
    print_delta("beam @depth2", &beam2, &baseline);

    let beam3 = evaluate(
        &BotSpec::beam(BEAM_WIDTH, 3).factory(),
        &seeds,
        DEFAULT_MAX_FRAMES,
    );
    append_stats(ledger, "beam-depth3", &beam3)?;
    print_stats("beam @depth3 (linear DT20 / SURVIVAL)", &beam3);
    print_delta("beam @depth3", &beam3, &baseline);

    println!("\n== score/sec comparison ==");
    println!(
        "  greedy (linear DT20 / SURVIVAL) : {:.2}",
        baseline.mean_score_per_second
    );
    println!(
        "  beam   @depth3 (linear)         : {:.2}",
        beam3.mean_score_per_second
    );
    ledger.write_summary(json!({
        "exit_reason": "complete",
        "arms": {
            "greedy": summary(&baseline),
            "beam-depth1": summary(&beam1),
            "beam-depth2": summary(&beam2),
            "beam-depth3": summary(&beam3),
        },
    }))?;
    Ok(())
}
