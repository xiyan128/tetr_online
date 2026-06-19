//! Follow-up to E1 (roadmap §2.1 "how far can we go" + §2.8): the actionable champion-scale tests.
//!
//! E1 showed depth past the d9 cap pays at narrow widths (d12 > d9 at 58-62%) but saturates by
//! ~d12, and narrow-deep loses to the wide champion (width is a survival hedge). The remaining
//! actionable questions are at the CHAMPION's scale:
//!   F1: does pushing the champion itself deeper help?  w128d12 vs w128d9.
//!   F2: at the champion's compute +45%, is the extra spend better as DEPTH or WIDTH (the
//!       reallocation crux)?  w128d12 (1408 nodes) vs w176d9 (1408 nodes) -- iso-node.
//!   F3 (§2.8): is best-first a cheaper/equal champion at matched nodes?  best_first(972,9) vs w128d9.
//!
//! Run: cargo run --release -p tetr-research --example f2_champion_depth -- [budget_secs_per_race]

use std::time::{Duration, Instant};

use tetr_core::ai::eval::Cc2Weights;
use tetr_research::bots::BotSpec;
use tetr_research::seeds::regions;
use tetr_research::sprt::{SprtConfig, SprtVerdict, sprt_race};
use tetr_research::versus::VersusFormat;

fn tp(width: usize, depth: u8) -> BotSpec {
    BotSpec::tp_beam(width, depth).cc2(Cc2Weights::attack_tuned())
}
fn bf(budget: u32, depth: u8) -> BotSpec {
    BotSpec::best_first(budget, depth).cc2(Cc2Weights::attack_tuned())
}

fn race(label: &str, q: &str, cand: BotSpec, inc: BotSpec, secs: u64) {
    let config = SprtConfig {
        p1: 0.55,
        alpha: 0.05,
        beta: 0.05,
        block_seeds: 8,
        seed_base: regions::SPRT,
        max_matches: u32::MAX,
        deadline: Some(Instant::now() + Duration::from_secs(secs)),
        ..SprtConfig::default()
    };
    let format = VersusFormat {
        max_plies: 240,
        rain_period: 8,
    };
    let r = sprt_race(&cand.factory(), &inc.factory(), format, config);
    let dec = (r.wins + r.losses).max(1);
    let verdict = match r.verdict {
        SprtVerdict::H1Accepted => "H1 cand>inc",
        SprtVerdict::H0Accepted => "H0 no-edge",
        SprtVerdict::Inconclusive => "inconclusive",
    };
    println!(
        "{:<26} {:>14} {:>4}-{:<4} {:>7.1}% {:>+7.2} {:>6}   [{}]",
        label,
        verdict,
        r.wins,
        r.losses,
        100.0 * r.wins as f64 / dec as f64,
        r.mean_margin,
        r.pairs,
        q,
    );
}

fn main() {
    let secs: u64 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(600);
    println!(
        "champion-scale follow-up | pair-GSPRT H0 p=0.5 H1 p=0.55 | rain 8, 240 plies | {}s/race\n",
        secs
    );
    println!(
        "{:<26} {:>14} {:>8} {:>8} {:>7} {:>6}",
        "cand vs incumbent", "verdict", "W-L", "cand_wr", "margin", "pairs"
    );
    // F1: push the champion deeper
    race(
        "w128d12 vs w128d9",
        "F1 deeper champion (1408 vs 972 nodes)",
        tp(128, 12),
        tp(128, 9),
        secs,
    );
    // F2: iso-node reallocation at the top -- depth vs width at equal compute (1408 nodes each)
    race(
        "w128d12 vs w176d9",
        "F2 iso-node: depth vs width @ +45% compute",
        tp(128, 12),
        tp(176, 9),
        secs,
    );
    // F3 (2.8): node-matched best-first vs the champion
    race(
        "bf(972,9) vs w128d9",
        "F3 §2.8 best-first node-matched",
        bf(972, 9),
        tp(128, 9),
        secs,
    );
    println!(
        "\nread F1: cand_wr > 50% => the deeper champion is stronger (bump the depth cap; pay +45% compute).\n\
         read F2: cand_wr > 50% => at the top the extra compute is better as DEPTH than WIDTH (champion\n\
         under-allocates depth); < 50% => width's survival hedge wins at scale (champion is well-allocated).\n\
         read F3: cand_wr >= ~50% => best-first matches the beam per node at versus (the cheaper-champion claim)."
    );
}
