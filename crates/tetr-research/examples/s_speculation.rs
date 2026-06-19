//! 2.7 (roadmap): does beam SPECULATION past the ~6-ply preview buy strength, or is the
//! deep tail discounted noise? E1 showed depth helps to ~d12 then saturates; since every ply
//! past the 6-piece preview is `SPEC_DECAY=0.75` bag rollout, this ablates speculation directly.
//!
//! With speculation OFF an empty-queue node is terminal, so a deep config collapses to the
//! concrete horizon (~d6). The races:
//!   S1: w16d12 spec-ON vs spec-OFF  -> expect a big ON win (= d12 vs effectively-d6).
//!   S2: w16d12 spec-OFF vs w16d6    -> mechanistic check: OFF should ~tie d6 (~50%).
//!   S3: w32d12 spec-ON vs spec-OFF  -> the same at wider width.
//! Read: ON >> OFF AND OFF ~= d6 means the deep search IS the speculation (it is load-bearing,
//! not noise) -- so the depth saturation is speculation's per-ply value decaying, the real lever.
//!
//! Run: cargo run --release -p tetr-research --example s_speculation -- [budget_secs_per_race]

use std::time::{Duration, Instant};

use tetr_core::ai::eval::Cc2Weights;
use tetr_research::bots::BotSpec;
use tetr_research::seeds::regions;
use tetr_research::sprt::{SprtConfig, SprtVerdict, sprt_race};
use tetr_research::versus::VersusFormat;

fn tp(width: usize, depth: u8) -> BotSpec {
    BotSpec::tp_beam(width, depth).cc2(Cc2Weights::attack_tuned())
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
        "{:<34} {:>14} {:>4}-{:<4} {:>7.1}% {:>+7.2} {:>6}   [{}]",
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
        .unwrap_or(360);
    println!(
        "2.7 speculation ablation | pair-GSPRT | rain 8, 240 plies | {}s/race\n",
        secs
    );
    println!(
        "{:<34} {:>14} {:>8} {:>8} {:>7} {:>6}",
        "cand vs incumbent", "verdict", "W-L", "cand_wr", "margin", "pairs"
    );
    race(
        "w16d12 spec-ON vs spec-OFF",
        "S1 does speculation help at d12?",
        tp(16, 12),
        tp(16, 12).no_speculation(),
        secs,
    );
    race(
        "w16d12 spec-OFF vs w16d6",
        "S2 mechanistic: OFF == concrete d6?",
        tp(16, 12).no_speculation(),
        tp(16, 6),
        secs,
    );
    race(
        "w32d12 spec-ON vs spec-OFF",
        "S3 does speculation help at d12 (wider)?",
        tp(32, 12),
        tp(32, 12).no_speculation(),
        secs,
    );
    println!(
        "\nread: ON >> OFF (S1/S3) AND OFF ~= d6 (S2, ~50%) => the deep search IS speculation, and it\n\
         is load-bearing (not noise). Combined with E1 (depth saturates ~d12), this localizes the\n\
         ceiling at speculation's decaying per-ply value -- the real lever is better belief over\n\
         future pieces (expectimax / a learned speculative value / a raised preview), not raw plies."
    );
}
