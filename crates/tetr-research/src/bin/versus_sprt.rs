//! Versus SPRT: a sequential test of candidate board weights vs the incumbent
//! — the adaptive-spend instrument the climb's run records call for ("an SPRT
//! racer per proposal"), and the long-test verdict on a recorded candidate.
//!
//! Method: Wald's SPRT over **death-decisive** paired matches. Each block
//! draws a FRESH disjoint seed set (never reused — the climb's v1 overfitting
//! channel is structurally absent) and plays it arm-swapped (candidate as A,
//! then as B), under the same rain format the candidate was climbed in. A
//! match where exactly one side topped out is one Bernoulli trial: candidate
//! survived ⇒ win. Cap-game tiebreaks are EXCLUDED from the test by design —
//! the net-attack tiebreak is structurally anti-defensive (see the
//! `garbage_ab` record) — but are reported for context.
//!
//!   H0: p = 0.5 (the candidate is no better at surviving)
//!   H1: p = P1  (default 0.55)
//!   accept H1 when LLR ≥ ln((1−β)/α); accept H0 when LLR ≤ ln(β/(1−α))
//!
//! Self-bounded by TIME_BUDGET_SECS: an undecided run reports the running LLR
//! and counts as an honest "inconclusive" (true effects near the indifference
//! zone take unbounded samples — that is the SPRT telling you the effect is
//! small).
//!
//! The default candidate is the climb's v3 accept (versus_climb.rs RUN RECORD
//! v3: validation deaths 20-15, margin +0.79, p ≈ 0.25 — promising, unproven).
//!
//! RUN RECORD (2026-06-10, defaults): **H0 ACCEPTED in 270 s** — decisive
//! 266-269 of 544 matches (9 ties), LLR −2.99, mean margin −0.17. The v3
//! candidate has no survival edge; its 20-15 validation was noise, exactly as
//! its p ≈ 0.25 warned. Resolution cost matched Wald's prediction (~525
//! decisive matches at these settings, ~2 matches/s) — racing an accept costs
//! ~5 minutes, which is the budget figure for wiring this into the climb as a
//! second-stage confirmer (screen with cheap blocks, SPRT only what passes).
//!
//! Env: TIME_BUDGET_SECS (3600), BLOCK_SEEDS (8 ⇒ 16 matches/block), P1
//!      (0.55), RAIN_PERIOD (8), MAX_PLIES (240), BEAM_DEPTH (2), BEAM_WIDTH
//!      (16), SEED_BASE (16384 — disjoint from the climb's train/validation
//!      regions at 0.. and 4096..).

use std::time::Instant;

use tetr_core::ai::Cc2Weights;
use tetr_core::player::PlayerController;
use tetr_research::cli::env_usize;
use tetr_research::{beam_cc2_weights_bot, evaluate_versus_format, seed_set_from, VersusFormat};

/// The climb's v3 candidate (versus_climb.rs RUN RECORD v3) — the parameters
/// this bin exists to judge.
const V3_CANDIDATE: [f32; Cc2Weights::BOARD_PARAM_COUNT] = [
    -0.003_662_888_2,
    -1.573_386_2,
    -0.195_788_15,
    -0.349_775_85,
    -1.538_758_6,
    -5.149_458,
    0.357_563_6,
    0.096_651_86,
    1.550_793,
    4.478_138_4,
    3.782_923,
];

fn env_f64(key: &str, default: f64) -> f64 {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn main() {
    let budget_secs = env_usize("TIME_BUDGET_SECS", 3600) as u64;
    let block_seeds = env_usize("BLOCK_SEEDS", 8);
    let p1 = env_f64("P1", 0.55);
    let alpha = 0.05f64;
    let beta = 0.05f64;
    let depth = env_usize("BEAM_DEPTH", 2) as u8;
    let width = env_usize("BEAM_WIDTH", 16);
    let seed_base = env_usize("SEED_BASE", 16384);
    let format = VersusFormat {
        max_plies: env_usize("MAX_PLIES", 240) as u32,
        rain_period: env_usize("RAIN_PERIOD", 8) as u32,
    };

    let upper = ((1.0 - beta) / alpha).ln(); // accept H1
    let lower = (beta / (1.0 - alpha)).ln(); // accept H0
    let win_llr = (p1 / 0.5).ln();
    let loss_llr = ((1.0 - p1) / 0.5).ln();

    let weights = Cc2Weights::attack_tuned().with_board_params(&V3_CANDIDATE);
    let cand = move |s: u64| -> Box<dyn PlayerController> {
        beam_cc2_weights_bot(s, width, depth, weights)
    };
    let incumbent = move |s: u64| -> Box<dyn PlayerController> {
        beam_cc2_weights_bot(s, width, depth, Cc2Weights::attack_tuned())
    };

    println!(
        "SPRT: v3 candidate vs attack_tuned | H0 p=0.5, H1 p={p1}, bounds [{lower:.3}, {upper:.3}] \
         | beam d{depth} w{width}, rain {}, {} plies, blocks of {} seeds (x2 orientations) \
         | budget {budget_secs}s",
        format.rain_period, format.max_plies, block_seeds
    );

    let start = Instant::now();
    let (mut llr, mut wins, mut losses) = (0.0f64, 0u32, 0u32);
    let (mut ties, mut margin_sum, mut matches) = (0u32, 0.0f64, 0u32);
    let mut block = 0usize;

    let verdict = loop {
        if start.elapsed().as_secs() >= budget_secs {
            break "INCONCLUSIVE (budget)";
        }
        let seeds = seed_set_from(seed_base + block * block_seeds, block_seeds);
        block += 1;

        // Arm-swapped: the candidate plays each seed from both chairs.
        let fwd = evaluate_versus_format(&cand, &incumbent, &seeds, format);
        let rev = evaluate_versus_format(&incumbent, &cand, &seeds, format);

        // (candidate_topped, opponent_topped, candidate margin) per match.
        let per_match = fwd
            .outcomes
            .iter()
            .map(|o| {
                (
                    o.a_topped,
                    o.b_topped,
                    f64::from(o.attack_a) - f64::from(o.attack_b),
                )
            })
            .chain(rev.outcomes.iter().map(|o| {
                (
                    o.b_topped,
                    o.a_topped,
                    f64::from(o.attack_b) - f64::from(o.attack_a),
                )
            }));
        for (cand_topped, opp_topped, margin) in per_match {
            matches += 1;
            margin_sum += margin;
            match (cand_topped, opp_topped) {
                (false, true) => {
                    wins += 1;
                    llr += win_llr;
                }
                (true, false) => {
                    losses += 1;
                    llr += loss_llr;
                }
                _ => ties += 1, // double death or cap: no survival evidence
            }
        }

        println!(
            "block {block:>3} | decisive {wins}-{losses} (ties {ties}) | LLR {llr:+.3} | \
             margin {:+.2} | {}s",
            margin_sum / f64::from(matches.max(1)),
            start.elapsed().as_secs()
        );

        if llr >= upper {
            break "H1 ACCEPTED — the candidate survives more (ship-grade evidence)";
        }
        if llr <= lower {
            break "H0 ACCEPTED — no survival edge at this effect size";
        }
    };

    println!(
        "\nVERDICT: {verdict}\n\
         decisive {wins}-{losses} of {matches} matches ({ties} ties/caps) | LLR {llr:+.3} \
         in [{lower:.3}, {upper:.3}] | mean margin {:+.2} | {} blocks, {}s",
        margin_sum / f64::from(matches.max(1)),
        block,
        start.elapsed().as_secs()
    );
    println!("sprt_llr {llr:.4}");
    println!("sprt_wins {wins}");
    println!("sprt_losses {losses}");
}
