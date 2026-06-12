//! The standalone pair-GSPRT racer: a ship-grade sequential verdict on one
//! candidate bot vs an incumbent bot — the same racer ([`crate::sprt`]) the
//! climb uses as its confirmer.
//!
//! RUN RECORD (2026-06-10, defaults, v3-candidate vs attack-tuned): **H0
//! ACCEPTED in 270 s** — decisive 266-269 of 544 matches (9 ties), LLR
//! −2.99, mean margin −0.17. The v3 candidate has no survival edge; its
//! 20-15 validation was noise, exactly as its p ≈ 0.25 warned. Racing an
//! accept costs ~5 minutes — the budget figure behind the climb's confirmer
//! default. (Pre-pair-test, env-var-era invocation; verdict stands,
//! trajectory reproduces only at the pre-pair commit.)
//!
//! RUN RECORD (2026-06-12, attack-tuned-d3): the first ship-grade strength
//! gain on this platform — the v3 epilogue's "deeper search" lever pays.
//! vs attack-tuned: **H1 in 4 s**, 53-8 of 64 (LLR +3.09, margin +13.0),
//! run `20260612-070034-race-66828`. vs cc2-default: **H1 in 6 s**, 58-18 of 80 (LLR +3.16,
//! margin +16.9), run `20260612-070056-race-67807`. Same-eval downstack also improves
//! (censored 18.67 vs 21.50, attack-while-digging 11.0 vs 7.3). Cost: ~16×
//! search nodes per move. attack-tuned-d3 is the incumbent to beat.

use std::time::Instant;

use crate::bots::Bot;
use crate::commands::Runtime;
use crate::seeds::regions;
use crate::sprt::{SprtConfig, SprtVerdict, sprt_race};
use crate::versus::VersusFormat;

#[derive(Clone, Copy, Debug, serde::Serialize)]
pub struct Spec {
    /// H1 per-decisive-game win probability (H0 is 0.5).
    pub p1: f64,
    pub alpha: f64,
    pub beta: f64,
    /// Seeds per block (each block = 2× this in matches, arm-swapped).
    pub block_seeds: usize,
    /// First seed index of the race's region.
    pub seed_base: usize,
    pub format: VersusFormat,
}

impl Default for Spec {
    fn default() -> Self {
        Self {
            p1: 0.55,
            alpha: 0.05,
            beta: 0.05,
            block_seeds: 8,
            seed_base: regions::SPRT,
            format: VersusFormat {
                max_plies: 240,
                rain_period: 8,
            },
        }
    }
}

/// Default wall-clock budget (`--budget-secs` overrides).
const DEFAULT_BUDGET_SECS: u64 = 3600;

pub fn run(spec: &Spec, cand: &Bot, incumbent: &Bot, rt: &Runtime) -> std::io::Result<()> {
    let budget = rt.budget(DEFAULT_BUDGET_SECS);
    let config = SprtConfig {
        p1: spec.p1,
        alpha: spec.alpha,
        beta: spec.beta,
        block_seeds: spec.block_seeds,
        seed_base: spec.seed_base,
        max_matches: u32::MAX,
        deadline: Some(Instant::now() + budget),
        ..SprtConfig::default()
    };

    println!(
        "pair-GSPRT: {} vs {} | H0 p=0.5, H1 p={} | rain {}, {} plies, \
         blocks of {} seeds (x2 orientations = paired) | budget {}s",
        cand.name,
        incumbent.name,
        spec.p1,
        spec.format.rain_period,
        spec.format.max_plies,
        spec.block_seeds,
        budget.as_secs()
    );

    let start = Instant::now();
    let report = sprt_race(
        &cand.spec.factory(),
        &incumbent.spec.factory(),
        spec.format,
        config,
    );

    let verdict = match report.verdict {
        SprtVerdict::H1Accepted => {
            "H1 ACCEPTED — the candidate survives more (ship-grade evidence)"
        }
        SprtVerdict::H0Accepted => "H0 ACCEPTED — no survival edge at this effect size",
        SprtVerdict::Inconclusive => "INCONCLUSIVE (budget)",
    };
    println!(
        "\nVERDICT: {verdict}\n\
         decisive {}-{} of {} matches / {} pairs ({} ties/caps) | pair scores [-2..+2] {:?} | \
         LLR {:+.3} (trinomial cross-check {:+.3}) | within-pair corr {} | \
         mean margin {:+.2} | {}s",
        report.wins,
        report.losses,
        report.matches,
        report.pairs,
        report.ties,
        report.pair_counts,
        report.llr,
        report.trinomial_llr,
        report
            .pair_correlation
            .map_or("n/a".to_string(), |c| format!("{c:+.2}")),
        report.mean_margin,
        start.elapsed().as_secs()
    );
    println!("sprt_llr {:.4}", report.llr);
    println!("sprt_trinomial_llr {:.4}", report.trinomial_llr);
    println!("sprt_pairs {}", report.pairs);
    println!("sprt_wins {}", report.wins);
    println!("sprt_losses {}", report.losses);
    Ok(())
}
