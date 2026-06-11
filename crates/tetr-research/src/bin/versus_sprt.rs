//! Versus SPRT: a sequential test of candidate board weights vs the incumbent
//! — the standalone face of the shared racer in [`tetr_research::sprt`] (the
//! climb calls the same racer as its per-accept confirmer).
//!
//! Method and exclusions are the module's (death-decisive matches only, fresh
//! arm-swapped seed blocks, honest `Inconclusive` on budget). The default
//! candidate is the climb's v3 accept (versus_climb.rs RUN RECORD v3).
//!
//! RUN RECORD (2026-06-10, defaults): **H0 ACCEPTED in 270 s** — decisive
//! 266-269 of 544 matches (9 ties), LLR −2.99, mean margin −0.17. The v3
//! candidate has no survival edge; its 20-15 validation was noise, exactly as
//! its p ≈ 0.25 warned. Resolution cost matched Wald's prediction (~525
//! decisive matches at these settings, ~2 matches/s) — racing an accept costs
//! ~5 minutes, which is the budget figure behind the climb's confirmer
//! default.
//!
//! Env: TIME_BUDGET_SECS (3600), BLOCK_SEEDS (8 ⇒ 16 matches/block), P1
//!      (0.55), RAIN_PERIOD (8), MAX_PLIES (240), BEAM_DEPTH (2), BEAM_WIDTH
//!      (16), SEED_BASE (regions::SPRT = 16384 — see `seeds::regions` for the
//!      full partition; disjoint from the climb's regions by construction).
//!
//! NOTE: the RUN RECORD above was produced with the then-default
//! BLOCK_SEEDS=8 (pre-parallelism). The default is now 24 (pool-saturating);
//! the verdict stands, and re-deriving the exact LLR trajectory needs
//! BLOCK_SEEDS=8.

use std::time::{Duration, Instant};

use tetr_core::ai::Cc2Weights;
use tetr_research::bots::BotSpec;
use tetr_research::cli::{env_f64, env_usize};
use tetr_research::seeds::regions;
use tetr_research::sprt::{SprtConfig, SprtVerdict, sprt_race};
use tetr_research::versus::VersusFormat;

/// The climb's v3 candidate (versus_climb.rs RUN RECORD v3) — judged and
/// REJECTED by this bin's run record above; kept as the default so the record
/// reproduces.
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

fn main() {
    let budget_secs = env_usize("TIME_BUDGET_SECS", 3600) as u64;
    let depth = env_usize("BEAM_DEPTH", 2) as u8;
    let width = env_usize("BEAM_WIDTH", 16);
    let format = VersusFormat {
        max_plies: env_usize("MAX_PLIES", 240) as u32,
        rain_period: env_usize("RAIN_PERIOD", 8) as u32,
    };
    let config = SprtConfig {
        p1: env_f64("P1", 0.55),
        block_seeds: env_usize("BLOCK_SEEDS", 8),
        seed_base: env_usize("SEED_BASE", regions::SPRT),
        max_matches: u32::MAX,
        deadline: Some(Instant::now() + Duration::from_secs(budget_secs)),
        verbose: true,
        ..SprtConfig::default()
    };

    let weights = Cc2Weights::attack_tuned().with_board_params(&V3_CANDIDATE);
    let cand = BotSpec::beam(width, depth).cc2(weights).factory();
    let incumbent = BotSpec::beam(width, depth)
        .cc2(Cc2Weights::attack_tuned())
        .factory();

    println!(
        "SPRT: v3 candidate vs attack_tuned | H0 p=0.5, H1 p={} | beam d{depth} w{width}, \
         rain {}, {} plies, blocks of {} seeds (x2 orientations) | budget {budget_secs}s",
        config.p1, format.rain_period, format.max_plies, config.block_seeds
    );

    let start = Instant::now();
    let report = sprt_race(&cand, &incumbent, format, config);

    let verdict = match report.verdict {
        SprtVerdict::H1Accepted => {
            "H1 ACCEPTED — the candidate survives more (ship-grade evidence)"
        }
        SprtVerdict::H0Accepted => "H0 ACCEPTED — no survival edge at this effect size",
        SprtVerdict::Inconclusive => "INCONCLUSIVE (budget)",
    };
    println!(
        "\nVERDICT: {verdict}\n\
         decisive {}-{} of {} matches ({} ties/caps) | LLR {:+.3} | mean margin {:+.2} | {}s",
        report.wins,
        report.losses,
        report.matches,
        report.ties,
        report.llr,
        report.mean_margin,
        start.elapsed().as_secs()
    );
    println!("sprt_llr {:.4}", report.llr);
    println!("sprt_wins {}", report.wins);
    println!("sprt_losses {}", report.losses);
}
