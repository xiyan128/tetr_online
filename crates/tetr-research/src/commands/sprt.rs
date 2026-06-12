//! Versus SPRT: a sequential test of candidate board weights vs the incumbent
//! — the standalone face of the shared racer in [`crate::sprt`] (the climb
//! calls the same racer as its per-accept confirmer).
//!
//! Method and exclusions are the module's (death-decisive seed PAIRS,
//! pair-level GSPRT, fresh arm-swapped blocks, honest `Inconclusive` on
//! budget). The registry's `race-v3-candidate` entry pins the climb's v3
//! accept (see the climb command's RUN RECORD v3) so the record below stays
//! runnable by name.
//!
//! RUN RECORD (2026-06-10, defaults): **H0 ACCEPTED in 270 s** — decisive
//! 266-269 of 544 matches (9 ties), LLR −2.99, mean margin −0.17. The v3
//! candidate has no survival edge; its 20-15 validation was noise, exactly as
//! its p ≈ 0.25 warned. Resolution cost matched Wald's prediction (~525
//! decisive matches at these settings, ~2 matches/s) — racing an accept costs
//! ~5 minutes, which is the budget figure behind the climb's confirmer
//! default.
//!
//! NOTE: the RUN RECORD above predates both the pair test (its LLR was the
//! per-game walk) and this CLI (it was an env-var invocation, then-default
//! block of 8). The verdict stands; re-deriving the trajectory needs the
//! pre-pair commit.

use std::time::Instant;

use serde_json::json;

use tetr_core::ai::Cc2Weights;

use crate::bots::BotSpec;
use crate::commands::{Beam, BoardParams, Runtime};
use crate::ledger::RunLedger;
use crate::seeds::regions;
use crate::sprt::{SprtConfig, SprtVerdict, sprt_race};
use crate::versus::VersusFormat;

/// The climb's v3 candidate — judged and REJECTED by this command's run
/// record; pinned in the registry so the record reproduces by name.
pub const V3_CANDIDATE: BoardParams = [
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

#[derive(Clone, Copy, Debug, serde::Serialize)]
pub struct Spec {
    /// Candidate CC2 board params (raced vs `attack_tuned`).
    pub cand_params: BoardParams,
    /// H1 per-decisive-game win probability (H0 is 0.5).
    pub p1: f64,
    /// Type-I error bound.
    pub alpha: f64,
    /// Type-II error bound.
    pub beta: f64,
    /// Seeds per block (each block = 2× this in matches, arm-swapped).
    pub block_seeds: usize,
    /// First seed index of the race's region.
    pub seed_base: usize,
    pub format: VersusFormat,
    pub beam: Beam,
}

impl Default for Spec {
    fn default() -> Self {
        Self {
            cand_params: V3_CANDIDATE,
            p1: 0.55,
            alpha: 0.05,
            beta: 0.05,
            block_seeds: 8,
            seed_base: regions::SPRT,
            format: VersusFormat {
                max_plies: 240,
                rain_period: 8,
            },
            beam: Beam::default(),
        }
    }
}

/// Default wall-clock budget (`--budget-secs` overrides).
const DEFAULT_BUDGET_SECS: u64 = 3600;

pub fn run(spec: &Spec, rt: &Runtime, ledger: &mut RunLedger) -> std::io::Result<()> {
    let Beam { width, depth } = spec.beam;
    let budget = rt.budget(DEFAULT_BUDGET_SECS);
    let config = SprtConfig {
        p1: spec.p1,
        alpha: spec.alpha,
        beta: spec.beta,
        block_seeds: spec.block_seeds,
        seed_base: spec.seed_base,
        max_matches: u32::MAX,
        deadline: Some(Instant::now() + budget),
        verbose: true,
        ..SprtConfig::default()
    };

    let weights = Cc2Weights::attack_tuned().with_board_params(&spec.cand_params);
    let cand = BotSpec::beam(width, depth).cc2(weights).factory();
    let incumbent = BotSpec::beam(width, depth)
        .cc2(Cc2Weights::attack_tuned())
        .factory();

    println!(
        "pair-GSPRT: candidate vs attack_tuned | H0 p=0.5, H1 p={} | beam d{depth} w{width}, \
         rain {}, {} plies, blocks of {} seeds (x2 orientations = paired) | budget {}s",
        spec.p1,
        spec.format.rain_period,
        spec.format.max_plies,
        spec.block_seeds,
        budget.as_secs()
    );

    let start = Instant::now();
    let report = sprt_race(&cand, &incumbent, spec.format, config);

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

    ledger.write_summary(json!({
        "exit_reason": if report.verdict == SprtVerdict::Inconclusive { "time_budget" } else { "complete" },
        "verdict": format!("{:?}", report.verdict),
        "wins": report.wins,
        "losses": report.losses,
        "ties": report.ties,
        "matches": report.matches,
        "pairs": report.pairs,
        "pair_counts": report.pair_counts,
        "llr": report.llr,
        "trinomial_llr": report.trinomial_llr,
        "pair_correlation": report.pair_correlation,
        "mean_margin": report.mean_margin,
    }))?;
    Ok(())
}
