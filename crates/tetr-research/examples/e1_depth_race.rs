//! E1 (roadmap §2.1): does search depth PAST the d9 grid wall buy real versus strength?
//!
//! E0 showed the ply-1 decision is settled by the ~d6 preview horizon for 60-77% of states,
//! but still flips for 10-18% past d9 -- enough to warrant the ship-grade test. This races the
//! narrow-deep configs the Pareto grid never had (max_depth is an unbounded u8) under the
//! platform's pair-GSPRT over death-decisive, rain-forced versus matches (arm-swapped + CRN).
//!
//! Three questions:
//!  A. WITHIN the narrow family, does deeper actually win? (w16d12 vs w16d9; w16d15 vs w16d12)
//!     The skeptic's pre-registered null: w16d12 ~ 914 < w16d9 (919.9) -- i.e. deeper LOSES.
//!  B. HEADLINE: does a cheap narrow-deep config match/beat the 5x-pricier champion w128d9?
//!  C. CALIBRATION: w16d9 vs w128d9 -- the grid says ~208 Elo gap; the test must see the champion win.
//!
//! Each race reports the GSPRT verdict (H1 = cand stronger at p=0.55) AND the raw decisive
//! win-rate (small edges past d9 show here even when the verdict is H0).
//!
//! Run: cargo run --release -p tetr-research --example e1_depth_race -- [budget_secs_per_race]

use std::time::{Duration, Instant};

use tetr_core::ai::eval::Cc2Weights;
use tetr_research::bots::BotSpec;
use tetr_research::seeds::regions;
use tetr_research::sprt::{SprtConfig, SprtVerdict, sprt_race};
use tetr_research::versus::VersusFormat;

fn cfg(width: usize, depth: u8) -> BotSpec {
    BotSpec::tp_beam(width, depth).cc2(Cc2Weights::attack_tuned())
}

fn main() {
    let per_race: u64 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(360);

    // (cand_w, cand_d, inc_w, inc_d, question)
    let races: &[(usize, u8, usize, u8, &str)] = &[
        // A -- within-family depth ladder (the core test: does depth past 9 pay?)
        (16, 12, 16, 9, "A1 within-family: d12 > d9 ?"),
        (16, 15, 16, 9, "A2 within-family: d15 > d9 ?"),
        (16, 15, 16, 12, "A3 within-family: d15 > d12 ?"),
        (24, 12, 24, 9, "A4 within-family (wider): d12 > d9 ?"),
        // B -- headline: narrow-deep vs the champion
        (16, 12, 128, 9, "B1 narrow-deep vs champion"),
        (24, 12, 128, 9, "B2 narrow-deep(wider) vs champion"),
        // C -- calibration: the test must see the champion beat w16d9
        (
            16,
            9,
            128,
            9,
            "C  calibration: w16d9 vs champion (expect champion wins)",
        ),
    ];

    println!(
        "E1 pair-GSPRT | H0 p=0.5, H1 p=0.55 | rain 8, 240 plies, arm-swapped CRN | {}s/race\n",
        per_race
    );
    println!(
        "{:<46} {:>14} {:>8} {:>9} {:>7} {:>8}",
        "question (cand vs incumbent)", "verdict", "W-L", "cand_wr", "margin", "pairs"
    );

    for &(cw, cd, iw, id, q) in races {
        let config = SprtConfig {
            p1: 0.55,
            alpha: 0.05,
            beta: 0.05,
            block_seeds: 8,
            seed_base: regions::SPRT,
            max_matches: u32::MAX,
            deadline: Some(Instant::now() + Duration::from_secs(per_race)),
            ..SprtConfig::default()
        };
        let format = VersusFormat {
            max_plies: 240,
            rain_period: 8,
        };
        let label = format!("w{cw}d{cd} vs w{iw}d{id}");
        let report = sprt_race(
            &cfg(cw, cd).factory(),
            &cfg(iw, id).factory(),
            format,
            config,
        );
        let decisive = (report.wins + report.losses).max(1);
        let wr = report.wins as f64 / decisive as f64;
        let verdict = match report.verdict {
            SprtVerdict::H1Accepted => "H1 cand>inc",
            SprtVerdict::H0Accepted => "H0 no-edge",
            SprtVerdict::Inconclusive => "inconclusive",
        };
        println!(
            "{:<46} {:>14} {:>4}-{:<4} {:>8.1}% {:>+7.2} {:>8}   [{}]",
            label,
            verdict,
            report.wins,
            report.losses,
            100.0 * wr,
            report.mean_margin,
            report.pairs,
            q,
        );
    }
    println!(
        "\nread A (within-family): cand_wr > 50% AND H1 => depth past 9 pays; ~50% / H0 => the\n\
         preview-horizon ceiling holds (deeper is decoration). read B: cand_wr >= ~50% means a\n\
         config at <1/5 the champion's nodes matches/beats it. read C: champion should clearly win\n\
         (cand_wr well under 50%), proving the test discriminates a known ~208-Elo gap."
    );
}
