//! Promotion panel: the one gate between "my climb accepted it" and "it is
//! the better bot".
//!
//! A climb optimizes against ONE incumbent in ONE format, so its winner may
//! be a one-trick candidate — overfit to the incumbent's style or to rain
//! pressure. Promotion therefore races the candidate against a PANEL across
//! formats, each cell a pair-level GSPRT ([`tetr_research::sprt`]) on fresh
//! campaign seeds:
//!
//! | opponent | bar | why |
//! |---|---|---|
//! | greedy baseline | H1 in every format | the floor — losing any format to greedy is disqualifying |
//! | campaign origin | H1 in every format | the campaign's whole point is beating where it started |
//! | incumbent | H1, or inconclusive while not behind on decisives | must not regress the best we have |
//!
//! An H0 anywhere REJECTS. A budget-starved inconclusive against greedy or
//! the origin also rejects — promotion needs evidence, and "we ran out of
//! time" is not evidence. The verdict prints as `promote_verdict` with one
//! machine line per cell.
//!
//! `FINAL_VALIDATION=1` draws from [`regions::FINAL`] instead of the
//! campaign's promotion region — the never-iterated reserve that backs ONE
//! verdict per external claim. Run it exactly once, when the claim is
//! drafted and nothing will be tuned afterwards; the run manifest records
//! the spend.
//!
//! Env: CAMPAIGN ("scratch"), CAND_PARAMS (the candidate's 11 CC2 board
//!      params; defaults to the origin, making the default run a null check
//!      — origin cells tie out, nothing promotes), INCUMBENT_PARAMS
//!      (defaults to the origin), CELL_MATCHES (800 per cell), P1 (0.55),
//!      ALPHA (0.05), RAIN_PERIOD (8 — the rainy half of the format axis),
//!      MAX_PLIES (240), BEAM_DEPTH (2), BEAM_WIDTH (16), TIME_BUDGET_SECS
//!      (3600, shared by all cells), FINAL_VALIDATION (presence flag).

use std::time::{Duration, Instant};

use serde_json::json;

use tetr_core::ai::Cc2Weights;
use tetr_core::player::PlayerController;
use tetr_research::bots::BotSpec;
use tetr_research::cli::{env_f32_array, env_f64, env_flag, env_string, env_usize};
use tetr_research::ledger::RunLedger;
use tetr_research::seeds::{Campaign, regions};
use tetr_research::sprt::{SprtConfig, SprtVerdict, sprt_race};
use tetr_research::versus::VersusFormat;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Opponent {
    Greedy,
    Origin,
    Incumbent,
}

impl Opponent {
    fn name(self) -> &'static str {
        match self {
            Opponent::Greedy => "greedy",
            Opponent::Origin => "origin",
            Opponent::Incumbent => "incumbent",
        }
    }
}

/// One cell's bar. Greedy and origin must be BEATEN (H1); the incumbent
/// must not be regressed — H1, or an in-budget inconclusive while not
/// behind on decisive games (the incumbent may simply be near-equal, which
/// no finite race resolves).
fn cell_passes(opponent: Opponent, verdict: SprtVerdict, wins: u32, losses: u32) -> bool {
    match opponent {
        Opponent::Greedy | Opponent::Origin => verdict == SprtVerdict::H1Accepted,
        Opponent::Incumbent => {
            verdict == SprtVerdict::H1Accepted
                || (verdict == SprtVerdict::Inconclusive && wins >= losses)
        }
    }
}

fn main() {
    const N: usize = Cc2Weights::BOARD_PARAM_COUNT;
    let origin = Cc2Weights::attack_tuned().board_params();
    let campaign = Campaign::derive(&env_string("CAMPAIGN", "scratch"));
    let cand_params = env_f32_array::<N>("CAND_PARAMS", origin);
    let incumbent_params = env_f32_array::<N>("INCUMBENT_PARAMS", origin);
    let cell_matches = env_usize("CELL_MATCHES", 800) as u32;
    let p1 = env_f64("P1", 0.55);
    let alpha = env_f64("ALPHA", 0.05);
    let rain = env_usize("RAIN_PERIOD", 8) as u32;
    let max_plies = env_usize("MAX_PLIES", 240) as u32;
    let depth = env_usize("BEAM_DEPTH", 2) as u8;
    let width = env_usize("BEAM_WIDTH", 16);
    let budget_secs = env_usize("TIME_BUDGET_SECS", 3600) as u64;
    let final_validation = env_flag("FINAL_VALIDATION");

    let stride = 4096usize.max(cell_matches as usize);
    let opponents = [Opponent::Greedy, Opponent::Origin, Opponent::Incumbent];
    let rains = [0u32, rain];
    let cells = opponents.len() * rains.len();
    // The campaign's promotion sub-region, or — exactly once per external
    // claim — the never-iterated FINAL reserve.
    let seed_base = if final_validation {
        eprintln!(
            "==== FINAL VALIDATION: spending the never-iterated region. One verdict \
             per claim; do not tune after this. ===="
        );
        regions::FINAL
    } else {
        campaign.promote(cells * stride)
    };

    let mut ledger = RunLedger::create(
        "promote",
        json!({
            "campaign": { "id": campaign.id, "slot": campaign.slot },
            "final_validation": final_validation,
            "candidate": cand_params.as_slice(),
            "incumbent": incumbent_params.as_slice(),
            "bot": format!("beam(d{depth}, w{width}) cc2 attack_tuned+board_params"),
        }),
    )
    .expect("promote: cannot create the run ledger");

    let cc2 = |params: &[f32; N]| {
        BotSpec::beam(width, depth)
            .cc2(Cc2Weights::attack_tuned().with_board_params(params))
            .factory()
    };
    let cand = cc2(&cand_params);
    let opponent_bot = |o: Opponent| -> Box<dyn Fn(u64) -> Box<dyn PlayerController> + Sync> {
        match o {
            Opponent::Greedy => Box::new(BotSpec::greedy().factory()),
            Opponent::Origin => Box::new(cc2(&origin)),
            Opponent::Incumbent => Box::new(cc2(&incumbent_params)),
        }
    };

    eprintln!(
        "Promotion panel — campaign '{}' (slot {}) | candidate vs {{greedy, origin, incumbent}} \
         x rain {{0, {rain}}} | {cell_matches} matches/cell | budget {budget_secs}s",
        campaign.id, campaign.slot
    );

    let start = Instant::now();
    let deadline = start + Duration::from_secs(budget_secs);
    let mut all_pass = true;
    for (i, (&opponent, &rain_period)) in opponents
        .iter()
        .flat_map(|o| rains.iter().map(move |r| (o, r)))
        .enumerate()
    {
        let format = VersusFormat {
            max_plies,
            rain_period,
        };
        let report = sprt_race(
            &cand,
            &*opponent_bot(opponent),
            format,
            SprtConfig {
                p1,
                alpha,
                seed_base: seed_base + i * stride,
                max_matches: cell_matches,
                deadline: Some(deadline),
                ..SprtConfig::default()
            },
        );
        let pass = cell_passes(opponent, report.verdict, report.wins, report.losses);
        all_pass &= pass;
        eprintln!(
            "cell {}/{cells} | vs {:<9} rain {rain_period} | {} | decisive {}-{} of {} pairs | \
             LLR {:+.2} | {}",
            i + 1,
            opponent.name(),
            match report.verdict {
                SprtVerdict::H1Accepted => "H1",
                SprtVerdict::H0Accepted => "H0",
                SprtVerdict::Inconclusive => "inconclusive",
            },
            report.wins,
            report.losses,
            report.pairs,
            report.llr,
            if pass { "PASS" } else { "FAIL" },
        );
        println!(
            "promote_cell_{}_rain{rain_period} {}",
            opponent.name(),
            if pass { "PASS" } else { "FAIL" }
        );
        let _ = ledger.append_outcome(&json!({
            "cell": i,
            "opponent": opponent.name(),
            "rain_period": rain_period,
            "verdict": format!("{:?}", report.verdict),
            "wins": report.wins,
            "losses": report.losses,
            "ties": report.ties,
            "pairs": report.pairs,
            "pair_counts": report.pair_counts,
            "llr": report.llr,
            "trinomial_llr": report.trinomial_llr,
            "pair_correlation": report.pair_correlation,
            "pass": pass,
        }));
    }

    let verdict = if all_pass { "PROMOTE" } else { "REJECT" };
    eprintln!(
        "\nPANEL VERDICT: {verdict} ({})",
        if all_pass {
            "every cell passed its bar"
        } else {
            "at least one cell failed — the candidate is not the better bot"
        }
    );
    println!("promote_verdict {verdict}");

    let _ = ledger.write_summary(json!({
        "exit_reason": if start.elapsed().as_secs() >= budget_secs {
            "time_budget"
        } else {
            "complete"
        },
        "verdict": verdict,
        "final_validation": final_validation,
    }));
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Greedy and origin demand proof; the incumbent demands non-regression.
    #[test]
    fn the_bars_match_the_doc_table() {
        use Opponent::*;
        use SprtVerdict::*;
        assert!(cell_passes(Greedy, H1Accepted, 10, 0));
        assert!(!cell_passes(Greedy, Inconclusive, 10, 0));
        assert!(!cell_passes(Origin, Inconclusive, 10, 0));
        assert!(!cell_passes(Origin, H0Accepted, 0, 10));
        assert!(cell_passes(Incumbent, H1Accepted, 10, 0));
        assert!(cell_passes(Incumbent, Inconclusive, 5, 5));
        assert!(cell_passes(Incumbent, Inconclusive, 6, 5));
        assert!(!cell_passes(Incumbent, Inconclusive, 4, 5));
        assert!(!cell_passes(Incumbent, H0Accepted, 5, 5));
    }
}
