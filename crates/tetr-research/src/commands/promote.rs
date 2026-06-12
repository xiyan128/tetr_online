//! Promotion panel: the one gate between "my climb accepted it" and "it is
//! the better bot".
//!
//! A climb optimizes against ONE incumbent in ONE format, so its winner may
//! be a one-trick candidate — overfit to the incumbent's style or to rain
//! pressure. Promotion therefore races the candidate against a PANEL across
//! formats, each cell a pair-level GSPRT ([`crate::sprt`]) on fresh
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
//! A spec with `final_validation: true` draws from [`regions::FINAL`]
//! instead of the campaign's promotion region — the never-iterated reserve
//! that backs ONE verdict per external claim. Register it as its own named
//! entry (the name and manifest then record the spend), run it exactly once,
//! when the claim is drafted and nothing will be tuned afterwards.
//!
//! The registered candidate comes from the registry — a promotion IS a named
//! configuration: paste the climb's `best_params` into a new entry and run
//! it by name.

use std::time::{Duration, Instant};

use serde_json::json;

use tetr_core::ai::Cc2Weights;
use tetr_core::player::PlayerController;

use crate::bots::BotSpec;
use crate::commands::{Beam, BoardParams, Runtime};
use crate::ledger::RunLedger;
use crate::seeds::{Campaign, regions};
use crate::sprt::{SprtConfig, SprtVerdict, sprt_race};
use crate::versus::VersusFormat;

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum Opponent {
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

#[derive(Clone, Debug, serde::Serialize)]
pub struct Spec {
    /// The campaign whose promotion sub-region judges this candidate.
    pub campaign: String,
    pub cand_params: BoardParams,
    pub incumbent_params: BoardParams,
    /// Match cap per panel cell.
    pub cell_matches: u32,
    pub p1: f64,
    pub alpha: f64,
    /// The rainy half of the format axis (the other half is rain 0).
    pub rain_period: u32,
    pub max_plies: u32,
    pub beam: Beam,
    /// Draw from the never-iterated FINAL reserve instead of the campaign
    /// region — one verdict per external claim.
    pub final_validation: bool,
}

impl Default for Spec {
    fn default() -> Self {
        let origin = Cc2Weights::attack_tuned().board_params();
        Self {
            campaign: "scratch".to_string(),
            cand_params: origin,
            incumbent_params: origin,
            cell_matches: 800,
            p1: 0.55,
            alpha: 0.05,
            rain_period: 8,
            max_plies: 240,
            beam: Beam::default(),
            final_validation: false,
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

/// Default wall-clock budget shared by all cells (`--budget-secs` overrides).
const DEFAULT_BUDGET_SECS: u64 = 3600;

pub fn run(spec: &Spec, rt: &Runtime, ledger: &mut RunLedger) -> std::io::Result<()> {
    let campaign = Campaign::derive(&spec.campaign);
    let origin = Cc2Weights::attack_tuned().board_params();
    let Beam { width, depth } = spec.beam;
    let budget = rt.budget(DEFAULT_BUDGET_SECS);

    let stride = 4096usize.max(spec.cell_matches as usize);
    let opponents = [Opponent::Greedy, Opponent::Origin, Opponent::Incumbent];
    let rains = [0u32, spec.rain_period];
    let cells = opponents.len() * rains.len();
    // The campaign's promotion sub-region, or — exactly once per external
    // claim — the never-iterated FINAL reserve.
    let seed_base = if spec.final_validation {
        eprintln!(
            "==== FINAL VALIDATION: spending the never-iterated region. One verdict \
             per claim; do not tune after this. ===="
        );
        regions::FINAL
    } else {
        campaign.promote(cells * stride)
    };

    let cc2 = |params: &BoardParams| {
        BotSpec::beam(width, depth)
            .cc2(Cc2Weights::attack_tuned().with_board_params(params))
            .factory()
    };
    let cand = cc2(&spec.cand_params);
    let opponent_bot = |o: Opponent| -> Box<dyn Fn(u64) -> Box<dyn PlayerController> + Sync> {
        match o {
            Opponent::Greedy => Box::new(BotSpec::greedy().factory()),
            Opponent::Origin => Box::new(cc2(&origin)),
            Opponent::Incumbent => Box::new(cc2(&spec.incumbent_params)),
        }
    };

    eprintln!(
        "Promotion panel — campaign '{}' (slot {}) | candidate vs {{greedy, origin, incumbent}} \
         x rain {{0, {}}} | {} matches/cell | budget {}s",
        campaign.id,
        campaign.slot,
        spec.rain_period,
        spec.cell_matches,
        budget.as_secs()
    );

    let start = Instant::now();
    let deadline = start + budget;
    let mut all_pass = true;
    for (i, (&opponent, &rain_period)) in opponents
        .iter()
        .flat_map(|o| rains.iter().map(move |r| (o, r)))
        .enumerate()
    {
        let format = VersusFormat {
            max_plies: spec.max_plies,
            rain_period,
        };
        let report = sprt_race(
            &cand,
            &*opponent_bot(opponent),
            format,
            SprtConfig {
                p1: spec.p1,
                alpha: spec.alpha,
                seed_base: seed_base + i * stride,
                max_matches: spec.cell_matches,
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
        ledger.append_outcome(&json!({
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
        }))?;
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

    ledger.write_summary(json!({
        "exit_reason": if start.elapsed() >= Duration::from_secs(budget.as_secs()) {
            "time_budget"
        } else {
            "complete"
        },
        "verdict": verdict,
        "final_validation": spec.final_validation,
    }))?;
    Ok(())
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
