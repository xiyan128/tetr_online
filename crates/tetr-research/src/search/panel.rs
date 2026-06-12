//! Promotion panel: the one gate between "my climb accepted it" and "it is
//! the better bot".
//!
//! A climb optimizes against ONE incumbent in ONE format, so its winner may
//! be a one-trick candidate. The panel races the candidate across formats
//! against NAMED opponents from the bot registry, each cell a pair-level
//! GSPRT on fresh campaign seeds. Two bars: `must_beat` opponents demand H1
//! in every format (the floor and the origin — losing either is
//! disqualifying); `must_not_lose_to` opponents demand non-regression (H1,
//! or an in-budget inconclusive while not behind on decisives — a
//! near-equal incumbent is not a regression). H0 anywhere REJECTS, and a
//! budget-starved inconclusive against a `must_beat` opponent also rejects —
//! promotion needs evidence, and "we ran out of time" is not evidence.
//!
//! A spec with `final_validation: true` draws from [`regions::FINAL`] — the
//! never-iterated reserve that backs ONE verdict per external claim.
//! Register it as its own named entry, run it exactly once, when the claim
//! is drafted and nothing will be tuned afterwards.

use std::time::{Duration, Instant};

use crate::bots::{self, Bot};
use crate::seeds::{Campaign, regions};
use crate::sprt::{SprtConfig, SprtVerdict, sprt_race};
use crate::versus::VersusFormat;

#[derive(Clone, Debug, serde::Serialize)]
pub struct Spec {
    /// The campaign whose promotion sub-region judges this candidate.
    pub campaign: String,
    /// Registered bot names the candidate must beat (H1) in every format.
    pub must_beat: Vec<String>,
    /// Registered bot names the candidate must not regress against.
    pub must_not_lose_to: Vec<String>,
    /// Match cap per panel cell.
    pub cell_matches: u32,
    pub p1: f64,
    pub alpha: f64,
    /// The rainy half of the format axis (the other half is rain 0).
    pub rain_period: u32,
    pub max_plies: u32,
    /// Draw from the never-iterated FINAL reserve instead of the campaign
    /// region — one verdict per external claim.
    pub final_validation: bool,
}

impl Default for Spec {
    fn default() -> Self {
        Self {
            campaign: "scratch".to_string(),
            must_beat: vec!["greedy".to_string(), "attack-tuned".to_string()],
            must_not_lose_to: vec!["attack-tuned".to_string()],
            cell_matches: 800,
            p1: 0.55,
            alpha: 0.05,
            rain_period: 8,
            max_plies: 240,
            final_validation: false,
        }
    }
}

/// One cell's bar (see the module docs for the rationale).
fn cell_passes(must_beat: bool, verdict: SprtVerdict, wins: u32, losses: u32) -> bool {
    if must_beat {
        verdict == SprtVerdict::H1Accepted
    } else {
        verdict == SprtVerdict::H1Accepted
            || (verdict == SprtVerdict::Inconclusive && wins >= losses)
    }
}

/// Default wall-clock budget shared by all cells (`--budget-secs` overrides).
const DEFAULT_BUDGET_SECS: u64 = 3600;

pub fn run(spec: &Spec, cand: &Bot, budget: Option<Duration>) -> std::io::Result<()> {
    let campaign = Campaign::derive(&spec.campaign);
    let budget = budget.unwrap_or(Duration::from_secs(DEFAULT_BUDGET_SECS));

    let opponents: Vec<(String, bool)> = spec
        .must_beat
        .iter()
        .map(|n| (n.clone(), true))
        .chain(spec.must_not_lose_to.iter().map(|n| (n.clone(), false)))
        .collect();
    let rains = [0u32, spec.rain_period];
    let cells = opponents.len() * rains.len();
    let stride = 4096usize.max(spec.cell_matches as usize);
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

    eprintln!(
        "Promotion panel — campaign '{}' (slot {}) | candidate {} vs {:?} x rain {{0, {}}} | \
         {} matches/cell | budget {}s",
        campaign.id,
        campaign.slot,
        cand.name,
        opponents
            .iter()
            .map(|(n, _)| n.as_str())
            .collect::<Vec<_>>(),
        spec.rain_period,
        spec.cell_matches,
        budget.as_secs()
    );

    let start = Instant::now();
    let deadline = start + budget;
    let mut all_pass = true;
    for (i, ((name, must_beat), &rain_period)) in opponents
        .iter()
        .flat_map(|o| rains.iter().map(move |r| (o, r)))
        .enumerate()
    {
        let opponent = bots::find(name)
            .unwrap_or_else(|| panic!("panel opponent {name:?} is not a registered bot"));
        let format = VersusFormat {
            max_plies: spec.max_plies,
            rain_period,
        };
        let report = sprt_race(
            &cand.spec.factory(),
            &opponent.spec.factory(),
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
        let pass = cell_passes(*must_beat, report.verdict, report.wins, report.losses);
        all_pass &= pass;
        eprintln!(
            "cell {}/{cells} | vs {name:<18} rain {rain_period} | {} | decisive {}-{} of {} pairs | \
             LLR {:+.2} | {}",
            i + 1,
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
            "panel_cell_{name}_rain{rain_period} {}",
            if pass { "PASS" } else { "FAIL" }
        );
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
    println!("panel_verdict {verdict}");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// must-beat demands proof; must-not-lose-to demands non-regression.
    #[test]
    fn the_bars_match_the_doc_table() {
        use SprtVerdict::*;
        assert!(cell_passes(true, H1Accepted, 10, 0));
        assert!(!cell_passes(true, Inconclusive, 10, 0));
        assert!(!cell_passes(true, H0Accepted, 0, 10));
        assert!(cell_passes(false, H1Accepted, 10, 0));
        assert!(cell_passes(false, Inconclusive, 5, 5));
        assert!(cell_passes(false, Inconclusive, 6, 5));
        assert!(!cell_passes(false, Inconclusive, 4, 5));
        assert!(!cell_passes(false, H0Accepted, 5, 5));
    }
}
