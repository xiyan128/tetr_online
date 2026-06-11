//! Wald's SPRT over death-decisive paired versus matches — one racer, two
//! callers: the standalone `versus_sprt` bin (long verdicts on recorded
//! candidates) and `versus_climb`'s per-accept confirmer (screen proposals
//! with cheap blocks, spend real matches only on the ones that pass).
//!
//! The unit of evidence is a match where exactly one side topped out:
//! candidate survived ⇒ win. Cap-game tiebreaks and double deaths carry no
//! survival evidence and are excluded from the test (the net-attack tiebreak
//! is structurally anti-defensive — the `garbage_ab` record); they are still
//! counted for context. Blocks draw fresh disjoint seeds and are played
//! arm-swapped, so seed luck and chair order cancel.
//!
//!   H0: p = 0.5   H1: p = `p1`
//!   accept H1 when LLR ≥ ln((1−β)/α); accept H0 when LLR ≤ ln(β/(1−α))
//!
//! [`SprtState`] is the pure accumulator (unit-tested without playing a single
//! match); [`sprt_race`] is the driver that feeds it real matches under a
//! match cap and an optional wall-clock deadline, reporting an honest
//! `Inconclusive` when neither bound is hit in budget.

use std::time::Instant;

use tetr_core::player::PlayerController;

use rayon::prelude::*;

use crate::seeds::seed_set_from;
use crate::versus::{VersusFormat, VersusOutcome, play_versus_format};

/// Test design: hypotheses, error rates, block shape, and budgets.
#[derive(Clone, Copy, Debug)]
pub struct SprtConfig {
    /// The H1 win probability (H0 is always 0.5).
    pub p1: f64,
    /// Type-I error bound (accepting H1 when H0 is true).
    pub alpha: f64,
    /// Type-II error bound (accepting H0 when H1 is true).
    pub beta: f64,
    /// Seeds per block; each block plays `2 × block_seeds` matches (arm swap)
    /// as ONE fused parallel batch, so the block size is also the parallelism
    /// width — the default (24 ⇒ 48 matches) saturates a desktop pool. Bigger
    /// blocks overshoot a crossed bound by more matches; smaller ones starve
    /// the pool. (The original run record used the pre-parallel default, 8.)
    pub block_seeds: usize,
    /// First seed of the race's region. Callers own disjointness (the climb
    /// derives a fresh region per confirmation; the bin uses 16384+).
    pub seed_base: usize,
    /// Hard cap on matches played; hitting it reports `Inconclusive`.
    pub max_matches: u32,
    /// Optional wall-clock bound; crossing it reports `Inconclusive`. NOTE:
    /// a deadline couples the *stopping point* (not any match result) to
    /// machine speed and core count — two hosts can land different
    /// `Inconclusive` cuts of the same deterministic evidence stream. Bounds
    /// crossed before the deadline are machine-independent verdicts.
    pub deadline: Option<Instant>,
    /// Per-block progress lines on stderr.
    pub verbose: bool,
}

impl Default for SprtConfig {
    fn default() -> Self {
        Self {
            p1: 0.55,
            alpha: 0.05,
            beta: 0.05,
            block_seeds: 24,
            seed_base: crate::seeds::regions::SPRT,
            max_matches: 2000,
            deadline: None,
            verbose: false,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SprtVerdict {
    /// Ship-grade evidence the candidate survives more than the incumbent.
    H1Accepted,
    /// No survival edge at the tested effect size.
    H0Accepted,
    /// Budget exhausted between the bounds — the effect, if any, is small.
    Inconclusive,
}

/// The race's outcome plus the evidence it rests on.
#[derive(Clone, Copy, Debug)]
pub struct SprtReport {
    pub verdict: SprtVerdict,
    pub wins: u32,
    pub losses: u32,
    /// Double deaths and cap-tiebreak games (no survival evidence).
    pub ties: u32,
    pub matches: u32,
    pub llr: f64,
    /// Mean candidate net-attack margin over all matches (context only).
    pub mean_margin: f64,
}

/// The pure SPRT accumulator: log-likelihood ratio plus counts. Feeding it
/// outcomes and reading [`verdict`](Self::verdict) is the entire test; the
/// driver around it only supplies matches.
pub struct SprtState {
    llr: f64,
    wins: u32,
    losses: u32,
    ties: u32,
    upper: f64,
    lower: f64,
    win_llr: f64,
    loss_llr: f64,
}

impl SprtState {
    pub fn new(p1: f64, alpha: f64, beta: f64) -> Self {
        Self {
            llr: 0.0,
            wins: 0,
            losses: 0,
            ties: 0,
            upper: ((1.0 - beta) / alpha).ln(),
            lower: (beta / (1.0 - alpha)).ln(),
            win_llr: (p1 / 0.5).ln(),
            loss_llr: ((1.0 - p1) / 0.5).ln(),
        }
    }

    /// Record one match from the candidate's perspective.
    pub fn record(&mut self, cand_topped: bool, opp_topped: bool) {
        match (cand_topped, opp_topped) {
            (false, true) => {
                self.wins += 1;
                self.llr += self.win_llr;
            }
            (true, false) => {
                self.losses += 1;
                self.llr += self.loss_llr;
            }
            // Double death or neither (cap): no survival evidence.
            _ => self.ties += 1,
        }
    }

    /// The test's decision so far: `None` means keep sampling.
    pub fn verdict(&self) -> Option<SprtVerdict> {
        if self.llr >= self.upper {
            Some(SprtVerdict::H1Accepted)
        } else if self.llr <= self.lower {
            Some(SprtVerdict::H0Accepted)
        } else {
            None
        }
    }

    pub fn llr(&self) -> f64 {
        self.llr
    }

    pub fn counts(&self) -> (u32, u32, u32) {
        (self.wins, self.losses, self.ties)
    }

    /// The decision bounds `(lower, upper)` (for reporting).
    pub fn bounds(&self) -> (f64, f64) {
        (self.lower, self.upper)
    }
}

/// Race `cand` against `incumbent` until a bound or a budget ends the test.
pub fn sprt_race(
    cand: &(dyn Fn(u64) -> Box<dyn PlayerController> + Sync),
    incumbent: &(dyn Fn(u64) -> Box<dyn PlayerController> + Sync),
    format: VersusFormat,
    config: SprtConfig,
) -> SprtReport {
    let mut state = SprtState::new(config.p1, config.alpha, config.beta);
    let (mut matches, mut margin_sum) = (0u32, 0.0f64);
    let mut block = 0usize;

    let verdict = loop {
        if let Some(verdict) = state.verdict() {
            break verdict;
        }
        if matches >= config.max_matches {
            break SprtVerdict::Inconclusive;
        }
        if config.deadline.is_some_and(|d| Instant::now() >= d) {
            break SprtVerdict::Inconclusive;
        }

        let seeds = seed_set_from(
            config.seed_base + block * config.block_seeds,
            config.block_seeds,
        );
        block += 1;

        // Arm-swapped, one FUSED parallel batch: both orientations of every
        // seed go wide together (2 × block_seeds matches, one barrier per
        // block — two half-width halves with a join between them starved the
        // pool, the review's finding). The LLR after a full block is a sum,
        // so intra-block ordering cannot change any verdict.
        let jobs: Vec<(bool, u64)> = seeds
            .iter()
            .flat_map(|&s| [(false, s), (true, s)])
            .collect();
        let outcomes: Vec<(bool, VersusOutcome)> = jobs
            .par_iter()
            .map(|&(swapped, seed)| {
                let o = if swapped {
                    play_versus_format(incumbent, cand, seed, format)
                } else {
                    play_versus_format(cand, incumbent, seed, format)
                };
                (swapped, o)
            })
            .collect();
        for (swapped, o) in &outcomes {
            let (cand_topped, opp_topped, margin) = if *swapped {
                (
                    o.b_topped,
                    o.a_topped,
                    f64::from(o.attack_b) - f64::from(o.attack_a),
                )
            } else {
                (
                    o.a_topped,
                    o.b_topped,
                    f64::from(o.attack_a) - f64::from(o.attack_b),
                )
            };
            state.record(cand_topped, opp_topped);
            margin_sum += margin;
            matches += 1;
        }

        if config.verbose {
            let (wins, losses, ties) = state.counts();
            eprintln!(
                "  sprt block {block:>3} | decisive {wins}-{losses} (ties {ties}) | LLR {:+.3}",
                state.llr()
            );
        }
    };

    let (wins, losses, ties) = state.counts();
    SprtReport {
        verdict,
        wins,
        losses,
        ties,
        matches,
        llr: state.llr(),
        mean_margin: margin_sum / f64::from(matches.max(1)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_winning_streak_accepts_h1_at_walds_bound() {
        let mut state = SprtState::new(0.55, 0.05, 0.05);
        // ln(0.95/0.05) / ln(1.1) ≈ 30.9 ⇒ the 31st straight win crosses.
        let mut n = 0;
        while state.verdict().is_none() {
            state.record(false, true);
            n += 1;
            assert!(n <= 40, "a pure win streak must cross the upper bound");
        }
        assert_eq!(state.verdict(), Some(SprtVerdict::H1Accepted));
        assert_eq!(n, 31);
    }

    #[test]
    fn a_losing_streak_accepts_h0_symmetrically() {
        let mut state = SprtState::new(0.55, 0.05, 0.05);
        let mut n = 0;
        while state.verdict().is_none() {
            state.record(true, false);
            n += 1;
            assert!(n <= 40);
        }
        assert_eq!(state.verdict(), Some(SprtVerdict::H0Accepted));
        // Loss evidence is weaker per trial (|ln 0.9| < ln 1.1), so H0 takes
        // a few more straight losses than H1 takes straight wins.
        assert_eq!(n, 28);
    }

    #[test]
    fn ties_carry_no_evidence() {
        let mut state = SprtState::new(0.55, 0.05, 0.05);
        state.record(true, true); // double death
        state.record(false, false); // cap
        assert_eq!(state.llr(), 0.0);
        assert_eq!(state.counts(), (0, 0, 2));
        assert_eq!(state.verdict(), None);
    }

    #[test]
    fn even_evidence_stays_between_the_bounds() {
        let mut state = SprtState::new(0.55, 0.05, 0.05);
        for _ in 0..250 {
            state.record(false, true);
            state.record(true, false);
        }
        // 250-250: drifts slightly negative (loss evidence is weaker but the
        // pairs sum to ln(1.1) + ln(0.9) = ln(0.99) < 0), far from any bound.
        assert_eq!(state.verdict(), None);
        let (lower, upper) = state.bounds();
        assert!(state.llr() > lower && state.llr() < upper);
    }
}
