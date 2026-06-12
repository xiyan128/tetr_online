//! Pair-level GSPRT over death-decisive versus matches — the `race` eval's
//! engine, and the confirmation primitive any future search loop wraps.
//!
//! # Why the unit of evidence is the seed PAIR
//!
//! Every block plays each seed from both chairs (arm swap on common random
//! numbers). The two games of a seed share its piece stream and rain
//! schedule, so their outcomes are correlated — and feeding correlated games
//! to Wald's SPRT as independent Bernoulli draws voids the nominal α/β
//! bounds in whichever direction the correlation points (positive
//! within-pair correlation overdisperses the evidence stream: the walk
//! crosses bounds it shouldn't, inflating false accepts). The observation
//! here is therefore one seed's chair-swapped double game, scored
//! `t = wins − losses ∈ {−2..+2}` from the candidate's perspective, and the
//! test is a generalized SPRT (GSPRT, the fishtest design): the likelihood
//! ratio under a normal model with mean and variance ESTIMATED from the pair
//! scores themselves,
//!
//! ```text
//!   LLR ≈ (n/2)·ln(σ̂₀² / σ̂₁²),    σ̂ᵢ² = s² + (t̄ − μᵢ)²
//!   μ₀ = 0,    μ₁ = (2·p1 − 1) · d̄
//! ```
//!
//! where `s²` is the MLE variance of the pair scores and `d̄` the observed
//! decisive games per pair — so an all-ties stream starves the test toward
//! `Inconclusive` instead of biasing it. The empirical variance is what buys
//! the error bounds back under correlation: coupled pairs widen `s²` and the
//! test slows down rather than lying. `correlated_null_respects_alpha…` is
//! the receipt — measured over 1500 deterministic trials at nominal α = 5%:
//! a null with 60% coupled pairs false-accepts at **4.5%** under this test
//! and **13.7%** under the per-game trinomial walk it replaced (kept as a
//! cross-check field on the report); under a fully independent null the two
//! read 5.6% and 3.3%. Correlation is what flips the old test from
//! conservative to broken, and it is exactly what arm-swapped CRN pairs
//! produce.
//!
//! Ties (double death, double cap-survival) carry no survival evidence: the
//! cap-game net-attack tiebreak is structurally anti-defensive (the
//! `garbage_ab` record), so survival verdicts must never lean on it. A tie
//! contributes a decisive-count of zero, shrinking `μ₁`, never `t`.
//!
//! [`SprtState`] is the pure accumulator (unit-tested and simulation-tested
//! without playing a single match); [`sprt_race`] is the driver that feeds
//! it real paired matches under a match cap and an optional wall-clock
//! deadline, reporting an honest `Inconclusive` when neither bound is hit in
//! budget.

use std::time::Instant;

use tetr_core::player::PlayerController;

use rayon::prelude::*;

use crate::seeds::seed_set_from;
use crate::versus::{VersusFormat, VersusOutcome, play_versus_format};

/// Test design: hypotheses, error rates, block shape, and budgets.
#[derive(Clone, Copy, Debug)]
pub struct SprtConfig {
    /// The H1 per-decisive-game win probability (H0 is always 0.5); the pair
    /// model tests the implied mean pair score `μ₁ = (2·p1 − 1)·d̄`.
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
    /// derives fresh campaign regions per confirmation; the bin uses 16384+).
    pub seed_base: usize,
    /// Hard cap on matches played; hitting it reports `Inconclusive`.
    pub max_matches: u32,
    /// Pairs required before any verdict — the normal approximation behind
    /// the GSPRT needs a few observations before its variance estimate means
    /// anything. Streak crossings sit well past the default anyway.
    pub min_pairs: u32,
    /// Optional wall-clock bound; crossing it reports `Inconclusive`. NOTE:
    /// a deadline couples the *stopping point* (not any match result) to
    /// machine speed and core count — two hosts can land different
    /// `Inconclusive` cuts of the same deterministic evidence stream. Bounds
    /// crossed before the deadline are machine-independent verdicts.
    pub deadline: Option<Instant>,
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
            min_pairs: 8,
            deadline: None,
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
    /// Chair-swapped seed pairs observed (= matches / 2).
    pub pairs: u32,
    /// Pair-score histogram over `t = wins − losses ∈ {−2, −1, 0, +1, +2}`.
    pub pair_counts: [u32; 5],
    /// The deciding pair-GSPRT log-likelihood ratio.
    pub llr: f64,
    /// The legacy per-game Bernoulli walk — cross-check only, decides
    /// nothing; reported next to `llr` to show what an independence model
    /// would have concluded on the same stream.
    pub trinomial_llr: f64,
    /// Within-pair outcome correlation estimate (None until enough pairs);
    /// positive means a per-game independence model understates the stream's
    /// variance, i.e. the trinomial cross-check is anticonservative here.
    pub pair_correlation: Option<f64>,
    /// Mean candidate net-attack margin over all matches (context only).
    pub mean_margin: f64,
}

/// The pure sequential-test accumulator: order-free sufficient statistics
/// over pair scores plus the verdict bounds. Feeding it pairs and reading
/// [`verdict`](Self::verdict) is the entire test; the driver around it only
/// supplies matches.
pub struct SprtState {
    p1: f64,
    upper: f64,
    lower: f64,
    min_pairs: u32,
    pairs: u32,
    sum_t: f64,
    sum_t2: f64,
    sum_d: f64,
    pair_counts: [u32; 5],
    wins: u32,
    losses: u32,
    ties: u32,
    trinomial_llr: f64,
    win_llr: f64,
    loss_llr: f64,
}

impl SprtState {
    pub fn new(p1: f64, alpha: f64, beta: f64, min_pairs: u32) -> Self {
        Self {
            p1,
            upper: ((1.0 - beta) / alpha).ln(),
            lower: (beta / (1.0 - alpha)).ln(),
            min_pairs,
            pairs: 0,
            sum_t: 0.0,
            sum_t2: 0.0,
            sum_d: 0.0,
            pair_counts: [0; 5],
            wins: 0,
            losses: 0,
            ties: 0,
            trinomial_llr: 0.0,
            win_llr: (p1 / 0.5).ln(),
            loss_llr: ((1.0 - p1) / 0.5).ln(),
        }
    }

    /// Record one seed's chair-swapped double game from the candidate's
    /// perspective; `wins + losses ≤ 2`, the remainder were ties.
    pub fn record_pair(&mut self, wins: u32, losses: u32) {
        assert!(wins + losses <= 2, "a seed pair is exactly two games");
        let t = wins as i32 - losses as i32;
        self.pairs += 1;
        self.sum_t += f64::from(t);
        self.sum_t2 += f64::from(t * t);
        self.sum_d += f64::from(wins + losses);
        self.pair_counts[(t + 2) as usize] += 1;
        self.wins += wins;
        self.losses += losses;
        self.ties += 2 - wins - losses;
        self.trinomial_llr += f64::from(wins) * self.win_llr + f64::from(losses) * self.loss_llr;
    }

    /// The pair-GSPRT log-likelihood ratio from the sufficient statistics
    /// (recomputed in O(1); the order of pairs cannot matter).
    pub fn llr(&self) -> f64 {
        if self.pairs < 2 {
            return 0.0;
        }
        let n = f64::from(self.pairs);
        let mean = self.sum_t / n;
        let var = (self.sum_t2 / n - mean * mean).max(0.0);
        let mu1 = (2.0 * self.p1 - 1.0) * (self.sum_d / n);
        // σ̂ᵢ² = s² + (t̄ − μᵢ)² is each hypothesis's variance MLE; the ridge
        // only guards the all-ties stream (both σ̂² zero ⇒ LLR 0, not NaN).
        let s0 = (var + mean * mean).max(1e-9);
        let s1 = (var + (mean - mu1) * (mean - mu1)).max(1e-9);
        0.5 * n * (s0 / s1).ln()
    }

    /// The test's decision so far: `None` means keep sampling.
    pub fn verdict(&self) -> Option<SprtVerdict> {
        if self.pairs < self.min_pairs {
            return None;
        }
        let llr = self.llr();
        if llr >= self.upper {
            Some(SprtVerdict::H1Accepted)
        } else if llr <= self.lower {
            Some(SprtVerdict::H0Accepted)
        } else {
            None
        }
    }

    /// The per-game Bernoulli walk the pair test replaced — cross-check only.
    pub fn trinomial_llr(&self) -> f64 {
        self.trinomial_llr
    }

    /// Within-pair correlation estimate: empirical pair-score variance
    /// against what `d̄` independent ±1 games at the observed win rate would
    /// give. Positive ⇒ the independence model understates variance.
    pub fn pair_correlation(&self) -> Option<f64> {
        if self.pairs < 30 || self.wins + self.losses == 0 {
            return None;
        }
        let n = f64::from(self.pairs);
        let mean = self.sum_t / n;
        let var = (self.sum_t2 / n - mean * mean).max(0.0);
        let p = f64::from(self.wins) / f64::from(self.wins + self.losses);
        let indep = (self.sum_d / n) * (1.0 - (2.0 * p - 1.0) * (2.0 * p - 1.0));
        (indep > 1e-9).then(|| (var - indep) / indep)
    }

    pub fn pairs(&self) -> u32 {
        self.pairs
    }

    pub fn pair_counts(&self) -> [u32; 5] {
        self.pair_counts
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
/// `names` are the bots' registry names, used only for the game events.
pub fn sprt_race(
    names: (&str, &str),
    cand: &(dyn Fn(u64) -> Box<dyn PlayerController> + Sync),
    incumbent: &(dyn Fn(u64) -> Box<dyn PlayerController> + Sync),
    format: VersusFormat,
    config: SprtConfig,
) -> SprtReport {
    let mut state = SprtState::new(config.p1, config.alpha, config.beta, config.min_pairs);
    let (mut matches, mut margin_sum) = (0u32, 0.0f64);
    let mut block = 0usize;
    // Live position between the bounds, stderr-only and auto-hidden off-TTY
    // (the report is the record; the bar is just company for the silence).
    let pb = if config.max_matches == u32::MAX {
        crate::progress::spinner("race")
    } else {
        crate::progress::bar(u64::from(config.max_matches), "race")
    };

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
        // pool, the review's finding). Verdicts are checked on full blocks
        // over order-free sufficient statistics, so intra-block scheduling
        // cannot change any outcome.
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
        // `jobs` lays each seed's two orientations adjacently and the
        // parallel collect preserves order, so chunks of two are exactly the
        // chair-swapped pairs.
        for pair in outcomes.chunks_exact(2) {
            let (mut wins, mut losses) = (0u32, 0u32);
            for (swapped, o) in pair {
                let (a, b) = if *swapped {
                    (names.1, names.0)
                } else {
                    (names.0, names.1)
                };
                crate::events::emit(
                    "game",
                    serde_json::json!({
                        "mode": "versus",
                        "seed": crate::events::seed_hex(o.seed),
                        "a": a,
                        "b": b,
                        "a_topped": o.a_topped,
                        "b_topped": o.b_topped,
                        "a_attack": o.attack_a,
                        "b_attack": o.attack_b,
                        "plies": o.plies,
                    }),
                );
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
                match (cand_topped, opp_topped) {
                    (false, true) => wins += 1,
                    (true, false) => losses += 1,
                    // Double death or neither (cap): no survival evidence.
                    _ => {}
                }
                margin_sum += margin;
                matches += 1;
            }
            state.record_pair(wins, losses);
        }

        let (wins, losses, _) = state.counts();
        let (lower, upper) = state.bounds();
        pb.set_position(u64::from(matches));
        pb.set_message(format!(
            "block {block} | {wins}-{losses} of {} pairs | LLR {:+.2} in [{lower:+.2}, {upper:+.2}]",
            state.pairs(),
            state.llr(),
        ));
    };
    pb.finish_and_clear();

    let (wins, losses, ties) = state.counts();
    SprtReport {
        verdict,
        wins,
        losses,
        ties,
        matches,
        pairs: state.pairs(),
        pair_counts: state.pair_counts(),
        llr: state.llr(),
        trinomial_llr: state.trinomial_llr(),
        pair_correlation: state.pair_correlation(),
        mean_margin: margin_sum / f64::from(matches.max(1)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::SplitMix64;

    /// (n/2)·ln(4/(2−μ₁)²) with μ₁ = 0.2 crosses ln(19) ≈ 2.944 at n = 28:
    /// a pure double-win streak convinces the pair test in 28 pairs.
    #[test]
    fn a_double_win_streak_accepts_h1() {
        let mut state = SprtState::new(0.55, 0.05, 0.05, 8);
        let mut n = 0;
        while state.verdict().is_none() {
            state.record_pair(2, 0);
            n += 1;
            assert!(n <= 40, "a pure WW streak must cross the upper bound");
        }
        assert_eq!(state.verdict(), Some(SprtVerdict::H1Accepted));
        assert_eq!(n, 28);
    }

    /// The loss streak's σ̂₁² sits farther from σ̂₀² than the win streak's
    /// ((2+μ₁)² vs (2−μ₁)²), so H0 needs 31 pairs — the two bounds are not
    /// mirror images.
    #[test]
    fn a_double_loss_streak_accepts_h0() {
        let mut state = SprtState::new(0.55, 0.05, 0.05, 8);
        let mut n = 0;
        while state.verdict().is_none() {
            state.record_pair(0, 2);
            n += 1;
            assert!(n <= 40, "a pure LL streak must cross the lower bound");
        }
        assert_eq!(state.verdict(), Some(SprtVerdict::H0Accepted));
        assert_eq!(n, 31);
    }

    /// Ties carry no evidence in either direction: the test starves rather
    /// than drifts.
    #[test]
    fn an_all_ties_stream_never_decides() {
        let mut state = SprtState::new(0.55, 0.05, 0.05, 8);
        for _ in 0..500 {
            state.record_pair(0, 0);
        }
        assert_eq!(state.llr(), 0.0);
        assert_eq!(state.verdict(), None);
    }

    /// A perfectly split stream (every pair 1–1) is *evidence for H0* — under
    /// H1 a split pair has probability 2·p1·(1−p1) < ½ — and the
    /// zero-variance degenerate form of the GSPRT recognises it as soon as
    /// verdicts open.
    #[test]
    fn an_all_splits_stream_accepts_h0_at_min_pairs() {
        let mut state = SprtState::new(0.55, 0.05, 0.05, 8);
        let mut n = 0;
        while state.verdict().is_none() {
            state.record_pair(1, 1);
            n += 1;
            assert!(n <= 100);
        }
        assert_eq!(state.verdict(), Some(SprtVerdict::H0Accepted));
        assert_eq!(n, 8);
    }

    /// One simulated SPRT under a pair-outcome model: with probability
    /// `coupled` the seed forces a double win or double loss (fair coin —
    /// the strongest positive within-pair correlation a null can have);
    /// otherwise the two games are independent Bernoulli(`p`). Returns the
    /// GSPRT verdict plus whether the trinomial walk crossed the UPPER bound
    /// first — the trinomial is latched at its first boundary crossing,
    /// mirroring how a sequential test actually stops.
    fn simulate_one(
        rng: &mut SplitMix64,
        coupled: f64,
        p: f64,
        max_pairs: u32,
    ) -> (Option<SprtVerdict>, bool) {
        let mut state = SprtState::new(0.55, 0.05, 0.05, 8);
        let (lower, upper) = state.bounds();
        let mut trinomial_h1 = false;
        let mut trinomial_done = false;
        let unit = |r: &mut SplitMix64| (r.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
        for _ in 0..max_pairs {
            let (w, l) = if unit(rng) < coupled {
                if unit(rng) < 0.5 { (2, 0) } else { (0, 2) }
            } else {
                let w = u32::from(unit(rng) < p) + u32::from(unit(rng) < p);
                (w, 2 - w)
            };
            state.record_pair(w, l);
            if !trinomial_done {
                if state.trinomial_llr() >= upper {
                    trinomial_h1 = true;
                    trinomial_done = true;
                } else if state.trinomial_llr() <= lower {
                    trinomial_done = true;
                }
            }
            if let Some(v) = state.verdict() {
                return (Some(v), trinomial_h1);
            }
        }
        (None, trinomial_h1)
    }

    fn false_accept_rates(coupled: f64, trials: u64) -> (f64, f64) {
        let (mut gsprt_h1, mut trinomial_h1) = (0u64, 0u64);
        for trial in 0..trials {
            let mut rng = SplitMix64::new(0xC0FFEE ^ trial);
            let (g, t) = simulate_one(&mut rng, coupled, 0.5, 1000);
            gsprt_h1 += u64::from(g == Some(SprtVerdict::H1Accepted));
            trinomial_h1 += u64::from(t);
        }
        (
            gsprt_h1 as f64 / trials as f64,
            trinomial_h1 as f64 / trials as f64,
        )
    }

    /// THE design claim: under a null with strong positive within-pair
    /// correlation, the pair test holds its nominal α = 0.05 while the
    /// per-game trinomial walk (what this module used to be) violates it on
    /// the very same outcome stream. Deterministic seeds; bounds leave room
    /// for Monte-Carlo noise (1500 trials ⇒ s.e. ≈ 0.006 at p ≈ 0.05).
    #[test]
    fn correlated_null_respects_alpha_where_trinomial_does_not() {
        let (gsprt, trinomial) = false_accept_rates(0.6, 1500);
        eprintln!("coupled null: gsprt {gsprt:.4}, trinomial {trinomial:.4}");
        assert!(
            gsprt <= 0.075,
            "pair GSPRT false-accept rate {gsprt:.3} exceeds the nominal α band"
        );
        assert!(
            trinomial >= 0.08,
            "trinomial cross-check false-accept rate {trinomial:.3} — expected the \
             independence model to break here; did the simulation change?"
        );
    }

    /// The same bound holds where the old test was also (nearly) honest:
    /// fully independent games.
    #[test]
    fn independent_null_respects_alpha() {
        let (gsprt, trinomial) = false_accept_rates(0.0, 1500);
        eprintln!("independent null: gsprt {gsprt:.4}, trinomial {trinomial:.4}");
        assert!(
            gsprt <= 0.075,
            "pair GSPRT false-accept rate {gsprt:.3} under the independent null"
        );
    }

    /// Power sanity: a real p = 0.60 candidate (independent games) is
    /// accepted nearly always, well inside the pair cap.
    #[test]
    fn a_real_edge_is_accepted() {
        let trials = 400u64;
        let mut accepted = 0u64;
        for trial in 0..trials {
            let mut rng = SplitMix64::new(0xBEEF ^ trial);
            let (g, _) = simulate_one(&mut rng, 0.0, 0.60, 1000);
            accepted += u64::from(g == Some(SprtVerdict::H1Accepted));
        }
        let rate = accepted as f64 / trials as f64;
        assert!(
            rate >= 0.85,
            "H1 acceptance rate {rate:.3} for a p=0.60 edge"
        );
    }
}
