//! Aggregating many games into a reliable measurement.
//!
//! One game is luck — the seven-bag seed dominates a single result. A trustworthy
//! number comes from playing a *set* of seeds and reporting the distribution.
//! [`evaluate`] runs a [`Contender`] across seeds and summarizes each metric with
//! mean / median / spread via [`MetricSummary`].

use std::cmp::Ordering;

use crate::arena::outcome::GameOutcome;
use crate::arena::runner::play;
use crate::arena::{Contender, GameSetup};

/// Summary statistics for one metric across a sample of games.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MetricSummary {
    /// Sample size.
    pub n: usize,
    pub mean: f64,
    pub median: f64,
    /// Population standard deviation (spread; reliability of the mean).
    pub std_dev: f64,
    pub min: f64,
    pub max: f64,
}

impl MetricSummary {
    /// Summarize a sample. An empty sample yields all-zero with `n == 0`.
    pub fn of(values: &[f64]) -> Self {
        if values.is_empty() {
            return Self {
                n: 0,
                mean: 0.0,
                median: 0.0,
                std_dev: 0.0,
                min: 0.0,
                max: 0.0,
            };
        }

        let n = values.len();
        let mean = values.iter().sum::<f64>() / n as f64;
        let variance = values.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / n as f64;

        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
        let median = if n % 2 == 1 {
            sorted[n / 2]
        } else {
            (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
        };

        Self {
            n,
            mean,
            median,
            std_dev: variance.sqrt(),
            min: sorted[0],
            max: sorted[n - 1],
        }
    }
}

/// The reliable measurement of one contender on one setup: every game's outcome,
/// plus per-metric summary statistics over the seed set.
///
/// This measures a *single* implementation. Comparing and ranking many
/// implementations is a layer built on top of this one.
#[derive(Clone, Debug)]
pub struct Evaluation {
    pub contender: String,
    pub setup: String,
    /// Every game played, in seed order — kept for drill-down.
    pub games: Vec<GameOutcome>,
    pub pieces: MetricSummary,
    pub lines: MetricSummary,
    pub lines_per_piece: MetricSummary,
    pub tetris_rate: MetricSummary,
    /// Fraction of games that ended by topping out (0.0..=1.0).
    pub topout_rate: f64,
}

/// Evaluate `contender` on `setup` across `seeds`.
///
/// Deterministic and order-independent — each game is independent and fully
/// determined by its seed, so the result is reproducible regardless of seed order
/// or how the games are scheduled. The seed-set size is the reliability knob:
/// more seeds, tighter spread.
pub fn evaluate(contender: &Contender, setup: &GameSetup, seeds: &[u64]) -> Evaluation {
    let games: Vec<GameOutcome> = seeds.iter().map(|&seed| play(contender, setup, seed)).collect();

    let metric = |f: fn(&GameOutcome) -> f64| {
        MetricSummary::of(&games.iter().map(f).collect::<Vec<_>>())
    };

    let topout_rate = if games.is_empty() {
        0.0
    } else {
        games.iter().filter(|g| g.topped_out()).count() as f64 / games.len() as f64
    };

    Evaluation {
        contender: contender.name().to_owned(),
        setup: setup.name().to_owned(),
        pieces: metric(|g| f64::from(g.pieces_placed)),
        lines: metric(|g| f64::from(g.lines_cleared)),
        lines_per_piece: metric(GameOutcome::lines_per_piece),
        tetris_rate: metric(GameOutcome::tetris_rate),
        topout_rate,
        games,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn summary_on_a_known_sample() {
        // Textbook sample with population std dev exactly 2.0.
        let values = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let s = MetricSummary::of(&values);

        assert_eq!(s.n, 8);
        assert!((s.mean - 5.0).abs() < 1e-9);
        assert!((s.std_dev - 2.0).abs() < 1e-9);
        assert!((s.median - 4.5).abs() < 1e-9); // (4 + 5) / 2
        assert!((s.min - 2.0).abs() < 1e-9);
        assert!((s.max - 9.0).abs() < 1e-9);
    }

    #[test]
    fn summary_odd_count_median() {
        let s = MetricSummary::of(&[3.0, 1.0, 2.0]);
        assert!((s.median - 2.0).abs() < 1e-9);
    }

    #[test]
    fn empty_summary_is_zeroed() {
        let s = MetricSummary::of(&[]);
        assert_eq!(s.n, 0);
        assert_eq!(s.mean, 0.0);
        assert_eq!(s.std_dev, 0.0);
    }
}
