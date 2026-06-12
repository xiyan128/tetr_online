//! The **downstack (cheese)** benchmark: clearing seeded garbage rows
//! efficiently tests digging / board-reading — the skill that separates elite
//! versus bots — and, unlike empty-board APP, is NOT gameable by combo-farming.
//! Fewer pieces to clear the cheese = stronger.

use rayon::prelude::*;
use tetr_core::engine::{CellKind, Engine, EngineConfig, EngineEvent, PieceType};
use tetr_core::player::{PlayerController, drive_engine};

use crate::accounting::{controller_seed, fold_combo};
use crate::rng::SplitMix64;

/// Garbage-hole column per row for a seeded cheese board (independent per row =
/// maximum messiness). Both bots face the identical cheese for a given seed.
pub fn cheese_holes(seed: u64, rows: usize) -> Vec<usize> {
    let mut rng = SplitMix64::new(seed);
    (0..rows).map(|_| (rng.next_u64() % 10) as usize).collect()
}

/// One cheese-clear game's result.
#[derive(Debug, Clone, Copy, serde::Serialize)]
pub struct DownstackOutcome {
    pub seed: u64,
    pub garbage_rows: u32,
    /// Pieces placed until the cheese was cleared (or the cap / top-out hit).
    pub pieces: u32,
    /// Cleared `garbage_rows` lines without topping out.
    pub cleared: bool,
    pub topped_out: bool,
    /// Attack sent while digging (guideline table) — the OFFENSE proxy: a bot that
    /// clears garbage as Tetrises / T-spins / B2B counter-attacks; one that clears
    /// sloppily sends ~0. Higher = better, given a comparable pieces-to-clear.
    pub attack: u32,
}

/// Play one cheese-clear game: start with `garbage_rows` seeded garbage rows and
/// measure how many pieces the bot needs to clear that many lines (fewer = better).
pub fn play_downstack(
    make_bot: &dyn Fn(u64) -> Box<dyn PlayerController>,
    seed: u64,
    garbage_rows: u32,
    max_pieces: u32,
) -> DownstackOutcome {
    let mut engine = Engine::new(EngineConfig::default(), seed);
    // Paint the cheese: each row full except its hole column. (`set_cell` is the
    // engine's board-setup seam; the piece colour is irrelevant to line clears.)
    for (y, &hole) in cheese_holes(seed, garbage_rows as usize).iter().enumerate() {
        for x in 0..10isize {
            if x as usize != hole {
                engine.set_cell(x, y as isize, CellKind::Some(PieceType::I));
            }
        }
    }

    let mut bot = make_bot(controller_seed(seed));
    let mut pieces = 0u32;
    let mut frames = 0u32;
    let mut topped = false;
    let mut combo = 0u32;
    let mut total_attack = 0u32;
    let max_frames = max_pieces.saturating_mul(64).max(10_000);

    while frames < max_frames {
        frames += 1;
        let mut locked = false;
        for event in drive_engine(&mut engine, &mut *bot) {
            if let Some(clear) = fold_combo(&event, &engine, &mut combo) {
                total_attack += clear.attack;
            }
            match &event {
                EngineEvent::Locked { .. } => {
                    pieces += 1;
                    locked = true;
                }
                EngineEvent::GameOver { .. } => topped = true,
                _ => {}
            }
        }
        if topped {
            break;
        }
        // Lines only rise at a lock; the cheese is cleared once we've cleared that many.
        if locked && engine.snapshot().lines as u32 >= garbage_rows {
            break;
        }
        if pieces >= max_pieces {
            break;
        }
    }

    let cleared = engine.snapshot().lines as u32 >= garbage_rows && !topped;
    DownstackOutcome {
        seed,
        garbage_rows,
        pieces,
        cleared,
        topped_out: topped,
        attack: total_attack,
    }
}

/// Aggregate downstack stats over a seed set.
#[derive(Debug, Clone)]
pub struct DownstackStats {
    pub games: usize,
    /// Per-game cap used to censor failed clears. It is part of the metric definition.
    pub max_pieces: u32,
    /// Optimization-safe mean: cleared games contribute their piece count and failed
    /// games contribute [`max_pieces`](Self::max_pieces). Lower is better. This scalar
    /// is only interpretable alongside the recorded cap.
    pub mean_pieces_censored: f32,
    /// Descriptive mean pieces over games that cleared; selective failure can game it.
    pub mean_pieces_to_clear: f32,
    /// Mean attack sent while clearing, over games that cleared it (OFFENSE proxy).
    pub mean_attack: f32,
    pub clear_rate: f32,
    pub outcomes: Vec<DownstackOutcome>,
}

impl DownstackStats {
    /// Aggregate a set of already-played outcomes under a fixed censoring cap.
    pub fn from_outcomes(outcomes: Vec<DownstackOutcome>, max_pieces: u32) -> Self {
        let cleared: Vec<&DownstackOutcome> = outcomes.iter().filter(|o| o.cleared).collect();
        let (mean_pieces_to_clear, mean_attack) = if cleared.is_empty() {
            (0.0, 0.0)
        } else {
            let n = cleared.len() as f32;
            (
                cleared.iter().map(|o| o.pieces as f32).sum::<f32>() / n,
                cleared.iter().map(|o| o.attack as f32).sum::<f32>() / n,
            )
        };
        let games = outcomes.len();
        let n = games.max(1) as f32;
        let mean_pieces_censored = outcomes
            .iter()
            .map(|outcome| {
                (if outcome.cleared {
                    outcome.pieces
                } else {
                    max_pieces
                }) as f32
            })
            .sum::<f32>()
            / n;

        Self {
            games,
            max_pieces,
            mean_pieces_censored,
            mean_pieces_to_clear,
            mean_attack,
            clear_rate: cleared.len() as f32 / n,
            outcomes,
        }
    }
}

/// Evaluate a bot's cheese-clear efficiency over `seeds`.
pub fn evaluate_downstack(
    make_bot: &(dyn Fn(u64) -> Box<dyn PlayerController> + Sync),
    seeds: &[u64],
    garbage_rows: u32,
    max_pieces: u32,
) -> DownstackStats {
    // Order-stable parallel games; bit-identical to sequential (see versus).
    let outcomes: Vec<DownstackOutcome> = seeds
        .par_iter()
        .map(|&seed| play_downstack(make_bot, seed, garbage_rows, max_pieces))
        .collect();
    DownstackStats::from_outcomes(outcomes, max_pieces)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bots::BotSpec;

    /// The parallel suite must be bit-identical to sequential play of the
    /// same seeds — the versus gate's downstack counterpart.
    #[test]
    fn parallel_evaluation_matches_sequential() {
        let make = BotSpec::greedy().factory();
        let seeds = crate::seeds::seed_set(6);
        let parallel = evaluate_downstack(&make, &seeds, 4, 25);
        let sequential: Vec<DownstackOutcome> = seeds
            .iter()
            .map(|&s| play_downstack(&make, s, 4, 25))
            .collect();
        for (p, s) in parallel.outcomes.iter().zip(&sequential) {
            assert_eq!(
                (p.seed, p.pieces, p.cleared, p.topped_out, p.attack),
                (s.seed, s.pieces, s.cleared, s.topped_out, s.attack),
            );
        }
    }

    fn outcome(seed: u64, pieces: u32, cleared: bool) -> DownstackOutcome {
        DownstackOutcome {
            seed,
            garbage_rows: 9,
            pieces,
            cleared,
            topped_out: !cleared,
            attack: 0,
        }
    }

    #[test]
    fn all_failures_are_censored_at_the_cap() {
        let stats =
            DownstackStats::from_outcomes(vec![outcome(1, 3, false), outcome(2, 80, false)], 100);
        assert_eq!(stats.mean_pieces_censored, 100.0);
        assert_eq!(stats.clear_rate, 0.0);
    }

    #[test]
    fn selective_failure_cannot_improve_the_censored_metric() {
        let a = DownstackStats::from_outcomes(
            vec![
                outcome(1, 20, true),
                outcome(2, 22, true),
                outcome(3, 24, true),
                outcome(4, 30, true),
            ],
            100,
        );
        let b = DownstackStats::from_outcomes(
            vec![
                outcome(1, 20, true),
                outcome(2, 22, true),
                outcome(3, 24, true),
                outcome(4, 30, false),
            ],
            100,
        );

        assert_eq!(a.mean_pieces_to_clear, 24.0);
        assert_eq!(b.mean_pieces_to_clear, 22.0);
        assert_eq!(a.mean_pieces_censored, 24.0);
        assert_eq!(b.mean_pieces_censored, 41.5);
        assert!(a.mean_pieces_censored < b.mean_pieces_censored);
    }

    #[test]
    fn one_piece_cap_real_game_is_censored() {
        let make = BotSpec::greedy().factory();
        let stats = evaluate_downstack(&make, &[0], 9, 1);
        assert_eq!(stats.mean_pieces_censored, 1.0);
        assert_eq!(stats.clear_rate, 0.0);
    }
}
