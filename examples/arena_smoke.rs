//! Demo of the play-evaluation harness: measure two bot weight-profiles and print
//! play-quality statistics.
//!
//! ```sh
//! cargo run --release --example arena_smoke --features arena
//! ```
//!
//! Compares the two Tier-1 weight profiles at flawless difficulty, isolating the
//! profile's effect: SURVIVAL (the shipped default — reward every clear) vs
//! DOWNSTACK (Cold Clear's weights — penalize small clears to hold out for
//! Tetrises). With a 1-ply greedy the latter can't plan the setups it waits for,
//! which the numbers make visible.

use core::time::Duration;
use tetr_online::ai::{
    AiController, GreedyPlanner, LinearEvaluator, SearchBudget, SearchPolicy, Weights,
};
use tetr_online::arena::{evaluate, seed_set, Contender, Evaluation, GameSetup};

fn main() {
    let setup = GameSetup::standard("standard", 100);
    let seeds = seed_set(20);

    let contenders = [
        weighted("survival (default)", Weights::SURVIVAL),
        weighted("downstack", Weights::DOWNSTACK),
    ];

    println!(
        "setup: {} · {} pieces/game · {} seeds · flawless difficulty\n",
        setup.name(),
        setup.max_pieces(),
        seeds.len(),
    );
    println!(
        "{:<19} {:>13} {:>13} {:>7} {:>9} {:>9}",
        "weights", "pieces", "lines", "lpp", "tetris%", "topout%",
    );
    println!("{}", "-".repeat(74));
    for contender in &contenders {
        print_row(&evaluate(contender, &setup, &seeds));
    }
}

/// A flawless greedy contender using a specific board/reward weight profile.
///
/// Builds a [`SearchPolicy`] (the brain) directly — greedy planner + the chosen
/// weights, zero imperfection to isolate the profile — and hands it to the
/// model-agnostic controller shell with no reaction delay.
fn weighted(name: &'static str, weights: Weights) -> Contender {
    Contender::new(name, move |seed| {
        let policy = SearchPolicy::new(
            Box::new(GreedyPlanner::new()),
            Box::new(LinearEvaluator::new(weights)),
            SearchBudget::greedy(),
            0.0, // flawless: isolate the weight profile from the imperfection handicap
            seed,
        );
        Box::new(AiController::with_policy(Box::new(policy), Duration::ZERO))
    })
}

fn print_row(e: &Evaluation) {
    println!(
        "{:<19} {:>5.0}±{:<5.0} {:>5.1}±{:<5.1} {:>7.3} {:>8.1}% {:>8.1}%",
        e.contender,
        e.pieces.mean,
        e.pieces.std_dev,
        e.lines.mean,
        e.lines.std_dev,
        e.lines_per_piece.mean,
        e.tetris_rate.mean * 100.0,
        e.topout_rate * 100.0,
    );
}
