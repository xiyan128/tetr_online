//! Capped-marathon eval: score/sec + APP for one bot — the tight-loop
//! headline metric (`score_per_second` is the marathon proxy,
//! `attack_per_piece` the versus/CC2 metric; both are autoresearch parse
//! contracts). The piece cap keeps iterations fast; score/sec stays a
//! faithful early-game scoring-rate proxy.

use serde_json::json;

use crate::bots::Bot;
use crate::commands::Runtime;
use crate::events;
use crate::marathon::{DEFAULT_MAX_FRAMES, evaluate_capped};
use crate::seeds::seed_set;

#[derive(Clone, Copy, Debug, serde::Serialize)]
pub struct Spec {
    pub seeds: usize,
    /// Iteration piece cap — part of the metric definition.
    pub max_pieces: u32,
}

impl Default for Spec {
    fn default() -> Self {
        Self {
            seeds: 6,
            max_pieces: 150,
        }
    }
}

pub fn run(spec: &Spec, bot: &Bot, _rt: &Runtime) -> std::io::Result<()> {
    let seeds = seed_set(spec.seeds);
    let stats = evaluate_capped(
        &bot.spec.factory(),
        &seeds,
        DEFAULT_MAX_FRAMES,
        spec.max_pieces,
    );
    for o in &stats.outcomes {
        events::emit(
            "game",
            json!({
                "mode": "marathon",
                "seed": events::seed_hex(o.seed),
                "a": bot.name,
                "score": o.score,
                "level": o.level,
                "lines": o.lines,
                "pieces": o.pieces,
                "frames": o.frames,
                "topped": o.topped_out,
                "completed": o.completed,
                "attack": o.total_attack,
            }),
        );
    }
    events::emit(
        "result",
        json!({
            "score_per_second": stats.mean_score_per_second,
            "attack_per_piece": stats.mean_attack_per_piece,
        }),
    );
    println!("score_per_second {:.2}", stats.mean_score_per_second);
    println!("attack_per_piece {:.4}", stats.mean_attack_per_piece);
    eprintln!(
        "{} cap={} | {} seeds | APP={:.4} attack/game={:.1} | score={:.0} level={:.2} pieces={:.0} completion={:.0}%",
        bot.name,
        spec.max_pieces,
        seeds.len(),
        stats.mean_attack_per_piece,
        stats.mean_attack,
        stats.mean_score,
        stats.mean_level,
        stats.mean_pieces,
        stats.completion_rate * 100.0,
    );
    Ok(())
}
