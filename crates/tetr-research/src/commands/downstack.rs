//! Downstack (cheese) eval for one bot: censored pieces-to-clear (the
//! optimization-safe scalar — failures count as the cap) plus clear rate,
//! both on parsed stdout. Lower censored pieces = better digging.

use serde_json::json;

use crate::bots::Bot;
use crate::commands::Runtime;
use crate::downstack::evaluate_downstack;
use crate::events;
use crate::seeds::seed_set;

#[derive(Clone, Copy, Debug, serde::Serialize)]
pub struct Spec {
    pub seeds: usize,
    /// Cheese height.
    pub garbage_rows: u32,
    /// Censoring cap — part of the metric definition (the censored mean is
    /// only comparable at one cap).
    pub max_pieces: u32,
}

impl Default for Spec {
    fn default() -> Self {
        Self {
            seeds: 6,
            garbage_rows: 9,
            max_pieces: 100,
        }
    }
}

pub fn run(spec: &Spec, bot: &Bot, _rt: &Runtime) -> std::io::Result<()> {
    let seeds = seed_set(spec.seeds);
    let ds = evaluate_downstack(
        &bot.spec.factory(),
        &seeds,
        spec.garbage_rows,
        spec.max_pieces,
    );
    for o in &ds.outcomes {
        events::game(json!({
            "seed": events::seed_hex(o.seed),
            "pieces": o.pieces,
            "cleared": o.cleared,
            "topped": o.topped_out,
            "attack": o.attack,
        }));
    }
    println!("downstack_pieces_censored {:.2}", ds.mean_pieces_censored);
    println!("downstack_clear_rate {:.2}", ds.clear_rate);
    eprintln!(
        "{} | {} seeds | {} garbage rows, cap {} | clear_rate={:.0}% cleared-only mean={:.2} attack={:.1}",
        bot.name,
        seeds.len(),
        spec.garbage_rows,
        spec.max_pieces,
        ds.clear_rate * 100.0,
        ds.mean_pieces_to_clear,
        ds.mean_attack,
    );
    Ok(())
}
