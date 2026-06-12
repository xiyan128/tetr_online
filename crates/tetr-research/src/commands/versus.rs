//! Head-to-head eval under the engine's garbage rules — ARM-SWAPPED: every
//! seed is played from both chairs on common random numbers (the crate
//! convention), so chair order and seed luck cancel and per-chair attack
//! means double as a symmetry check. Deaths are reported first-class; the
//! capped-game win rate leans on the net-attack tiebreak, which is
//! structurally anti-defensive (cancelled lines count for nothing) — treat
//! it as context. Ship-grade survival VERDICTS belong to `race`.
//!
//! Awareness A/Bs are this eval with a blinded twin from the bot registry
//! (`run versus cc2-default cc2-default-blind`): same brain, pending queue
//! hidden. Mirror pairings are bland without rain (≤6% decisive — the
//! recorded number); rain is the decisiveness dial.

use serde_json::json;

use crate::bots::Bot;
use crate::commands::Runtime;
use crate::events;
use crate::seeds::seed_set;
use crate::versus::{VersusFormat, evaluate_versus_format};

#[derive(Clone, Copy, Debug, serde::Serialize)]
pub struct Spec {
    /// Seed count (doubled by the arm swap).
    pub seeds: usize,
    pub format: VersusFormat,
}

impl Default for Spec {
    fn default() -> Self {
        Self {
            seeds: 12,
            format: VersusFormat {
                max_plies: 160,
                rain_period: 0,
            },
        }
    }
}

pub fn run(spec: &Spec, a: &Bot, b: &Bot, _rt: &Runtime) -> std::io::Result<serde_json::Value> {
    let seeds = seed_set(spec.seeds);
    let fwd = evaluate_versus_format(&a.spec.factory(), &b.spec.factory(), &seeds, spec.format);
    let rev = evaluate_versus_format(&b.spec.factory(), &a.spec.factory(), &seeds, spec.format);

    let (mut a_deaths, mut b_deaths) = (0u32, 0u32);
    for o in &fwd.outcomes {
        a_deaths += u32::from(o.a_topped);
        b_deaths += u32::from(o.b_topped);
    }
    for o in &rev.outcomes {
        a_deaths += u32::from(o.b_topped);
        b_deaths += u32::from(o.a_topped);
    }
    for (swapped, stats) in [(false, &fwd), (true, &rev)] {
        for o in &stats.outcomes {
            events::game(json!({
                "seed": events::seed_hex(o.seed),
                "swapped": swapped,
                "a_topped": o.a_topped,
                "b_topped": o.b_topped,
                "a_attack": o.attack_a,
                "b_attack": o.attack_b,
                "plies": o.plies,
            }));
        }
    }
    let a_wins = fwd.a_wins + rev.b_wins;
    let b_wins = fwd.b_wins + rev.a_wins;
    let games = seeds.len() * 2;
    let draws = games - a_wins - b_wins;

    eprintln!(
        "{} vs {} | {} {a_wins} / {} {b_wins} / draw {draws} | deaths {} {a_deaths}, {} {b_deaths} \
         (of {games} games) | {} seeds x2, {} plies, rain {}",
        a.name,
        b.name,
        a.name,
        b.name,
        a.name,
        b.name,
        seeds.len(),
        spec.format.max_plies,
        spec.format.rain_period,
    );
    // Per-chair attack means: each bot's pair should agree across chairs
    // (the arm-swap symmetry check); the tiebreak caveat above applies.
    eprintln!(
        "mean net attack: {} {:.1}/{:.1} (fwd/rev) | {} {:.1}/{:.1}",
        a.name, fwd.mean_attack_a, rev.mean_attack_b, b.name, fwd.mean_attack_b, rev.mean_attack_a,
    );
    Ok(json!({
        "a_win_rate": a_wins as f64 / games.max(1) as f64,
        "a_wins": a_wins, "b_wins": b_wins, "draws": draws,
        "a_deaths": a_deaths, "b_deaths": b_deaths,
    }))
}
