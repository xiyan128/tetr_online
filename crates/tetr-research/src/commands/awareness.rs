//! The garbage-awareness A/B: ONE bot, sighted vs its blinded twin.
//!
//! The aware arm sees the pending-garbage queue (its search models
//! cancellation + rising exactly — the engine-mirrored transition); the
//! blind arm is the identical bot behind `BlindToGarbage`. Same weights,
//! same search, same seeds, same piece sequences.
//!
//! Two reporting decisions matter (both prompted by adversarial review):
//! arms swap (every seed played twice, aware as A then as B), and a
//! survival-centric verdict — most games end at the ply cap, where the
//! net-attack tiebreak is structurally hostile to awareness (cancelled lines
//! never count), so deaths are the headline and cap tiebreaks are shown for
//! what they are.

use crate::bots::BotSpec;
use crate::commands::Runtime;
use crate::seeds::seed_set;
use crate::versus::{VersusFormat, VersusResult, VersusStats, evaluate_versus_format};

#[derive(Clone, Copy, Debug, serde::Serialize)]
pub struct Spec {
    /// Seed count (doubled by the arm swap).
    pub seeds: usize,
    pub format: VersusFormat,
}

impl Default for Spec {
    fn default() -> Self {
        Self {
            seeds: 48,
            format: VersusFormat {
                max_plies: 160,
                rain_period: 0,
            },
        }
    }
}

/// Deaths and cap-game outcomes for the aware arm of one orientation.
fn tally(stats: &VersusStats, aware_is_a: bool) -> (u32, u32, u32, u32) {
    let (mut aware_deaths, mut blind_deaths, mut aware_cap_wins, mut blind_cap_wins) =
        (0u32, 0u32, 0u32, 0u32);
    for o in &stats.outcomes {
        let (aware_topped, blind_topped) = if aware_is_a {
            (o.a_topped, o.b_topped)
        } else {
            (o.b_topped, o.a_topped)
        };
        aware_deaths += u32::from(aware_topped);
        blind_deaths += u32::from(blind_topped);
        if !o.a_topped && !o.b_topped {
            let aware_won = match o.result {
                VersusResult::AWins => aware_is_a,
                VersusResult::BWins => !aware_is_a,
                VersusResult::Draw => continue,
            };
            if aware_won {
                aware_cap_wins += 1;
            } else {
                blind_cap_wins += 1;
            }
        }
    }
    (aware_deaths, blind_deaths, aware_cap_wins, blind_cap_wins)
}

pub fn run(spec: &Spec, bot: &BotSpec, _rt: &Runtime) -> std::io::Result<()> {
    let seeds = seed_set(spec.seeds);
    eprintln!(
        "Garbage-awareness A/B — {bot:?}, {} seeds x2 (arm swap), {} plies, rain {}",
        seeds.len(),
        spec.format.max_plies,
        spec.format.rain_period
    );

    // Orientation 1: aware as A. Orientation 2: aware as B. Same seeds; the
    // blind arm is the same spec with the pending queue hidden.
    let fwd = evaluate_versus_format(&bot.factory(), &bot.blind().factory(), &seeds, spec.format);
    let rev = evaluate_versus_format(&bot.blind().factory(), &bot.factory(), &seeds, spec.format);

    let (fd_a, fd_b, fc_a, fc_b) = tally(&fwd, true);
    let (rd_a, rd_b, rc_a, rc_b) = tally(&rev, false);
    let (aware_deaths, blind_deaths) = (fd_a + rd_a, fd_b + rd_b);
    let (aware_cap_wins, blind_cap_wins) = (fc_a + rc_a, fc_b + rc_b);
    let games = (seeds.len() * 2) as u32;
    let deaths = aware_deaths + blind_deaths;

    println!(
        "aware_death_rate {:.3}",
        f64::from(aware_deaths) / f64::from(games)
    );
    println!(
        "blind_death_rate {:.3}",
        f64::from(blind_deaths) / f64::from(games)
    );
    eprintln!(
        "DEATHS (the survival verdict): aware {aware_deaths} vs blind {blind_deaths} (of {games} games, {deaths} decisive)"
    );
    eprintln!(
        "CAP-GAME attack tiebreaks (anti-aware metric, shown for completeness): aware {aware_cap_wins} vs blind {blind_cap_wins}"
    );
    eprintln!(
        "mean net attack: fwd A(aware) {:.1} B(blind) {:.1} | rev A(blind) {:.1} B(aware) {:.1}",
        fwd.mean_attack_a, fwd.mean_attack_b, rev.mean_attack_a, rev.mean_attack_b
    );
    Ok(())
}
