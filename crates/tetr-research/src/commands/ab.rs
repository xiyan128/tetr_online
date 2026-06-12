//! The garbage-awareness A/B: the same bot, sighted vs blinded.
//!
//! Side ARM-AWARE sees the pending-garbage queue (and its search models
//! cancellation + rising exactly — the engine-mirrored transition); the other
//! arm is the identical bot behind `BlindToGarbage`, planning as if nothing
//! were ever queued. Same weights, same search, same seeds, same piece
//! sequences.
//!
//! Two reporting decisions matter (both prompted by adversarial review):
//!
//! - **Arms swap**: every seed is played twice, aware as side A then as side B,
//!   removing any residual side asymmetry (A always moves first within a ply).
//! - **Survival-centric verdict**: most games end at the ply cap, where
//!   `decide_versus` falls back to NET attack — a metric structurally hostile
//!   to awareness (cancelled lines never count, and the aware bot deliberately
//!   trades net attack for cancellation). So deaths are reported first-class:
//!   `aware_death_rate` vs `blind_death_rate` is the headline, with the
//!   cap-game attack tiebreak shown separately for what it is.

use serde_json::json;

use tetr_core::ai::Cc2Weights;

use crate::bots::BotSpec;
use crate::commands::{Beam, Runtime};
use crate::ledger::RunLedger;
use crate::seeds::seed_set;
use crate::versus::{VersusFormat, VersusResult, VersusStats, evaluate_versus_format};

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum Bot {
    /// Beam over CC2 defaults.
    Beam,
    /// Best-first (the `node_budget` arm; depth floors at 6).
    Bf,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum WeightsArg {
    /// CC2 defaults.
    Default,
    /// `attack_tuned` — the shipped operating point's attack output, the
    /// pressure regime where deaths (the verdict awareness exists for) occur.
    Attack,
}

#[derive(Clone, Copy, Debug, serde::Serialize)]
pub struct Spec {
    /// Seed count (doubled by the arm swap).
    pub seeds: usize,
    /// The spec under test.
    pub bot: Bot,
    pub beam: Beam,
    /// Best-first total node expansions (`bot: Bf`).
    pub node_budget: u32,
    /// Ply cap per match.
    pub max_plies: u32,
    /// Queue one symmetric line every N plies (0 = off) — the decisiveness
    /// knob (mirror matches almost never kill without it).
    pub rain_period: u32,
    /// Evaluator weights for the best-first arm.
    pub weights: WeightsArg,
}

impl Spec {
    pub fn bot(bot: Bot) -> Self {
        Self {
            seeds: 48,
            bot,
            beam: Beam::default(),
            node_budget: 192,
            max_plies: 160,
            rain_period: 0,
            weights: WeightsArg::Default,
        }
    }
}

/// Deaths and cap-game outcomes for the aware arm of one orientation.
/// `aware_is_a`: which side the aware bot played in this run.
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

pub fn run(spec: &Spec, _rt: &Runtime, ledger: &mut RunLedger) -> std::io::Result<()> {
    let seeds = seed_set(spec.seeds);
    let Beam { width, depth } = spec.beam;
    let format = VersusFormat {
        max_plies: spec.max_plies,
        rain_period: spec.rain_period,
    };

    let make: BotSpec = match spec.bot {
        Bot::Bf => {
            let depth = if depth < 4 { 6 } else { depth };
            let weights = match spec.weights {
                WeightsArg::Attack => Cc2Weights::attack_tuned(),
                WeightsArg::Default => Cc2Weights::DEFAULT,
            };
            eprintln!(
                "Garbage-awareness A/B — CC2-eval best-first(nodes={}, depth={depth}), {} seeds x2 (arm swap), {} plies, rain {}",
                spec.node_budget,
                seeds.len(),
                spec.max_plies,
                spec.rain_period
            );
            BotSpec::best_first(spec.node_budget, depth).cc2(weights)
        }
        Bot::Beam => {
            eprintln!(
                "Garbage-awareness A/B — CC2-eval beam(depth={depth}, width={width}), {} seeds x2 (arm swap), {} plies, rain {}",
                seeds.len(),
                spec.max_plies,
                spec.rain_period
            );
            BotSpec::beam(width, depth).cc2(Cc2Weights::DEFAULT)
        }
    };

    // Orientation 1: aware as A. Orientation 2: aware as B. Same seeds; the
    // blind arm is the same spec with the pending queue hidden.
    let fwd = evaluate_versus_format(&make.factory(), &make.blind().factory(), &seeds, format);
    let rev = evaluate_versus_format(&make.blind().factory(), &make.factory(), &seeds, format);
    for outcome in &fwd.outcomes {
        ledger.append_outcome(&json!({ "orientation": "aware_a", "outcome": outcome }))?;
    }
    for outcome in &rev.outcomes {
        ledger.append_outcome(&json!({ "orientation": "aware_b", "outcome": outcome }))?;
    }

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
    ledger.write_summary(json!({
        "exit_reason": "complete",
        "games": games,
        "aware_deaths": aware_deaths,
        "blind_deaths": blind_deaths,
        "aware_death_rate": f64::from(aware_deaths) / f64::from(games),
        "blind_death_rate": f64::from(blind_deaths) / f64::from(games),
        "aware_cap_wins": aware_cap_wins,
        "blind_cap_wins": blind_cap_wins,
        "death_decisive_games": deaths,
        "mean_attack": {
            "forward_aware": fwd.mean_attack_a,
            "forward_blind": fwd.mean_attack_b,
            "reverse_blind": rev.mean_attack_a,
            "reverse_aware": rev.mean_attack_b,
        },
    }))?;
    Ok(())
}
