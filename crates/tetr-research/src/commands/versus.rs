//! Head-to-head eval under the engine's garbage rules: bot A vs bot B over a
//! seed set, reporting win rates, deaths, and mean attack. Remember the
//! conventions: capped-game win rate leans on the anti-defensive net-attack
//! tiebreak — survival VERDICTS belong to `race`, not here.

use crate::bots::BotSpec;
use crate::commands::Runtime;
use crate::seeds::seed_set;
use crate::versus::{VersusFormat, evaluate_versus_format};

#[derive(Clone, Copy, Debug, serde::Serialize)]
pub struct Spec {
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

pub fn run(spec: &Spec, a: &BotSpec, b: &BotSpec, _rt: &Runtime) -> std::io::Result<()> {
    let seeds = seed_set(spec.seeds);
    let stats = evaluate_versus_format(&a.factory(), &b.factory(), &seeds, spec.format);
    let (mut a_deaths, mut b_deaths) = (0u32, 0u32);
    for o in &stats.outcomes {
        a_deaths += u32::from(o.a_topped);
        b_deaths += u32::from(o.b_topped);
    }
    println!("versus_a_win_rate {:.2}", stats.a_win_rate());
    println!("versus_a_deaths {a_deaths}");
    println!("versus_b_deaths {b_deaths}");
    eprintln!(
        "A {a:?} vs B {b:?} | A {} / B {} / draw {} | deaths A {a_deaths} B {b_deaths} | attack A {:.1} B {:.1} | {} seeds, {} plies, rain {}",
        stats.a_wins,
        stats.b_wins,
        stats.draws,
        stats.mean_attack_a,
        stats.mean_attack_b,
        seeds.len(),
        spec.format.max_plies,
        spec.format.rain_period,
    );
    Ok(())
}
