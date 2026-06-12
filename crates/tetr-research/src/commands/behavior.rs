//! Behavior + APP suite report for one bot across the standard garbage
//! scenarios. APP (attack per piece) is the primary strike metric; also
//! reports DS/P, survival, attack/line (concentration vs combo-spam), and
//! the clear-type behavior histogram. Custom-weight arms are registered
//! bots, not knobs.

use crate::behavior::{ScenarioReport, evaluate_scenario, standard_suite};
use crate::bots::Bot;
use crate::commands::Runtime;
use crate::seeds::seed_set;

#[derive(Clone, Copy, Debug, serde::Serialize)]
pub struct Spec {
    /// Seed count per scenario.
    pub seeds: usize,
}

impl Default for Spec {
    fn default() -> Self {
        Self { seeds: 24 }
    }
}

fn print_report(r: &ScenarioReport) {
    eprintln!(
        "\n[{}] survival {:.0}% | APP {:.3} | DS/P {:.2} | atk/line {:.2} | pieces {:.0} | garbage_recv {:.1} | {:.1} ms/piece",
        r.scenario.label(),
        r.survival_rate * 100.0,
        r.mean_app,
        r.mean_dsp,
        r.mean_attack_per_line,
        r.mean_pieces,
        r.mean_garbage_received,
        r.mean_ms_per_piece,
    );
    let t = &r.totals;
    eprintln!(
        "    clears: S{} D{} T{} Quad{} | TSmini{} TSS{} TSD{} TST{} | B2B{} comboClears{} maxCombo{} PC{}",
        t.singles,
        t.doubles,
        t.triples,
        t.tetrises,
        t.tspin_mini,
        t.tspin_single,
        t.tspin_double,
        t.tspin_triple,
        t.b2b_clears,
        t.combo_clears,
        t.max_combo,
        t.perfect_clears,
    );
    println!("APP[{}] {:.3}", r.scenario.label(), r.mean_app);
}

pub fn run(spec: &Spec, bot: &Bot, _rt: &Runtime) -> std::io::Result<()> {
    let seeds = seed_set(spec.seeds);
    eprintln!(
        "Behavior + APP suite | {} | {} seeds",
        bot.name,
        seeds.len()
    );
    for scenario in standard_suite() {
        let report = evaluate_scenario(&bot.spec.factory(), &seeds, scenario);
        print_report(&report);
    }
    Ok(())
}
