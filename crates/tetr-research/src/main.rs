//! `tetr-research` — run a registered eval on registered bots.
//!
//! ```text
//! cargo run --release -p tetr-research -- run downstack dt20
//! cargo run --release -p tetr-research -- run race v3-candidate attack-tuned
//! cargo run --release -p tetr-research -- run cc2-board-climb --budget-secs 3600
//! cargo run --release -p tetr-research -- resume runs/20260612-...-cc2-board-climb-123
//! ```
//!
//! Everything is a name: evals live in [`tetr_research::registry`], bots in
//! [`tetr_research::bots`] — read those files for the catalogs (there is no
//! `list`; the registries are code). A recorded result reproduces from
//! `(commit, eval, bots…)`, all stamped into the run receipt. The flags are
//! machine-local circumstances (budgets, paths), never experiment identity.

use std::path::PathBuf;

use clap::{Args, Parser, Subcommand};
use serde_json::json;

use tetr_research::bots::{self, Bot};
use tetr_research::commands::{self, Runtime};
use tetr_research::events;
use tetr_research::ledger::RunDir;
use tetr_research::registry::{self, Experiment};

#[derive(Parser, Debug)]
#[command(
    name = "tetr-research",
    version,
    about = "Deterministic experiment platform: `run <eval> [bots…]` (catalogs: src/registry.rs, src/bots.rs)",
    max_term_width = 100
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

/// Machine-local knobs only — these bound how much of a deterministic
/// experiment this invocation materializes, never which experiment runs.
#[derive(Args, Debug, Default)]
struct RuntimeArgs {
    /// Wall-clock budget in seconds (each eval documents its default).
    #[arg(long)]
    budget_secs: Option<u64>,
    /// Path to a Cold Clear 2 build (`cc2-baseline-*` evals).
    #[arg(long)]
    cc2_bin: Option<PathBuf>,
    /// Run-directory root (default: `<git toplevel>/runs`).
    #[arg(long)]
    runs_root: Option<PathBuf>,
    /// Run despite a dirty tree (or no git checkout). The receipt still
    /// stamps `git.dirty`; such runs are exploratory — not re-runnable from
    /// `(commit, eval, bots…)` — and analysis filters them by default.
    #[arg(long)]
    allow_dirty: bool,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Run a registered eval on registered bots.
    Run {
        /// The eval's registry name (src/registry.rs).
        eval: String,
        /// The bot name(s) the eval's slots need (src/bots.rs).
        bots: Vec<String>,
        #[command(flatten)]
        rt: RuntimeArgs,
    },
    /// CRN pair duel between two arms under the sudden-death venue.
    /// `duel --a beam:M@w8d5 --b policy:M` is the G_pi probe; any
    /// candidate-vs-incumbent race is the same command.
    Duel {
        /// Arm A (see src/arm.rs for the grammar).
        #[arg(long)]
        a: tetr_research::arm::Arm,
        /// Arm B.
        #[arg(long)]
        b: tetr_research::arm::Arm,
        /// CRN pairs to play (2 games each, arms swapped).
        #[arg(long, default_value_t = 64)]
        pairs: usize,
        /// First seed of the region (the caller owns disjointness).
        #[arg(long)]
        seeds: u64,
        #[command(flatten)]
        venue: VenueArgs,
        #[command(flatten)]
        rt: RuntimeArgs,
    },
    /// Latched trinomial pair-GSPRT gate: arm A (candidate) vs arm B
    /// (incumbent). The verdict latches at the first boundary crossing;
    /// in-flight pairs are reported but never decide.
    Gate {
        #[arg(long)]
        a: tetr_research::arm::Arm,
        #[arg(long)]
        b: tetr_research::arm::Arm,
        /// Hard cap on pairs (hitting it = Inconclusive).
        #[arg(long, default_value_t = 400)]
        max_pairs: usize,
        /// First seed of the region (the caller owns disjointness).
        #[arg(long)]
        seeds: u64,
        /// H1 per-decisive-game win probability (H0 is 0.5).
        #[arg(long, default_value_t = 0.55)]
        p1: f64,
        /// Pairs before any verdict is allowed.
        #[arg(long, default_value_t = 32)]
        min_pairs: u32,
        #[command(flatten)]
        venue: VenueArgs,
        #[command(flatten)]
        rt: RuntimeArgs,
    },
}

/// Default wall-clock budget for an instrument run: 6 hours, a safety cap on a
/// hung/slow arm — well beyond any healthy duel or gate.
const DEFAULT_INSTRUMENT_BUDGET_SECS: u64 = 6 * 60 * 60;

/// The sudden-death venue knobs (defaults = the calibrated venue).
#[derive(Args, Debug)]
struct VenueArgs {
    /// Ply cap before sudden-death escalation begins.
    #[arg(long, default_value_t = 240)]
    max_plies: u32,
    /// Rain period (garbage line to both seats every N plies).
    #[arg(long, default_value_t = 8)]
    rain: u32,
}

impl VenueArgs {
    fn venue(&self) -> tetr_research::instruments::Venue {
        tetr_research::instruments::Venue {
            max_plies: self.max_plies,
            rain_period: self.rain,
        }
    }
}

fn die(message: &str) -> ! {
    eprintln!("{message}");
    std::process::exit(2);
}

fn find_or_die(name: &str) -> registry::Entry {
    registry::find(name).unwrap_or_else(|| {
        die(&format!(
            "unknown eval {name:?} — the catalog is src/registry.rs"
        ))
    })
}

fn bot_or_die(name: &str) -> Bot {
    bots::find(name).unwrap_or_else(|| {
        die(&format!(
            "unknown bot {name:?} — the catalog is src/bots.rs"
        ))
    })
}

/// Write the receipt, resolve the bots, and dispatch.
fn execute(
    entry: &registry::Entry,
    bot_names: &[String],
    args: &RuntimeArgs,
) -> std::io::Result<()> {
    if bot_names.len() != entry.experiment.bot_slots() {
        die(&format!(
            "{} takes {} bot name(s): run {} {}",
            entry.name,
            entry.experiment.bot_slots(),
            entry.name,
            entry.experiment.usage(),
        ));
    }
    if !args.allow_dirty && tetr_research::ledger::dirty() != Some(false) {
        die(
            "refusing to run: the working tree is dirty (or not a git checkout), so this \
             run would not be re-runnable from (commit, eval, bots…).\n\
             commit first, or pass --allow-dirty to record an exploratory run.",
        );
    }
    let bots: Vec<Bot> = bot_names.iter().map(|n| bot_or_die(n)).collect();
    let rt = Runtime {
        budget_secs: args.budget_secs,
        cc2_bin: args.cc2_bin.clone(),
    };
    let run_dir = RunDir::create(
        args.runs_root.as_deref(),
        entry.name,
        json!({
            "experiment": entry.name,
            "spec": registry::spec_json(&entry.experiment),
            "bots": bot_names,
            "runtime": &rt,
        }),
    )?;
    events::install(run_dir.dir())?;

    use Experiment::*;
    let bot = |i: usize| bots[i];
    let result = match &entry.experiment {
        Marathon(spec) => commands::marathon::run(spec, &bot(0), &rt),
        Pc(spec) => commands::pc::run(spec, &bot(0), &rt),
        Downstack(spec) => commands::downstack::run(spec, &bot(0), &rt),
        Versus(spec) => commands::versus::run(spec, &bot(0), &bot(1), &rt),
        Race(spec) => commands::race::run(spec, &bot(0), &bot(1), &rt),
        Cc2Baseline(spec) => commands::cc2_baseline::run(spec, &rt),
        AppClimb(spec) => commands::climb_app::run(spec, &bot(0), &rt),
    }?;
    // The entire stdout contract: ONE self-describing JSON line per run
    // (humans read stderr; pipelines read this).
    let mut line = json!({
        "run": run_dir.dir().display().to_string(),
        "eval": entry.name,
        "bots": bot_names,
    });
    if let (serde_json::Value::Object(line), serde_json::Value::Object(result)) =
        (&mut line, result)
    {
        line.extend(result);
    }
    println!("{line}");
    Ok(())
}

/// Shared instrument preamble: the dirty-tree refusal + a receipt dir; and
/// the shared epilogue: ONE self-describing JSON line on stdout.
fn run_instrument(
    name: &str,
    spec: serde_json::Value,
    rt: &RuntimeArgs,
    body: impl FnOnce() -> serde_json::Value,
) -> std::io::Result<()> {
    if !rt.allow_dirty && tetr_research::ledger::dirty() != Some(false) {
        die(
            "refusing to run: the working tree is dirty (or not a git checkout), so this \
             run would not be re-runnable from (commit, args…).\n\
             commit first, or pass --allow-dirty to record an exploratory run.",
        );
    }
    let run_dir = RunDir::create(rt.runs_root.as_deref(), name, spec)?;
    events::install(run_dir.dir())?;
    let result = body();
    let mut line = json!({ "run": run_dir.dir().display().to_string(), "eval": name });
    if let (serde_json::Value::Object(line), serde_json::Value::Object(result)) =
        (&mut line, result)
    {
        line.extend(result);
    }
    println!("{line}");
    Ok(())
}

fn main() -> std::io::Result<()> {
    match Cli::parse().command {
        Command::Run { eval, bots, rt } => execute(&find_or_die(&eval), &bots, &rt),
        Command::Duel {
            a,
            b,
            pairs,
            seeds,
            venue,
            rt,
        } => {
            let budget = std::time::Duration::from_secs(
                rt.budget_secs.unwrap_or(DEFAULT_INSTRUMENT_BUDGET_SECS),
            );
            run_instrument(
                "duel",
                json!({
                    "experiment": "duel",
                    "spec": { "a": a.to_string(), "b": b.to_string(), "pairs": pairs,
                              "seeds": seeds, "venue": venue.venue() },
                    "runtime": { "budget_secs": budget.as_secs() },
                }),
                &rt,
                || tetr_research::instruments::duel(&a, &b, venue.venue(), seeds, pairs, budget),
            )
        }
        Command::Gate {
            a,
            b,
            max_pairs,
            seeds,
            p1,
            min_pairs,
            venue,
            rt,
        } => {
            let budget = std::time::Duration::from_secs(
                rt.budget_secs.unwrap_or(DEFAULT_INSTRUMENT_BUDGET_SECS),
            );
            run_instrument(
                "gate",
                json!({
                    "experiment": "gate",
                    "spec": { "a": a.to_string(), "b": b.to_string(), "max_pairs": max_pairs,
                              "seeds": seeds, "p1": p1, "min_pairs": min_pairs,
                              "venue": venue.venue() },
                    "runtime": { "budget_secs": budget.as_secs() },
                }),
                &rt,
                || {
                    tetr_research::instruments::gate(
                        &a,
                        &b,
                        venue.venue(),
                        seeds,
                        max_pairs,
                        p1,
                        min_pairs,
                        budget,
                    )
                },
            )
        }
    }
}
