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
    let bots: Vec<Bot> = bot_names.iter().map(|n| bot_or_die(n)).collect();
    let rt = Runtime {
        budget_secs: args.budget_secs,
        cc2_bin: args.cc2_bin.clone(),
    };
    RunDir::create(
        args.runs_root.as_deref(),
        entry.name,
        json!({
            "experiment": entry.name,
            "spec": registry::spec_json(&entry.experiment),
            "bots": bot_names,
            "runtime": &rt,
        }),
    )?;

    use Experiment::*;
    let bot = |i: usize| bots[i];
    match &entry.experiment {
        Marathon(spec) => commands::marathon::run(spec, &bot(0), &rt),
        Downstack(spec) => commands::downstack::run(spec, &bot(0), &rt),
        Versus(spec) => commands::versus::run(spec, &bot(0), &bot(1), &rt),
        Race(spec) => commands::race::run(spec, &bot(0), &bot(1), &rt),
        Cc2Baseline(spec) => commands::cc2_baseline::run(spec, &rt),
    }
}

fn main() -> std::io::Result<()> {
    match Cli::parse().command {
        Command::Run { eval, bots, rt } => execute(&find_or_die(&eval), &bots, &rt),
    }
}
