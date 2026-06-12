//! `tetr-research` — run named experiments from the registry.
//!
//! ```text
//! cargo run --release -p tetr-research -- list
//! cargo run --release -p tetr-research -- show cc2-board-climb
//! cargo run --release -p tetr-research -- run  cc2-board-climb --budget-secs 3600
//! cargo run --release -p tetr-research -- resume runs/20260612-...-cc2-board-climb-123
//! ```
//!
//! Experiments are configured in ONE place — [`tetr_research::registry`], as
//! Rust literals — and addressed here purely by name. The flags on `run` /
//! `resume` are machine-local circumstances (budgets, paths), never
//! experiment identity; see the registry docs for the reproducibility
//! contract this enforces.

use std::path::PathBuf;

use clap::{Args, Parser, Subcommand};
use serde_json::json;

use tetr_research::commands::{self, Runtime};
use tetr_research::ledger::RunLedger;
use tetr_research::registry::{self, Experiment};

#[derive(Parser, Debug)]
#[command(
    name = "tetr-research",
    version,
    about = "Deterministic experiment platform: run registry entries by name",
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
    /// Wall-clock budget in seconds (each experiment documents its default).
    #[arg(long)]
    budget_secs: Option<u64>,
    /// Climbs: new iterations this invocation (0 = unbounded).
    #[arg(long, default_value_t = 0)]
    max_iters: u32,
    /// Path to a Cold Clear 2 build (`cc2-baseline-*` entries).
    #[arg(long)]
    cc2_bin: Option<PathBuf>,
    /// Run-directory root (default: `<git toplevel>/runs`).
    #[arg(long)]
    runs_root: Option<PathBuf>,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// List every registered experiment.
    List {
        /// One JSON object per entry instead of the table.
        #[arg(long)]
        json: bool,
    },
    /// Print a registered experiment's spec — the exact JSON its manifest records.
    Show { name: String },
    /// Run a registered experiment by name.
    Run {
        name: String,
        #[command(flatten)]
        rt: RuntimeArgs,
    },
    /// Continue an interrupted run from its run directory (climbs only).
    Resume {
        run_dir: PathBuf,
        #[command(flatten)]
        rt: RuntimeArgs,
    },
    /// List recorded runs from the ledger, oldest first.
    Runs {
        /// Show only the most recent N.
        #[arg(long, default_value_t = 20)]
        last: usize,
        /// Run-directory root (default: `<git toplevel>/runs`).
        #[arg(long)]
        runs_root: Option<PathBuf>,
    },
}

fn die(message: &str) -> ! {
    eprintln!("{message}");
    std::process::exit(2);
}

fn find_or_die(name: &str) -> registry::Entry {
    registry::find(name).unwrap_or_else(|| {
        die(&format!(
            "unknown experiment {name:?} — `list` shows the registry"
        ))
    })
}

fn runtime(args: &RuntimeArgs, resume_from: Option<PathBuf>) -> Runtime {
    Runtime {
        budget_secs: args.budget_secs,
        max_iters: args.max_iters,
        cc2_bin: args.cc2_bin.clone(),
        resume_from,
    }
}

/// Create the run's ledger and dispatch to the experiment's command. The
/// manifest's `extra` is uniform across kinds: the entry name, the full
/// typed spec, and this invocation's runtime.
fn execute(
    entry: &registry::Entry,
    args: &RuntimeArgs,
    resume: Option<PathBuf>,
) -> std::io::Result<()> {
    let rt = runtime(args, resume);
    let extra = json!({
        "experiment": entry.name,
        "spec": registry::spec_json(&entry.experiment),
        "runtime": &rt,
    });
    let mut ledger = match &args.runs_root {
        Some(root) => RunLedger::create_at(root, entry.name, extra)?,
        None => RunLedger::create(entry.name, extra)?,
    };

    match (&entry.experiment, &rt.resume_from) {
        (Experiment::Climb(spec), Some(prior)) => {
            commands::climb::resume(spec, &rt, &prior.clone(), &mut ledger)
        }
        (_, Some(_)) => die("only climbs carry checkpoints; `resume` works on climb runs"),
        (Experiment::Climb(spec), None) => commands::climb::run(spec, &rt, &mut ledger),
        (Experiment::Metric(spec), None) => commands::metric::run(spec, &rt, &mut ledger),
        (Experiment::Marathon(spec), None) => commands::marathon::run(spec, &rt, &mut ledger),
        (Experiment::Behavior(spec), None) => commands::behavior::run(spec, &rt, &mut ledger),
        (Experiment::Ab(spec), None) => commands::ab::run(spec, &rt, &mut ledger),
        (Experiment::Race(spec), None) => commands::sprt::run(spec, &rt, &mut ledger),
        (Experiment::Promote(spec), None) => commands::promote::run(spec, &rt, &mut ledger),
        (Experiment::Cc2Native(spec), None) => commands::cc2_native::run(spec, &rt, &mut ledger),
        (Experiment::Cc2Baseline(spec), None) => {
            commands::cc2_baseline::run(spec, &rt, &mut ledger)
        }
    }
}

fn main() -> std::io::Result<()> {
    match Cli::parse().command {
        Command::List { json } => {
            for entry in registry::entries() {
                if json {
                    println!(
                        "{}",
                        json!({
                            "name": entry.name,
                            "about": entry.about,
                            "spec": registry::spec_json(&entry.experiment),
                        })
                    );
                } else {
                    let kind = registry::spec_json(&entry.experiment)["kind"]
                        .as_str()
                        .unwrap_or("?")
                        .to_string();
                    println!("{:<24} {kind:<13} {}", entry.name, entry.about);
                }
            }
            Ok(())
        }
        Command::Show { name } => {
            let entry = find_or_die(&name);
            println!(
                "{}",
                serde_json::to_string_pretty(&registry::spec_json(&entry.experiment))?
            );
            Ok(())
        }
        Command::Run { name, rt } => execute(&find_or_die(&name), &rt, None),
        Command::Resume { run_dir, rt } => {
            // The stored manifest names the experiment and freezes its spec;
            // a drifted registry entry is refused — register a new name
            // instead of mutating one with recorded runs.
            let spec_path = run_dir.join("spec.json");
            let stored: serde_json::Value = std::fs::File::open(&spec_path)
                .map_err(|e| {
                    die(&format!("cannot open {}: {e}", spec_path.display()));
                })
                .and_then(|f| {
                    serde_json::from_reader(f)
                        .map_err(|e| die(&format!("{} does not parse: {e}", spec_path.display())))
                })
                .unwrap();
            let name = stored["extra"]["experiment"]
                .as_str()
                .unwrap_or_else(|| die("this run predates the registry; resume it by hand"));
            let entry = find_or_die(name);
            let current = registry::spec_json(&entry.experiment);
            if stored["extra"]["spec"] != current {
                die(&format!(
                    "registry entry '{name}' changed since this run was recorded;\n\
                     register the changed configuration under a NEW name, or check out\n\
                     the run's commit. stored:\n{}\ncurrent:\n{}",
                    stored["extra"]["spec"], current
                ));
            }
            execute(&entry, &rt, Some(run_dir))
        }
        Command::Runs { last, runs_root } => commands::runs::list(runs_root.as_deref(), last),
    }
}
