//! `tetr-climb` — the search side: optimize a bot, then gate the result.
//!
//! ```text
//! cargo run --release -p tetr-research --bin tetr-climb -- climb cc2-board-v4
//! cargo run --release -p tetr-research --bin tetr-climb -- panel default cc2-board-v4-r1
//! ```
//!
//! Climbs mutate a subject bot's weights against the versus objective behind
//! the screen → confirm → anchor gate chain; panels render the PROMOTE /
//! REJECT verdict. Configurations are named literals in `src/search/mod.rs`
//! (read it — there is no `list`); measurement itself stays in the
//! `tetr-research` binary. Interrupted climbs are simply rerun: the walk
//! replays deterministically from its spec.

use std::time::Duration;

use clap::{Parser, Subcommand};
use serde_json::json;

use tetr_research::bots;
use tetr_research::ledger::RunDir;
use tetr_research::search;

#[derive(Parser, Debug)]
#[command(
    name = "tetr-climb",
    version,
    about = "Optimize bots and gate the results (catalog: src/search/mod.rs)",
    max_term_width = 100
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Run a named climb configuration.
    Climb {
        name: String,
        /// Wall-clock budget in seconds (default 1800).
        #[arg(long)]
        budget_secs: Option<u64>,
        /// New iterations this invocation (0 = unbounded).
        #[arg(long, default_value_t = 0)]
        max_iters: u32,
    },
    /// Judge a candidate bot against a named panel configuration.
    Panel {
        name: String,
        candidate: String,
        /// Wall-clock budget in seconds shared by all cells (default 3600).
        #[arg(long)]
        budget_secs: Option<u64>,
    },
}

fn die(message: &str) -> ! {
    eprintln!("{message}");
    std::process::exit(2);
}

fn main() -> std::io::Result<()> {
    match Cli::parse().command {
        Command::Climb {
            name,
            budget_secs,
            max_iters,
        } => {
            let (_, spec) = search::climbs()
                .into_iter()
                .find(|(n, _)| *n == name)
                .unwrap_or_else(|| {
                    die(&format!(
                        "unknown climb {name:?} — the catalog is src/search/mod.rs"
                    ))
                });
            RunDir::create(
                None,
                &format!("climb-{name}"),
                json!({ "experiment": format!("climb-{name}"), "spec": &spec }),
            )?;
            search::climb::run(&spec, budget_secs.map(Duration::from_secs), max_iters)
        }
        Command::Panel {
            name,
            candidate,
            budget_secs,
        } => {
            let (_, spec) = search::panels()
                .into_iter()
                .find(|(n, _)| *n == name)
                .unwrap_or_else(|| {
                    die(&format!(
                        "unknown panel {name:?} — the catalog is src/search/mod.rs"
                    ))
                });
            let cand = bots::find(&candidate).unwrap_or_else(|| {
                die(&format!(
                    "unknown bot {candidate:?} — the catalog is src/bots.rs"
                ))
            });
            RunDir::create(
                None,
                &format!("panel-{name}"),
                json!({
                    "experiment": format!("panel-{name}"),
                    "spec": &spec,
                    "bots": [cand.name],
                }),
            )?;
            search::panel::run(&spec, &cand, budget_secs.map(Duration::from_secs))
        }
    }
}
