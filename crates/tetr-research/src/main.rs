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
    /// Generate self-play decision shards (the datagen data plant). No `--net`
    /// (CC2 eval) makes the round-0 BC corpus; a `--net <dir>` makes round-1+
    /// self-play. Writes game-aligned shards to `--out`.
    Datagen {
        /// Net model dir for the leaf eval; omit for CC2 (round-0 BC corpus).
        #[arg(long)]
        net: Option<PathBuf>,
        /// Beam width.
        #[arg(long, default_value_t = 8)]
        width: usize,
        /// Beam depth.
        #[arg(long, default_value_t = 5)]
        depth: u8,
        /// Guided vehicle: net policy top-m placements per node (0 = off;
        /// needs --net).
        #[arg(long, default_value_t = 0)]
        topm: usize,
        /// Parallel workers (games partitioned round-robin; each worker owns
        /// out/wN/ so shard numbering never collides).
        #[arg(long, default_value_t = 1)]
        workers: usize,
        /// Two-arm mode: the opponent seat plays a plain CC2 TP-beam at the
        /// same width/depth (grounded-opponent games; net seat alternates by
        /// game parity). Requires --net.
        #[arg(long)]
        opp_cc2: bool,
        /// Number of games (seeds `base..base+games`).
        #[arg(long, default_value_t = 100)]
        games: u64,
        /// First seed of the region (the caller owns disjointness).
        #[arg(long)]
        seeds: u64,
        /// Output dir for shards.
        #[arg(long)]
        out: PathBuf,
        #[command(flatten)]
        venue: VenueArgs,
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
        Command::Datagen {
            net,
            width,
            depth,
            topm,
            games,
            seeds,
            out,
            venue,
            workers,
            opp_cc2,
        } => {
            use tetr_research::datagen::{BeamConfig, datagen_game};
            let eval: Box<dyn tetr_core::ai::eval::Evaluator> = match &net {
                Some(dir) => Box::new(
                    tetr_nn::serve::NetEvaluator::load(dir)
                        .unwrap_or_else(|e| die(&format!("net load {}: {e}", dir.display()))),
                ),
                None => Box::new(tetr_core::ai::Cc2Evaluator::new(
                    tetr_core::ai::eval::Cc2Weights::attack_tuned(),
                )),
            };
            let cfg = BeamConfig {
                width,
                depth,
                transpose: true,
                top_m: topm,
            };
            let venue_fmt = tetr_research::versus::VersusFormat {
                max_plies: venue.max_plies,
                rain_period: venue.rain,
                sudden_death: true,
            };
            let t0 = std::time::Instant::now();
            let n_workers = workers.max(1);
            drop(eval); // each worker owns its eval; the probe load above validated the dir
            let wld_total = std::sync::Mutex::new([0u64; 3]);
            std::thread::scope(|scope| -> std::io::Result<()> {
                let mut handles = Vec::new();
                for w in 0..n_workers {
                    let out_w = if n_workers == 1 {
                        out.clone()
                    } else {
                        out.join(format!("w{w}"))
                    };
                    let net_ref = &net;
                    let venue_ref = &venue_fmt;
                    let wld_ref = &wld_total;
                    handles.push(scope.spawn(move || -> std::io::Result<()> {
                        let eval: Box<dyn tetr_core::ai::eval::Evaluator> = match net_ref {
                            Some(dir) => Box::new(
                                tetr_nn::serve::NetEvaluator::load(dir)
                                    .map_err(|e| std::io::Error::other(e.to_string()))?,
                            ),
                            None => Box::new(tetr_core::ai::Cc2Evaluator::new(
                                tetr_core::ai::eval::Cc2Weights::attack_tuned(),
                            )),
                        };
                        let cc2 = tetr_core::ai::Cc2Evaluator::new(
                            tetr_core::ai::eval::Cc2Weights::attack_tuned(),
                        );
                        let mut writer = tetr_nn::shards::ShardWriter::create(&out_w, 1024)?;
                        let mut wld = [0u64; 3];
                        let mut i = w as u64;
                        while i < games {
                            let res = tetr_research::datagen::datagen_game_vs(
                                &mut writer,
                                [&*eval, &cc2],
                                cfg,
                                net_ref.as_deref(),
                                opp_cc2,
                                venue_ref,
                                seeds + i,
                                (seeds + i) as u32,
                            )?;
                            let idx = match res.result {
                                tetr_research::versus::VersusResult::AWins => 0,
                                tetr_research::versus::VersusResult::BWins => 1,
                                tetr_research::versus::VersusResult::Draw => 2,
                            };
                            wld[idx] += 1;
                            i += n_workers as u64;
                        }
                        writer.flush()?;
                        let mut total = wld_ref.lock().expect("wld lock");
                        for k in 0..3 {
                            total[k] += wld[k];
                        }
                        Ok(())
                    }));
                }
                for h in handles {
                    h.join().expect("datagen worker panicked")?;
                }
                Ok(())
            })?;
            let wld = *wld_total.lock().expect("wld lock");
            let secs = t0.elapsed().as_secs_f64();
            println!(
                "{}",
                json!({
                    "experiment": "datagen",
                    "eval": net.as_ref().map(|d| d.display().to_string()).unwrap_or_else(|| "cc2".into()),
                    "width": width, "depth": depth, "topm": topm, "games": games, "seeds": seeds,
                    "out": out.display().to_string(),
                    "wall_secs": secs,
                    "games_per_hr": games as f64 * 3600.0 / secs,
                    "a_b_draw": wld,
                })
            );
            Ok(())
        }
    }
}
