//! Elo-vs-compute Pareto sweep over the beam's (width, depth) search shape.
//!
//! Holds eval + sight constant (TP-beam over the attack-tuned CC2 weights, garbage-
//! aware — the champion family) and varies ONLY `(width, depth)`, so the result is the
//! pure search-shape strength/compute frontier: "how much playing strength does each
//! extra node of search buy, and where is the knee?".
//!
//! Two axes, measured independently:
//! - **compute (x):** median per-decision wall-time (native release) of one full
//!   `think_to_completion` over a fixed bank of realistic mid-game states, plus the
//!   deterministic node count. This is the work a single piece costs.
//! - **Elo (y):** fit in `analysis/elo-pareto/elo_pareto.py` from a versus tournament's
//!   pairwise win/loss/draw matrix (Bradley–Terry MLE).
//!
//! Conventions honored (see `tetr_research` lib docs):
//! - **Arm-swap + CRN.** Every pair plays each seed from both chairs; chair luck cancels.
//! - **Death decides; the cap tiebreak is biased.** Games are made decisive by symmetric
//!   garbage *rain* (`rain_period`); a game that still reaches the ply cap with both alive
//!   is scored a DRAW, never by the (anti-defensive) net-attack tiebreak.
//! - **Determinism.** Seeds are drawn from a disjoint measurement region; every game is a
//!   pure function of `(spec, seed)`.
//! - **Self-bounding + checkpointed.** The tournament honors a wall-clock budget and
//!   appends each finished matchup to `pairs.csv`, so an interrupted run still yields a
//!   connected (if sparser) graph the fit can use.
//!
//! Run:  `cargo run --release -p tetr-research --example elo_pareto -- [compute|full] [out_dir] [budget_secs]`

use std::io::Write;
use std::time::{Duration, Instant};

use tetr_core::ai::eval::{Cc2Evaluator, Cc2Weights};
use tetr_core::ai::{BeamPlanner, Mind, SearchBudget, SearchState, think_to_completion};
use tetr_research::bots::BotSpec;
use tetr_research::fixtures::state_bank;
use tetr_research::seeds::seed_set_from;
use tetr_research::versus::{VersusFormat, VersusOutcome, VersusResult, evaluate_versus_format};

// ---- the grid: dense in the cheap/mid region where the knee lives ----------------
const WIDTHS: &[usize] = &[4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128];
const DEPTHS: &[u8] = &[2, 3, 4, 5, 6, 7, 9];

// ---- tournament knobs ------------------------------------------------------------
/// Symmetric rain forces near-equal strong bots into decisive games (mirror matches are
/// otherwise ≤6% lethal). One line to both every 4 plies — heavy enough to kill < cap.
const RAIN_PERIOD: u32 = 4;
/// Generous ply cap; a game reaching it with both alive is a DRAW (no net-attack tiebreak).
const MAX_PLIES: u32 = 150;
/// Seeds per chair direction; ×2 for the arm swap = games per matchup. Sized to the core
/// count so each direction's `par_iter` fills the machine in one wave (free precision).
const SEEDS_PER_PAIR: usize = 12;
/// A disjoint seed region for this measurement (above the campaign slabs, below FINAL).
const PARETO_REGION: usize = 1 << 62;
/// States in the compute-timing bank.
const COMPUTE_STATES: usize = 40;

#[derive(Clone, Copy)]
struct Cfg {
    wi: usize,
    di: usize,
    width: usize,
    depth: u8,
}

impl Cfg {
    fn label(&self) -> String {
        format!("w{}d{}", self.width, self.depth)
    }
    fn spec(&self) -> BotSpec {
        BotSpec::tp_beam(self.width, self.depth).cc2(Cc2Weights::attack_tuned())
    }
}

fn grid() -> Vec<Cfg> {
    let mut v = Vec::new();
    for (wi, &width) in WIDTHS.iter().enumerate() {
        for (di, &depth) in DEPTHS.iter().enumerate() {
            v.push(Cfg {
                wi,
                di,
                width,
                depth,
            });
        }
    }
    v
}

/// Median per-decision compute (ms) and the (board-independent, width-bounded) node count.
fn measure_compute(cfg: &Cfg, states: &[SearchState], eval: &Cc2Evaluator) -> (f64, u32) {
    let budget = SearchBudget::beam(cfg.depth);
    // warm the allocator / caches
    for s in states.iter().take(3) {
        let mut p = BeamPlanner::transposing(cfg.width);
        std::hint::black_box(think_to_completion(&mut p, s, eval, budget));
    }
    let mut us: Vec<u128> = Vec::with_capacity(states.len());
    let mut nodes = 0u32;
    for s in states {
        let mut p = BeamPlanner::transposing(cfg.width);
        let t = Instant::now();
        std::hint::black_box(think_to_completion(&mut p, s, eval, budget));
        us.push(t.elapsed().as_micros());
        nodes = nodes.max(p.nodes_expanded());
    }
    us.sort_unstable();
    (us[us.len() / 2] as f64 / 1000.0, nodes)
}

/// Score one matchup A-vs-B over arm-swapped seeds. Returns (a_wins, b_wins, draws) in
/// config-A / config-B terms, with death-decides and cap-reached-both-alive → draw.
fn run_match(a: &Cfg, b: &Cfg, seeds: &[u64], fmt: VersusFormat) -> (u32, u32, u32) {
    let fa = a.spec().factory();
    let fb = b.spec().factory();
    let fwd = evaluate_versus_format(&fa, &fb, seeds, fmt); // A in chair-A
    let rev = evaluate_versus_format(&fb, &fa, seeds, fmt); // B in chair-A (arm swap)
    let (mut aw, mut bw, mut dr) = (0u32, 0u32, 0u32);
    let mut tally = |o: &VersusOutcome, a_is_chair_a: bool| {
        // death decides; if neither topped out the cap was hit → honest draw.
        let res = if o.a_topped || o.b_topped {
            o.result
        } else {
            VersusResult::Draw
        };
        let config_a_won = match res {
            VersusResult::AWins => a_is_chair_a,
            VersusResult::BWins => !a_is_chair_a,
            VersusResult::Draw => {
                dr += 1;
                return;
            }
        };
        if config_a_won {
            aw += 1;
        } else {
            bw += 1;
        }
    };
    for o in &fwd.outcomes {
        tally(o, true);
    }
    for o in &rev.outcomes {
        tally(o, false);
    }
    (aw, bw, dr)
}

/// The match schedule: a connected grid lattice (each config vs its width/depth neighbors
/// and one long-range diagonal), cheapest-first so a truncated run keeps the dense
/// cheap/mid frontier and the strong configs stay connected through their neighbors.
fn schedule(g: &[Cfg]) -> Vec<(usize, usize)> {
    let idx = |wi: usize, di: usize| wi * DEPTHS.len() + di;
    let mut pairs: Vec<(usize, usize)> = Vec::new();
    let push = |x: usize, y: usize, pairs: &mut Vec<(usize, usize)>| {
        let (lo, hi) = if x < y { (x, y) } else { (y, x) };
        if lo != hi && !pairs.contains(&(lo, hi)) {
            pairs.push((lo, hi));
        }
    };
    for c in g {
        // width neighbor (same depth), depth neighbor (same width), and a diagonal.
        if c.wi + 1 < WIDTHS.len() {
            push(idx(c.wi, c.di), idx(c.wi + 1, c.di), &mut pairs);
        }
        if c.di + 1 < DEPTHS.len() {
            push(idx(c.wi, c.di), idx(c.wi, c.di + 1), &mut pairs);
        }
        if c.wi + 1 < WIDTHS.len() && c.di + 1 < DEPTHS.len() {
            push(idx(c.wi, c.di), idx(c.wi + 1, c.di + 1), &mut pairs);
        }
        // a two-step width hop ties the lattice together more rigidly for the Elo fit.
        if c.wi + 2 < WIDTHS.len() {
            push(idx(c.wi, c.di), idx(c.wi + 2, c.di), &mut pairs);
        }
    }
    // cheapest-first ≈ smaller (width*depth) sum first.
    let cost = |i: usize| g[i].width * g[i].depth as usize;
    pairs.sort_by_key(|&(x, y)| cost(x) + cost(y));
    pairs
}

fn main() -> std::io::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let mode = args.get(1).map(String::as_str).unwrap_or("full");
    let out_dir = args
        .get(2)
        .cloned()
        .unwrap_or_else(|| "analysis/elo-pareto".to_string());
    let budget_secs: u64 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(1800);
    std::fs::create_dir_all(&out_dir)?;

    let g = grid();
    let eval = Cc2Evaluator::new(Cc2Weights::attack_tuned());

    // ---- compute axis ----
    eprintln!(
        "measuring compute for {} configs over {COMPUTE_STATES} states ...",
        g.len()
    );
    let states = state_bank(
        COMPUTE_STATES,
        BotSpec::tp_beam(16, 4).cc2(Cc2Weights::attack_tuned()),
    );
    eprintln!("  (state bank: {} realistic mid-game boards)", states.len());
    let mut cfg_csv = std::fs::File::create(format!("{out_dir}/configs.csv"))?;
    writeln!(cfg_csv, "label,width,depth,compute_ms,nodes")?;
    let mut computed = Vec::new();
    for c in &g {
        let (ms, nodes) = measure_compute(c, &states, &eval);
        writeln!(
            cfg_csv,
            "{},{},{},{:.4},{}",
            c.label(),
            c.width,
            c.depth,
            ms,
            nodes
        )?;
        computed.push((c.label(), ms, nodes));
    }
    cfg_csv.flush()?;
    eprintln!(
        "  compute: cheapest {} = {:.3} ms, champion {} = {:.1} ms",
        computed.first().map(|x| x.0.clone()).unwrap_or_default(),
        computed.first().map(|x| x.1).unwrap_or(0.0),
        computed.last().map(|x| x.0.clone()).unwrap_or_default(),
        computed.last().map(|x| x.1).unwrap_or(0.0),
    );
    if mode == "compute" {
        eprintln!("wrote {out_dir}/configs.csv (compute-only mode)");
        return Ok(());
    }

    // ---- Elo axis: the versus tournament ----
    let seeds = seed_set_from(PARETO_REGION, SEEDS_PER_PAIR);
    let fmt = VersusFormat {
        max_plies: MAX_PLIES,
        rain_period: RAIN_PERIOD,
        sudden_death: false,
    };
    let pairs = schedule(&g);
    let games_per_pair = SEEDS_PER_PAIR * 2;
    eprintln!(
        "tournament: {} matchups x {} games (rain {}, cap {}), budget {}s ...",
        pairs.len(),
        games_per_pair,
        RAIN_PERIOD,
        MAX_PLIES,
        budget_secs
    );

    let mut pair_csv = std::fs::File::create(format!("{out_dir}/pairs.csv"))?;
    writeln!(pair_csv, "a,b,a_wins,b_wins,draws,games")?;
    let start = Instant::now();
    let deadline = Duration::from_secs(budget_secs);
    let mut done = 0usize;
    for &(x, y) in &pairs {
        if start.elapsed() > deadline {
            eprintln!(
                "  budget reached after {done}/{} matchups — stopping (partial, connected)",
                pairs.len()
            );
            break;
        }
        let (a, b) = (&g[x], &g[y]);
        let (aw, bw, dr) = run_match(a, b, &seeds, fmt);
        writeln!(
            pair_csv,
            "{},{},{},{},{},{}",
            a.label(),
            b.label(),
            aw,
            bw,
            dr,
            games_per_pair
        )?;
        pair_csv.flush()?; // checkpoint every matchup
        done += 1;
        if done.is_multiple_of(20) {
            eprintln!(
                "  {done}/{} matchups ({:.0}s elapsed)",
                pairs.len(),
                start.elapsed().as_secs_f64()
            );
        }
    }
    eprintln!(
        "wrote {out_dir}/configs.csv + {out_dir}/pairs.csv ({done} matchups, {:.0}s)",
        start.elapsed().as_secs_f64()
    );
    Ok(())
}
