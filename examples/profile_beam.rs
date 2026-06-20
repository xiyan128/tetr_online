//! Throwaway perf evidence harness (not shipped; delete after the review).
//!
//! Drives each interactive planner like the in-game *sliced* venue does — re-root
//! then spend a fixed 16-node quantum per poll (the wasm worst-case quantum) — and
//! records the cost of every individual poll across a full decision. The max poll is
//! the frame-blocking burst the user feels as lag.
//!
//! Also the flamegraph/`sample` target: `cargo run --release --example profile_beam profile`
//! loops the champion decision so an external sampler can attach.
//!
//! `cargo run --release --example profile_beam`         → the evidence table
//! `cargo run --release --example profile_beam profile` → ~12s busy-loop for sampling

use std::time::Instant;

use tetr_online::ai::{
    AiController, BeamPlanner, BestFirstPlanner, Cc2Evaluator, Cc2Weights, Evaluator, Handicap,
    Mind, SearchBudget, SearchPolicy, SearchState, ThinkProgress, think_to_completion,
};
use tetr_online::engine::{CellKind, Engine, EngineConfig, EngineEvent, InputFrame, PieceType};
use tetr_online::player::drive_engine;

const SEED: u64 = 0xB007_5EED;
/// The per-poll node quanta the sliced runner uses in-game (one poll per
/// FixedUpdate tick): 32 on native, 16 in-browser.
const NATIVE_QUANTUM: u32 = 32;
const WASM_QUANTUM: u32 = 16;

fn make_state(stack: bool) -> SearchState {
    let mut e = Engine::new(EngineConfig::default(), SEED);
    e.step(InputFrame::default()); // spawn the first piece
    if stack {
        // A 6-row holey stack — mirrors the `HoleyStack` bench fixture.
        for y in 0..6 {
            for x in 0..10 {
                e.set_cell(x, y, CellKind::Some(PieceType::I));
            }
        }
        for y in 0..6 {
            e.set_cell(9, y, CellKind::None);
        }
        e.set_cell(2, 1, CellKind::None);
        e.set_cell(5, 2, CellKind::None);
        e.set_cell(7, 0, CellKind::None);
    }
    SearchState::from_snapshot(&e.snapshot()).expect("fixture has an active piece")
}

/// Drive one full decision poll-by-poll at the in-game quantum, metering the node
/// budget exactly as `think_to_completion` does. Returns `(per-poll ms, per-poll
/// node delta)` for every poll — so `.max()` is the worst single-frame burst.
fn poll_by_poll(
    mind: &mut dyn Mind,
    state: &SearchState,
    eval: &dyn Evaluator,
    budget: SearchBudget,
    quantum: u32,
) -> Vec<(f64, u32)> {
    let mut polls = Vec::new();
    mind.reroot(state, eval, budget.max_depth);
    loop {
        let remaining = match budget.nodes {
            0 => u32::MAX,
            cap => cap.saturating_sub(mind.nodes_expanded()),
        };
        if remaining == 0 {
            break;
        }
        let before = mind.nodes_expanded();
        let t = Instant::now();
        let progress = mind.think(quantum.min(remaining), eval);
        let ms = t.elapsed().as_secs_f64() * 1e3;
        polls.push((ms, mind.nodes_expanded() - before));
        if progress == ThinkProgress::Exhausted {
            break;
        }
    }
    polls
}

fn report(
    label: &str,
    state: &SearchState,
    eval: &dyn Evaluator,
    budget: SearchBudget,
    fresh: &dyn Fn() -> Box<dyn Mind>,
    quantum: u32,
) {
    // Best per-poll wall-clock across a few passes (min = least noise); node counts
    // are deterministic so one pass fixes them.
    let mut best_polls: Option<Vec<(f64, u32)>> = None;
    let mut total_ms_min = f64::MAX;
    for _ in 0..7 {
        let mut m = fresh();
        let t = Instant::now();
        let polls = poll_by_poll(m.as_mut(), state, eval, budget, quantum);
        let total = t.elapsed().as_secs_f64() * 1e3;
        if total < total_ms_min {
            total_ms_min = total;
            best_polls = Some(polls);
        }
    }
    let polls = best_polls.unwrap();
    let n_polls = polls.len();
    let total_nodes: u32 = polls.iter().map(|&(_, n)| n).sum();
    let (max_ms, max_nodes) = polls
        .iter()
        .copied()
        .fold((0.0f64, 0u32), |(mm, mn), (ms, n)| (mm.max(ms), mn.max(n)));
    // The sliced venue runs ONE poll per FixedUpdate tick (one per render frame at
    // 60Hz), so `polls` == frames-to-decide and the wall-clock to commit one piece
    // is `polls / 60`s even when the actual compute (`total`) is far smaller — the
    // search is throttled by the frame loop, not the CPU.
    let frames_wall_ms = n_polls as f64 * (1000.0 / 60.0);
    println!(
        "{label:<26} q={quantum:<2} polls={n_polls:>3}  compute={total_ms_min:>6.1}ms ({total_nodes:>5} nodes)  worst poll {max_ms:>5.2}ms ({max_nodes:>4} nodes)  => {frames_wall_ms:>6.0}ms/piece in-game ({:.1}/s)",
        1000.0 / frames_wall_ms,
    );
}

/// Check whether planners honor the in-game quantum: one think at q=1 should expand
/// fewer nodes than one think at the 16-node wasm slice.
fn quantum_proof(state: &SearchState, eval: &dyn Evaluator) {
    println!("\n== Quantum-honoring proof (nodes expanded by ONE think() call) ==");
    for (name, mut mind, depth) in [
        (
            "best-first",
            Box::new(BestFirstPlanner::new()) as Box<dyn Mind>,
            6u8,
        ),
        (
            "beam w32",
            Box::new(BeamPlanner::transposing(32)) as Box<dyn Mind>,
            4,
        ),
        (
            "beam w128",
            Box::new(BeamPlanner::transposing(128)) as Box<dyn Mind>,
            9,
        ),
    ] {
        mind.reroot(state, eval, depth);
        mind.think(1, eval);
        let n_small = mind.nodes_expanded();
        let mut big = match name {
            "best-first" => Box::new(BestFirstPlanner::new()) as Box<dyn Mind>,
            "beam w32" => Box::new(BeamPlanner::transposing(32)) as Box<dyn Mind>,
            _ => Box::new(BeamPlanner::transposing(128)) as Box<dyn Mind>,
        };
        big.reroot(state, eval, depth);
        big.think(WASM_QUANTUM, eval);
        let n_big = big.nodes_expanded();
        let verdict = if n_small == n_big {
            "IGNORES quantum"
        } else {
            "honors quantum"
        };
        println!(
            "  {name:<12} think(1)={n_small:>5} nodes   think({WASM_QUANTUM})={n_big:>5} nodes   -> {verdict}"
        );
    }
}

/// Drive the APP Champion (w128/d9 TP beam) through a real headless game and report
/// per-piece decision time. Blocking venue (`with_policy`) + zero reaction = pure
/// compute, no frame pacing: the full decision lands inline on one poll per piece,
/// so total-wall / pieces ≈ mean decision time, and the slowest poll ≈ the worst
/// piece. This is the "outside Bevy, how long to place one piece" number.
fn champion_game(target: usize) {
    let h = Handicap::perfect(); // reaction = 0, imperfection = 0 → pure search cost
    let policy = SearchPolicy::new(
        Box::new(BeamPlanner::transposing(128)),
        Box::new(Cc2Evaluator::new(Cc2Weights::attack_tuned())),
        SearchBudget::beam(9),
        h.imperfection,
        SEED,
    );
    let mut engine = Engine::new(EngineConfig::default(), SEED);
    let mut bot = AiController::with_policy(Box::new(policy), h.reaction);

    let mut placed = 0usize;
    let mut decisions: Vec<f64> = Vec::new(); // ms of the expensive (decision) frames
    let start = Instant::now();
    'game: for _ in 0..target * 400 + 1000 {
        let t = Instant::now();
        let events = drive_engine(&mut engine, &mut bot);
        let ms = t.elapsed().as_secs_f64() * 1e3;
        if ms > 1.0 {
            decisions.push(ms); // ~one full search per piece; execution frames are ~free
        }
        for e in &events {
            match e {
                EngineEvent::Locked { .. } => placed += 1,
                EngineEvent::GameOver { .. } => break 'game,
                _ => {}
            }
        }
        if placed >= target {
            break;
        }
    }
    let total = start.elapsed().as_secs_f64() * 1e3;
    decisions.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let pct = |p: f64| decisions[((decisions.len() as f64 * p) as usize).min(decisions.len() - 1)];
    println!("== APP Champion (w128/d9 TP beam) — headless, blocking, {placed} pieces ==");
    println!(
        "  mean wall / piece : {:.1} ms",
        total / placed.max(1) as f64
    );
    println!(
        "  decision frame    : p50 {:.1} ms   p90 {:.1} ms   max {:.1} ms",
        pct(0.5),
        pct(0.9),
        pct(0.99)
    );
    println!(
        "  total game        : {:.0} ms for {placed} pieces  ({:.1} pieces/s)",
        total,
        placed as f64 / (total / 1000.0)
    );
    println!("  (native release; wasm runs ~2-4x slower)");
}

fn main() {
    let eval = Cc2Evaluator::new(Cc2Weights::attack_tuned());
    if std::env::args().any(|a| a == "game") {
        champion_game(60);
        return;
    }
    let profile_mode = std::env::args().any(|a| a == "profile");

    if profile_mode {
        // Busy-loop the champion decision so an external sampler can attach.
        let state = make_state(true);
        let start = Instant::now();
        let mut iters = 0u64;
        while start.elapsed().as_secs() < 12 {
            std::hint::black_box(think_to_completion(
                &mut BeamPlanner::transposing(128),
                &state,
                &eval,
                SearchBudget::beam(9),
            ));
            iters += 1;
        }
        eprintln!("champion decisions completed: {iters}");
        return;
    }

    for (sname, stack) in [("empty", false), ("holey6", true)] {
        let state = make_state(stack);
        println!("\n== Scenario: {sname} — NATIVE quantum 32, one poll per 60Hz frame ==");
        report(
            "best-first 192/d6",
            &state,
            &eval,
            SearchBudget::best_first(192, 6),
            &|| Box::new(BestFirstPlanner::new()),
            NATIVE_QUANTUM,
        );
        report(
            "beam w32/d4 (interactive)",
            &state,
            &eval,
            SearchBudget::beam(4),
            &|| Box::new(BeamPlanner::transposing(32)),
            NATIVE_QUANTUM,
        );
        report(
            "beam w128/d9 (champion)",
            &state,
            &eval,
            SearchBudget::beam(9),
            &|| Box::new(BeamPlanner::transposing(128)),
            NATIVE_QUANTUM,
        );
        report(
            "beam w128/d9 @ wasm q16",
            &state,
            &eval,
            SearchBudget::beam(9),
            &|| Box::new(BeamPlanner::transposing(128)),
            WASM_QUANTUM,
        );
    }

    quantum_proof(&make_state(true), &eval);
    println!("\n(16.67ms = one 60Hz frame; native release. wasm ~2-4x slower.)");
}
