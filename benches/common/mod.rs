//! Shared fixtures and helpers for the criterion bench suite.
//!
//! Every benchmark target (`benches/engine.rs`, `benches/ai.rs`, and any future
//! one) pulls its inputs from here via `mod common;`. Centralizing fixtures keeps
//! results **comparable** (the same board states feed every bench) and makes
//! adding a benchmark a matter of picking a scenario, not hand-rolling a board.
//!
//! # Design rules
//!
//! - **Public API only.** Benches compile as separate crates, so everything here
//!   goes through `tetr_online`'s public surface — no internal access, no test
//!   hooks beyond the `#[doc(hidden)]` `Engine::set_cell` the crate already
//!   exposes for fixture construction.
//! - **Deterministic.** Fixed seeds, no wall-clock, no `rand::rng()`. A bench is a
//!   pure function of its fixture, so run-to-run variance is measurement noise, not
//!   input drift. (The engine and AI cores guarantee this; we just hold the seeds
//!   fixed.)
//! - **A spread of complexity.** [`Scenario`] ranges from an empty board to a
//!   near-top-out stack so a benchmark reveals how cost *scales* with the board,
//!   not just one happy-path number.
//!
//! # Adding a fixture
//!
//! Add a variant to [`Scenario`], paint it in [`paint`], and name it in
//! [`Scenario::name`]. Every bench that loops over [`Scenario::ALL`] picks it up
//! for free.

// Each bench target uses a different subset of these helpers; the unused ones in
// any given target would otherwise warn.
#![allow(dead_code)]

use tetr_online::ai::{movegen, AiController, Handicap, SearchState};
use tetr_online::engine::{
    classify_t_spin, ActivePiece, Board, CellKind, Engine, EngineConfig, EngineEvent, InputFrame,
    LockOutcome, PieceType, TSpinKind,
};
use tetr_online::player::drive_engine;

/// Fixed RNG seed for the engine's seven-bag generator across all benches.
pub const ENGINE_SEED: u64 = 0xB007_5EED;

/// Fixed RNG seed for the bot's error-injection / tie-breaking across all benches.
pub const AI_SEED: u64 = 0x0A15_EED0;

/// The fixed simulation timestep the host steps the engine at (60 Hz). Benches use
/// it for `dt`-bearing frames so gravity/lock timing advance realistically.
pub const SIM_DT: f32 = 1.0 / 60.0;

/// A board state to benchmark across, ordered from cheapest to most constrained.
///
/// Looping a benchmark over [`Scenario::ALL`] turns one `bench_function` call into
/// a family of measurements that show how the operation scales with board
/// occupancy, holes, and remaining headroom.
#[derive(Clone, Copy, Debug)]
pub enum Scenario {
    /// Empty board, freshly spawned piece — the richest movegen case (every
    /// column × rotation is reachable).
    Empty,
    /// Three flat rows with a one-wide well — a typical early-game surface.
    LightStack,
    /// Six rows with several buried holes — exercises hole / transition counting
    /// and forces movegen to navigate an uneven surface.
    HoleyStack,
    /// A tall stack with a single deep well and little headroom — the constrained
    /// case where reachability is tight.
    NearTopOut,
}

impl Scenario {
    /// Every scenario, cheapest first. Benches iterate this.
    pub const ALL: [Scenario; 4] = [
        Scenario::Empty,
        Scenario::LightStack,
        Scenario::HoleyStack,
        Scenario::NearTopOut,
    ];

    /// Stable, filename-safe label used as the criterion benchmark-id parameter.
    pub fn name(self) -> &'static str {
        match self {
            Scenario::Empty => "empty",
            Scenario::LightStack => "light_stack",
            Scenario::HoleyStack => "holey_stack",
            Scenario::NearTopOut => "near_top_out",
        }
    }
}

/// A fresh engine at the default config and [`ENGINE_SEED`], stepped once so the
/// first piece has spawned (a snapshot before the first step has no active piece).
pub fn fresh_engine() -> Engine {
    let mut engine = Engine::new(EngineConfig::default(), ENGINE_SEED);
    // A zero-input frame: spawns the first piece and does nothing else.
    engine.step(InputFrame::default());
    engine
}

/// An engine painted into the given [`Scenario`]'s board state (piece spawned).
///
/// Locked cells are stamped directly with the crate's `#[doc(hidden)]`
/// `Engine::set_cell`. Rows always leave a gap so no fixture contains a "full"
/// row that the engine would have cleared — the boards stay rule-plausible.
pub fn scenario_engine(scenario: Scenario) -> Engine {
    let mut engine = fresh_engine();
    paint(&mut engine, scenario);
    engine
}

/// A [`SearchState`] for `scenario` with `batches` one-line garbage batches
/// queued against the player — the pressured-fixture variant: the per-child
/// garbage transition (cancel on clears, capped rising on clear-less locks)
/// only runs when pending is non-empty, so the plain fixtures never exercise
/// its cost.
pub fn pressured_search_state(scenario: Scenario, batches: u32) -> SearchState {
    let mut engine = scenario_engine(scenario);
    for _ in 0..batches {
        engine.queue_garbage(1);
    }
    SearchState::from_snapshot(&engine.snapshot()).expect("scenario fixture has an active piece")
}

/// A [`SearchState`] for the given scenario — the universal fixture.
///
/// `SearchState` exposes public `board` and `active` fields, so this one value
/// feeds the engine primitives, the movegen, the evaluator, and the planner alike.
pub fn search_state(scenario: Scenario) -> SearchState {
    let snapshot = scenario_engine(scenario).snapshot();
    SearchState::from_snapshot(&snapshot).expect("scenario fixture has an active piece")
}

/// A spawn-pose factory matching the default board geometry, for the hold-aware
/// [`movegen::generate_with_hold`] (which needs to know where a swapped-in piece
/// would spawn). Captures the geometry by value so it is `'static`.
pub fn spawner() -> impl Fn(PieceType) -> ActivePiece {
    let cfg = EngineConfig::default();
    let (width, visible) = (cfg.board_width, cfg.visible_height);
    move |piece_type| movegen::spawn_piece(piece_type, width, visible)
}

/// The first reachable placement for a state, in movegen's canonical order. Used
/// as the unit of work for the primitive benches (lock / classify / evaluate).
pub fn first_placement(state: &SearchState) -> movegen::Placement {
    movegen::generate(&state.board, &state.active)
        .into_iter()
        .next()
        .expect("scenario has at least one reachable placement")
}

/// A realistic `(lock outcome, post-lock board, t-spin)` triple for the evaluator
/// bench — produced by actually simulating the scenario's first placement through
/// the same primitives the planner uses.
pub fn first_locked(scenario: Scenario) -> (LockOutcome, Board, Option<TSpinKind>) {
    let state = search_state(scenario);
    let placement = first_placement(&state);
    let mut board = state.board;
    let t_spin = classify_t_spin(&placement.piece, &board);
    let lock = board.lock_piece(&placement.piece);
    (lock, board.to_array2d(), t_spin)
}

/// Play a flawless, seeded bot against a fresh engine until it has placed `target`
/// pieces (or tops out / hits the frame cap), returning the pieces actually
/// placed. The flagship end-to-end throughput driver: deterministic given the
/// seeds, so it is a stable macro-benchmark for tuning evaluator weights or search
/// depth.
pub fn play_pieces(target: usize) -> usize {
    let mut engine = Engine::new(EngineConfig::default(), ENGINE_SEED);
    let mut bot = AiController::new(Handicap::perfect(), AI_SEED);

    let mut placed = 0usize;
    // Generous safety cap: a piece resolves in well under 200 frames of pulses.
    let frame_cap = target.saturating_mul(200).max(1_000);
    for _ in 0..frame_cap {
        for event in drive_engine(&mut engine, &mut bot) {
            match event {
                EngineEvent::Locked { .. } => placed += 1,
                EngineEvent::GameOver { .. } => return placed,
                _ => {}
            }
        }
        if placed >= target {
            break;
        }
    }
    placed
}

/// Stamp the locked-cell pattern for a scenario onto a freshly-spawned engine.
///
/// Coordinates are engine-space: origin bottom-left, `x in 0..10`, `y` increasing
/// upward, the visible field `y in 0..20` over a 20-row spawn buffer above it. The
/// spawned active piece sits high (around `y == 18`), so painting low rows never
/// collides with it.
fn paint(engine: &mut Engine, scenario: Scenario) {
    // Any locked piece-type renders identically for board-shape purposes.
    let fill = |engine: &mut Engine, x: isize, y: isize| {
        engine.set_cell(x, y, CellKind::Some(PieceType::I));
    };
    let carve = |engine: &mut Engine, x: isize, y: isize| {
        engine.set_cell(x, y, CellKind::None);
    };

    match scenario {
        Scenario::Empty => {}

        Scenario::LightStack => {
            // Three flat rows, columns 0..9 filled, a one-wide well at x = 9.
            for y in 0..3 {
                for x in 0..9 {
                    fill(engine, x, y);
                }
            }
        }

        Scenario::HoleyStack => {
            // Six solid rows...
            for y in 0..6 {
                for x in 0..10 {
                    fill(engine, x, y);
                }
            }
            // ...with the right column open (so no row is full)...
            for y in 0..6 {
                carve(engine, 9, y);
            }
            // ...and a few buried holes for the hole/transition features to find.
            carve(engine, 2, 1);
            carve(engine, 5, 2);
            carve(engine, 7, 0);
        }

        Scenario::NearTopOut => {
            // Columns 1..10 stacked 18 rows high, leaving a deep well at x = 0 and
            // only a sliver of headroom below the spawn rows.
            for y in 0..18 {
                for x in 1..10 {
                    fill(engine, x, y);
                }
            }
        }
    }
}
