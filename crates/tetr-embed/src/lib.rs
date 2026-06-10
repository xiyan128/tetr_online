//! `tetr-embed` — the headless wasm surface that powers the embeddable component.
//!
//! It wraps the engine-agnostic [`tetr_core`] crate (engine + AI + keyboard) in a
//! single [`Game`] handle a browser drives at a fixed 60 Hz. Bevy is nowhere in the
//! graph, so the wasm is a few hundred KB instead of the game's ~14 MB.
//!
//! # The contract the JS side relies on
//!
//! - Time is advanced by [`Game::tick`], which runs the engine on a **fixed 60 Hz
//!   accumulator**. This is load-bearing: the AI controller integrates a `1/60 s`
//!   slice per poll (see `tetr_core::ai::controller`), so it must be driven at that
//!   cadence, never with a raw `requestAnimationFrame` delta.
//! - In **AI mode** the engine is driven by an [`AiController`] (autoplay). In
//!   **human mode** it is driven by the [`KeyboardController`], fed a
//!   [`RawKeyboardFrame`] built from the DOM key state the JS reports through
//!   [`Game::key_down`] / [`Game::key_up`]. Swapping modes is the whole "click to
//!   take over / release to resume" mechanic — same engine, different controller.
//! - Render reads ([`Game::board_cells`] etc.) return flat typed arrays of the
//!   *cached* snapshot taken at the end of the last `tick`, so a frame is a handful
//!   of cheap copies, no per-getter re-snapshot.
//!
//! # Determinism
//!
//! A `Game` is a pure function of its `(seed, handicap)` and the sequence of inputs
//! it is driven with. In AI mode, two `Game`s built with the same `seed`, `reaction_ms`,
//! and `imperfection` and ticked with the same `dt` sequence produce byte-identical
//! games — the engine's 7-bag and the AI's RNG are both seeded, never wall-clock. Note
//! `reaction_ms` and `imperfection` are part of that input: same `seed` but a different
//! handicap is a *different* game, not the same board played better or worse.
//!
//! # Panic safety
//!
//! No public method panics on any JS-supplied input. A panic across the wasm boundary
//! aborts the entire module — taking down every `Game` on the page — so every numeric
//! argument from JS is sanitized at this boundary: non-finite `imperfection`/`dt`
//! values are neutralized (see [`Game::new`] / [`Game::tick`]) and out-of-range action
//! indices are ignored rather than indexed.

use core::time::Duration;

use tetr_core::ai::{AiController, Handicap};
use tetr_core::engine::{
    Engine, EngineConfig, EngineEvent, EngineSnapshot, SnapshotCell,
};
use tetr_core::player::{drive_engine, KeyboardController, RawKeyboardFrame};
use wasm_bindgen::prelude::*;

/// The fixed simulation slice: 60 Hz, matching the engine driver and the AI
/// controller's per-poll `dt` integration.
const SIM_DT: f32 = 1.0 / 60.0;

/// Cap fixed steps per `tick` so a long pause (backgrounded tab) can't trigger a
/// "spiral of death" catch-up. ~0.25 s of simulation at most.
const MAX_STEPS_PER_TICK: u32 = 15;

// Action bit flags. JS maps a `KeyboardEvent.code` to an action index (0..=7) and
// calls `key_down`/`key_up`; we keep the held set as a bitmask and derive
// `just_pressed` as the bits newly set since the previous fixed step.
const A_LEFT: u16 = 1 << 0;
const A_RIGHT: u16 = 1 << 1;
const A_SOFT: u16 = 1 << 2;
const A_HARD: u16 = 1 << 3;
const A_CW: u16 = 1 << 4;
const A_CCW: u16 = 1 << 5;
const A_HOLD: u16 = 1 << 6;
const A_PAUSE: u16 = 1 << 7;

/// Map a JS action index to its bit, or `None` if out of range.
fn action_bit(action: u8) -> Option<u16> {
    match action {
        0 => Some(A_LEFT),
        1 => Some(A_RIGHT),
        2 => Some(A_SOFT),
        3 => Some(A_HARD),
        4 => Some(A_CW),
        5 => Some(A_CCW),
        6 => Some(A_HOLD),
        7 => Some(A_PAUSE),
        _ => None,
    }
}

/// Pack cells into a flat `[x, y, pieceIndex, …]` array for the canvas renderer.
/// The piece index is the engine's guideline colour order (`I,O,T,S,Z,J,L`) via
/// [`PieceType::render_index`], so the JS palette is a 7-entry array indexed by it.
fn pack_cells(cells: &[SnapshotCell]) -> Vec<i32> {
    let mut out = Vec::with_capacity(cells.len() * 3);
    for c in cells {
        out.push(c.x as i32);
        out.push(c.y as i32);
        out.push(c.piece_type.render_index() as i32);
    }
    out
}

/// A compact tag for an engine event, surfaced from `tick` so the renderer can fire
/// effects: `1` lock (no clear), `2` line clear, `3` game over, `4` hold.
fn event_tag(e: &EngineEvent) -> Option<u8> {
    match e {
        EngineEvent::Locked { lines_cleared, .. } => Some(if *lines_cleared > 0 { 2 } else { 1 }),
        EngineEvent::GameOver { .. } => Some(3),
        EngineEvent::Held { .. } => Some(4),
        _ => None,
    }
}

/// Build the AI controller for a fresh game — the strongest shipped bot
/// ([`AiController::attack`]: best-first search over the tuned CC2 attack
/// evaluator, the same brain as the game's "Best-First Attack" model) at the
/// given handicap (reaction delay + imperfection), which still defines the
/// embed's "click to take over" / beatability contract: the dials degrade even
/// this brain into a beatable opponent, and at a fixed handicap the run stays
/// byte-identical per seed (the search is deterministic — no RNG, no clock).
fn make_ai(handicap: Handicap, seed: u32) -> AiController {
    AiController::attack(handicap, ai_seed(seed))
}

/// Which controller is currently driving the engine.
enum Mode {
    /// Autoplay: the [`AiController`] drives.
    Ai,
    /// Human take-over: the [`KeyboardController`] drives from DOM key state.
    Human,
}

/// A single embeddable game: an [`Engine`] plus the two controllers, a fixed-step
/// accumulator, and DOM key state. Construct with [`Game::new`], advance with
/// [`Game::tick`], read with the snapshot getters.
#[wasm_bindgen]
pub struct Game {
    engine: Engine,
    ai: AiController,
    keyboard: KeyboardController,
    handicap: Handicap,
    mode: Mode,
    /// Currently-held action bits.
    pressed: u16,
    /// Held bits as of the previous fixed step (for `just_pressed` edges).
    prev_pressed: u16,
    /// Leftover real time not yet consumed by a fixed step.
    acc: f32,
    /// Snapshot cached at the end of the last `tick` — what every getter reads.
    snap: EngineSnapshot,
}

#[wasm_bindgen]
impl Game {
    /// Create a game in autoplay (AI) mode.
    ///
    /// - `seed`: seeds the engine's piece sequence (and, derived from it, the AI's
    ///   RNG). Fully determines the run together with the handicap (see the module
    ///   "Determinism" note); any `u32` is valid.
    /// - `reaction_ms`: the AI's reaction delay in milliseconds before it acts on a
    ///   new piece. `0` is an instant bot.
    /// - `imperfection`: the AI's error rate in `0.0..=1.0` (`0` = flawless). Values
    ///   outside the range are clamped; non-finite values are treated as `0`.
    #[wasm_bindgen(constructor)]
    pub fn new(seed: u32, reaction_ms: u32, imperfection: f32) -> Game {
        #[cfg(target_arch = "wasm32")]
        console_error_panic_hook::set_once();

        let handicap = Handicap {
            reaction: Duration::from_millis(reaction_ms as u64),
            // Sanitize at the FFI boundary: `imperfection` is a probability the AI
            // feeds to `rand`, which panics on a NaN/out-of-range `p` — and a panic
            // over the wasm boundary aborts the whole module (every board on the
            // page). `f32::clamp` alone does NOT fix NaN (it propagates), so map any
            // non-finite value to 0 first.
            imperfection: sanitize_unit(imperfection),
        };
        let (engine, snap) = fresh_engine(seed);
        Game {
            engine,
            ai: make_ai(handicap, seed),
            keyboard: KeyboardController::default(),
            handicap,
            mode: Mode::Ai,
            pressed: 0,
            prev_pressed: 0,
            acc: 0.0,
            snap,
        }
    }

    /// Restart with a fresh `seed`, preserving the handicap and the current mode.
    pub fn reset(&mut self, seed: u32) {
        let (engine, snap) = fresh_engine(seed);
        self.engine = engine;
        self.snap = snap;
        self.ai = make_ai(self.handicap, seed);
        self.keyboard = KeyboardController::default();
        self.pressed = 0;
        self.prev_pressed = 0;
        self.acc = 0.0;
    }

    /// Advance real time by `dt_seconds`, running the engine on a fixed 60 Hz
    /// accumulator. Returns the event tags (see [`event_tag`]) emitted this tick so
    /// the renderer can fire line-clear / game-over effects.
    pub fn tick(&mut self, dt_seconds: f32) -> Vec<u8> {
        let mut tags = Vec::new();
        // Clamp the incoming delta so a backgrounded tab doesn't request a huge
        // catch-up; MAX_STEPS_PER_TICK is the hard backstop. A non-finite `dt` (a NaN
        // from a bad `performance.now()` delta) must be dropped to 0, not clamped:
        // `NaN.clamp(..)` yields NaN, which would poison `acc` permanently (every
        // `acc >= SIM_DT` then false) and silently freeze the game forever.
        let dt = if dt_seconds.is_finite() { dt_seconds.clamp(0.0, 0.25) } else { 0.0 };
        self.acc += dt;
        let mut steps = 0;
        while self.acc >= SIM_DT && steps < MAX_STEPS_PER_TICK {
            self.acc -= SIM_DT;
            steps += 1;
            let events = match self.mode {
                Mode::Ai => drive_engine(&mut self.engine, &mut self.ai),
                Mode::Human => {
                    let frame = self.raw_frame(SIM_DT);
                    self.keyboard.set_input(frame);
                    let events = drive_engine(&mut self.engine, &mut self.keyboard);
                    // One fixed step consumed this key state: collapse the edges so a
                    // held key is "just pressed" for exactly one step.
                    self.prev_pressed = self.pressed;
                    events
                }
            };
            for e in &events {
                if let Some(tag) = event_tag(e) {
                    tags.push(tag);
                }
            }
        }
        self.snap = self.engine.snapshot();
        tags
    }

    /// Switch to autoplay (AI drives). Also clears the held-key state, so the mode is
    /// the single source of truth: no key bits captured during take-over linger while
    /// the AI plays (symmetric with [`set_mode_human`](Self::set_mode_human)).
    pub fn set_mode_ai(&mut self) {
        self.mode = Mode::Ai;
        self.pressed = 0;
        self.prev_pressed = 0;
    }

    /// Switch to human take-over. Resets the held-key state and the DAS machine so a
    /// stale charge from a previous take-over never carries in.
    pub fn set_mode_human(&mut self) {
        self.mode = Mode::Human;
        self.pressed = 0;
        self.prev_pressed = 0;
        self.keyboard = KeyboardController::default();
    }

    /// Whether a human is currently in control.
    pub fn is_human(&self) -> bool {
        matches!(self.mode, Mode::Human)
    }

    /// Report a key press (action index 0..=7; see the module's action bits).
    pub fn key_down(&mut self, action: u8) {
        if let Some(bit) = action_bit(action) {
            self.pressed |= bit;
        }
    }

    /// Report a key release.
    pub fn key_up(&mut self, action: u8) {
        if let Some(bit) = action_bit(action) {
            self.pressed &= !bit;
        }
    }

    // ---- Snapshot reads (all from the cached `snap`) ----
    //
    // The allocating getters carry `#[must_use]`: each builds a fresh Vec that
    // wasm-bindgen copies across the boundary, so a call whose result is dropped is
    // pure wasted work on the per-frame hot path.

    /// Locked board cells, flat `[x, y, pieceIndex, …]`.
    #[must_use]
    pub fn board_cells(&self) -> Vec<i32> {
        pack_cells(&self.snap.board_cells)
    }

    /// The falling piece's cells, flat `[x, y, pieceIndex, …]` (empty if none).
    #[must_use]
    pub fn active_cells(&self) -> Vec<i32> {
        self.snap
            .active
            .as_ref()
            .map(|a| pack_cells(&a.cells))
            .unwrap_or_default()
    }

    /// The ghost (hard-drop preview) cells, flat `[x, y, pieceIndex, …]`.
    #[must_use]
    pub fn ghost_cells(&self) -> Vec<i32> {
        pack_cells(&self.snap.ghost_cells)
    }

    /// The next-queue piece indices.
    #[must_use]
    pub fn next_queue(&self) -> Vec<u8> {
        self.snap.next_queue.iter().map(|p| p.render_index()).collect()
    }

    /// The held piece index, or `-1` if the hold slot is empty.
    pub fn hold(&self) -> i32 {
        self.snap.hold.map_or(-1, |p| i32::from(p.render_index()))
    }

    /// The active piece index, or `-1` if there is none.
    pub fn active_piece(&self) -> i32 {
        self.snap
            .active
            .as_ref()
            .map_or(-1, |a| i32::from(a.piece_type.render_index()))
    }

    /// How far the active piece is through its lock delay, `0.0..=1.0` (drives the
    /// landing/lock pulse). `0` when airborne or absent.
    pub fn active_lock_fraction(&self) -> f32 {
        self.snap
            .active
            .as_ref()
            .map_or(0.0, |a| a.lock_timer_fraction)
    }

    /// Current score.
    pub fn score(&self) -> u32 {
        self.snap.score as u32
    }

    /// Lines cleared so far.
    pub fn lines(&self) -> u32 {
        self.snap.lines as u32
    }

    /// Current level.
    pub fn level(&self) -> u32 {
        self.snap.level as u32
    }

    /// Whether a Back-to-Back chain is active.
    pub fn back_to_back(&self) -> bool {
        self.snap.back_to_back_active
    }

    /// Whether the game has ended (top-out). The JS loop reads this to auto-restart
    /// the autoplay animation.
    pub fn game_over(&self) -> bool {
        self.snap.game_over.is_some()
    }

    /// Board width in cells.
    pub fn board_width(&self) -> u32 {
        self.snap.config.board_width as u32
    }

    /// Visible board height in cells (the buffer above is hidden).
    pub fn visible_height(&self) -> u32 {
        self.snap.config.visible_height as u32
    }

    /// Build a [`RawKeyboardFrame`] for one fixed step from the current/previous held
    /// bits, deriving the just-pressed edges.
    fn raw_frame(&self, dt: f32) -> RawKeyboardFrame {
        let now = self.pressed;
        let just = now & !self.prev_pressed;
        RawKeyboardFrame {
            dt_seconds: dt,
            left_pressed: now & A_LEFT != 0,
            right_pressed: now & A_RIGHT != 0,
            left_just_pressed: just & A_LEFT != 0,
            right_just_pressed: just & A_RIGHT != 0,
            soft_drop: now & A_SOFT != 0,
            hard_drop_just_pressed: just & A_HARD != 0,
            rotate_cw_just_pressed: just & A_CW != 0,
            rotate_ccw_just_pressed: just & A_CCW != 0,
            hold_just_pressed: just & A_HOLD != 0,
            pause_just_pressed: just & A_PAUSE != 0,
        }
    }
}

/// Derive the AI's RNG seed from the engine seed, kept distinct so the two streams
/// never align (mirrors `tetr_core::ai::DEFAULT_AI_SEED`'s intent).
fn ai_seed(seed: u32) -> u64 {
    (seed as u64) ^ tetr_core::ai::DEFAULT_AI_SEED
}

/// Clamp a value to `0.0..=1.0`, mapping any non-finite input (NaN/±Inf) to `0.0`.
/// `f32::clamp` propagates NaN, so it cannot be used alone to sanitize a probability.
fn sanitize_unit(v: f32) -> f32 {
    if v.is_finite() {
        v.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

/// Build a fresh default engine for `seed` and spawn its first piece, so the very
/// first render (before any `tick`) is non-empty. Returns the engine and its initial
/// snapshot. Shared by [`Game::new`] and [`Game::reset`] so the construction ritual
/// lives in exactly one place.
fn fresh_engine(seed: u32) -> (Engine, EngineSnapshot) {
    let mut engine = Engine::new(EngineConfig::default(), seed as u64);
    engine.step(tetr_core::engine::InputFrame::default());
    let snap = engine.snapshot();
    (engine, snap)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A NaN/out-of-range `imperfection` must not panic. The AI feeds it to `rand`,
    /// which panics on a NaN probability — and a panic over the wasm boundary aborts
    /// the whole module, so the sanitization at the FFI boundary is load-bearing.
    #[test]
    fn nan_imperfection_does_not_panic() {
        for bad in [f32::NAN, f32::INFINITY, -1.0, 2.0] {
            let mut g = Game::new(7, 0, bad);
            for _ in 0..300 {
                g.tick(1.0 / 60.0); // drives the AI's imperfection sampling
            }
        }
    }

    /// A non-finite `dt` must be dropped, not clamped: `NaN.clamp(..)` would poison
    /// `acc` permanently and silently freeze the sim. After a NaN tick, a normal tick
    /// sequence must still advance the game.
    #[test]
    fn nan_dt_does_not_freeze_the_sim() {
        let mut g = Game::new(7, 0, 0.0); // flawless AI: deterministic progress
        g.tick(f32::NAN);
        g.tick(f32::INFINITY);
        for _ in 0..600 {
            g.tick(1.0 / 60.0); // ~10s of real time
        }
        assert!(g.lines() > 0, "sim froze after a non-finite dt");
    }

    /// `reset` mid-game with a non-finite handicap path is reached via the preserved
    /// handicap; ensure a flawless game restarts cleanly and stays deterministic.
    #[test]
    fn reset_restarts_deterministically() {
        let mut a = Game::new(42, 0, 0.0);
        for _ in 0..300 {
            a.tick(1.0 / 60.0);
        }
        a.reset(7);
        let mut b = Game::new(7, 0, 0.0);
        for _ in 0..300 {
            a.tick(1.0 / 60.0);
            b.tick(1.0 / 60.0);
        }
        assert_eq!(a.score(), b.score(), "reset(seed) must match new(seed)");
        assert_eq!(a.lines(), b.lines());
    }
}
