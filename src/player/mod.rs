//! Player abstraction (roadmap ADR-4).
//!
//! A [`PlayerController`] turns the latest [`EngineSnapshot`] into the next
//! [`InputFrame`] that an `Engine` should be stepped with. Multiplayer is "N
//! engines + N controllers stepped in lockstep"; replays and AI are other
//! controllers behind the same trait.
//!
//! DAS (Delayed Auto Shift) is **player-side** state, not engine state (ADR-4 /
//! roadmap E0.13). The engine treats `InputFrame.left` / `.right` as a per-frame
//! one-cell pulse and never owns a charge timer. [`KeyboardController`] is the
//! reference owner of that DAS state machine: it converts a held arrow key into
//! the correct cadence of pulses (one tap, then an initial-delay wait, then
//! auto-repeat), restarts the delay on opposite-direction presses, and carries
//! the charge across piece boundaries (the controller is long-lived and the
//! engine never resets it).

use crate::engine::{EngineSnapshot, InputFrame, MoveDirection};

mod das;
mod keyboard;

pub use das::{DasConfig, DasState};
pub use keyboard::{KeyboardController, KeyboardInput};

/// Produces the next [`InputFrame`] for an `Engine`, given the latest snapshot.
///
/// Per ADR-4 the engine is stepped as `engine.step(controller.poll(&snapshot))`.
/// The snapshot lets stateful controllers (AI, replay) react to what the engine
/// did last frame; simple controllers may ignore it.
pub trait PlayerController {
    fn poll(&mut self, snapshot: &EngineSnapshot) -> InputFrame;
}

/// Resolve a `(left_held, right_held)` pair into the single horizontal direction
/// DAS should act on. When both are held, the most recently *just-pressed*
/// direction wins; if neither was just pressed this frame, `prev_active` (the
/// direction DAS is already charging) is kept so a simultaneous hold does not
/// stutter. Shared by the keyboard controller and reused by tests.
pub(crate) fn resolve_horizontal(
    left_held: bool,
    right_held: bool,
    left_just_pressed: bool,
    right_just_pressed: bool,
    prev_active: Option<MoveDirection>,
) -> (Option<MoveDirection>, bool) {
    match (left_held, right_held) {
        (true, false) => (Some(MoveDirection::Left), left_just_pressed),
        (false, true) => (Some(MoveDirection::Right), right_just_pressed),
        (true, true) if left_just_pressed => (Some(MoveDirection::Left), true),
        (true, true) if right_just_pressed => (Some(MoveDirection::Right), true),
        (true, true) => (prev_active, false),
        (false, false) => (None, false),
    }
}
