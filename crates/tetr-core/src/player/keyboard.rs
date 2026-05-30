//! Keyboard-driven [`PlayerController`] that owns player-side DAS.
//!
//! The Bevy driver feeds raw key state + frame `dt` in via
//! [`KeyboardController::set_input`] (built from `ButtonInput<KeyCode>` with
//! [`RawKeyboardFrame::from_keyboard`]), then calls [`PlayerController::poll`].
//! `poll` resolves the held horizontal direction, advances the DAS machine, and
//! emits an [`InputFrame`] whose `left`/`right` are per-frame one-cell pulses at
//! the DAS cadence. The other action flags are edge-triggered (just-pressed) so
//! the engine (which has no edge detection) sees one action per press.

use crate::engine::{EngineSnapshot, InputFrame, MoveDirection};
use crate::player::das::{DasConfig, DasState};
use crate::player::{resolve_horizontal, PlayerController};

/// Raw per-frame keyboard state, decoupled from Bevy so the controller can be
/// driven headlessly in tests. `pressed` = key currently down; `just_pressed` =
/// key transitioned to down this frame.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct RawKeyboardFrame {
    pub dt_seconds: f32,
    pub left_pressed: bool,
    pub right_pressed: bool,
    pub left_just_pressed: bool,
    pub right_just_pressed: bool,
    pub soft_drop: bool,
    pub hard_drop_just_pressed: bool,
    pub rotate_cw_just_pressed: bool,
    pub rotate_ccw_just_pressed: bool,
    pub hold_just_pressed: bool,
    pub pause_just_pressed: bool,
}

/// A [`PlayerController`] backed by the keyboard. Long-lived: it owns the DAS
/// charge so auto-repeat carries across piece boundaries (the engine never
/// resets it).
#[derive(Debug, Clone, Default)]
pub struct KeyboardController {
    config: DasConfig,
    das: DasState,
    input: RawKeyboardFrame,
}

impl KeyboardController {
    pub fn new(config: DasConfig) -> Self {
        Self {
            config,
            das: DasState::default(),
            input: RawKeyboardFrame::default(),
        }
    }

    /// Stage the raw keyboard state for the next [`poll`](PlayerController::poll).
    pub fn set_input(&mut self, input: RawKeyboardFrame) {
        self.input = input;
    }

    /// Build an [`InputFrame`] from a staged [`RawKeyboardFrame`] without mutating
    /// shared state — used internally by [`poll`] and directly by tests to drive
    /// the DAS machine deterministically.
    fn resolve_frame(&mut self, input: &RawKeyboardFrame) -> InputFrame {
        let (held_direction, just_pressed) = resolve_horizontal(
            input.left_pressed,
            input.right_pressed,
            input.left_just_pressed,
            input.right_just_pressed,
            self.das.active_direction(),
        );

        let pulse =
            self.das
                .next_pulse(held_direction, just_pressed, input.dt_seconds, &self.config);

        InputFrame {
            dt_seconds: input.dt_seconds,
            left: pulse == Some(MoveDirection::Left),
            right: pulse == Some(MoveDirection::Right),
            soft_drop: input.soft_drop,
            hard_drop: input.hard_drop_just_pressed,
            rotate_clockwise: input.rotate_cw_just_pressed,
            rotate_counterclockwise: input.rotate_ccw_just_pressed,
            hold: input.hold_just_pressed,
            pause: input.pause_just_pressed,
        }
    }
}

impl PlayerController for KeyboardController {
    fn poll(&mut self, _snapshot: &EngineSnapshot) -> InputFrame {
        let input = self.input;
        self.resolve_frame(&input)
    }
}

// The one Bevy touch-point in `tetr-core`: an adapter from Bevy's keyboard state to
// the engine-agnostic `RawKeyboardFrame`. Gated behind the `bevy` feature (off by
// default) so the core — and the embed wasm built from it — never pulls Bevy. The
// Bevy game enables the feature; the headless embed builds its own DOM-event adapter.
#[cfg(feature = "bevy")]
impl RawKeyboardFrame {
    /// Build raw input from Bevy's keyboard state for one frame.
    ///
    /// Bindings (per migration map): arrows for move/soft-drop, Space =
    /// hard drop, Up / X = rotate CW, Z = rotate CCW, LeftShift = hold,
    /// Escape = pause.
    pub fn from_keyboard(
        keyboard: &bevy::input::ButtonInput<bevy::input::keyboard::KeyCode>,
        dt_seconds: f32,
    ) -> Self {
        use bevy::input::keyboard::KeyCode;
        Self {
            dt_seconds,
            left_pressed: keyboard.pressed(KeyCode::ArrowLeft),
            right_pressed: keyboard.pressed(KeyCode::ArrowRight),
            left_just_pressed: keyboard.just_pressed(KeyCode::ArrowLeft),
            right_just_pressed: keyboard.just_pressed(KeyCode::ArrowRight),
            soft_drop: keyboard.pressed(KeyCode::ArrowDown),
            hard_drop_just_pressed: keyboard.just_pressed(KeyCode::Space),
            rotate_cw_just_pressed: keyboard.just_pressed(KeyCode::ArrowUp)
                || keyboard.just_pressed(KeyCode::KeyX),
            rotate_ccw_just_pressed: keyboard.just_pressed(KeyCode::KeyZ),
            hold_just_pressed: keyboard.just_pressed(KeyCode::ShiftLeft),
            pause_just_pressed: keyboard.just_pressed(KeyCode::Escape),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{Engine, EngineConfig};

    const CONFIG: DasConfig = DasConfig {
        delay_seconds: 0.3,
        repeat_seconds: 0.05,
    };

    /// A throwaway snapshot to satisfy `poll`'s signature; the keyboard
    /// controller ignores it.
    fn snapshot() -> EngineSnapshot {
        Engine::new(EngineConfig::default(), 0).snapshot()
    }

    fn hold_left(dt: f32) -> RawKeyboardFrame {
        RawKeyboardFrame {
            dt_seconds: dt,
            left_pressed: true,
            ..RawKeyboardFrame::default()
        }
    }

    fn tap_left() -> RawKeyboardFrame {
        RawKeyboardFrame {
            left_pressed: true,
            left_just_pressed: true,
            ..RawKeyboardFrame::default()
        }
    }

    fn poll(controller: &mut KeyboardController, input: RawKeyboardFrame) -> InputFrame {
        controller.set_input(input);
        controller.poll(&snapshot())
    }

    #[test]
    fn tap_emits_one_left_pulse() {
        let mut controller = KeyboardController::new(CONFIG);
        let frame = poll(&mut controller, tap_left());
        assert!(frame.left);
        assert!(!frame.right);
    }

    #[test]
    fn hold_past_delay_emits_initial_pulse_then_repeats() {
        let mut controller = KeyboardController::new(CONFIG);

        // Tap fires immediately.
        assert!(poll(&mut controller, tap_left()).left);
        // Holding short of the delay: no move.
        assert!(!poll(&mut controller, hold_left(0.29)).left);
        // Accumulating past the delay: first auto-shift.
        assert!(poll(&mut controller, hold_left(0.02)).left);
        // Short of the repeat interval: nothing.
        assert!(!poll(&mut controller, hold_left(0.04)).left);
        // Repeat interval reached: a repeat pulse.
        assert!(poll(&mut controller, hold_left(0.02)).left);
    }

    #[test]
    fn opposite_press_restarts_delay() {
        let mut controller = KeyboardController::new(CONFIG);
        // Charge Left fully.
        poll(&mut controller, tap_left());
        poll(&mut controller, hold_left(0.3));

        // Switch to Right: immediate tap, then a fresh delay (no carry-over).
        let switch = RawKeyboardFrame {
            right_pressed: true,
            right_just_pressed: true,
            ..RawKeyboardFrame::default()
        };
        assert!(poll(&mut controller, switch).right);

        let hold_right_short = RawKeyboardFrame {
            dt_seconds: 0.29,
            right_pressed: true,
            ..RawKeyboardFrame::default()
        };
        assert!(!poll(&mut controller, hold_right_short).right);
        let hold_right_cross = RawKeyboardFrame {
            dt_seconds: 0.02,
            right_pressed: true,
            ..RawKeyboardFrame::default()
        };
        assert!(poll(&mut controller, hold_right_cross).right);
    }

    #[test]
    fn charge_persists_across_piece_boundary() {
        // Simulate the driver never resetting the controller across a lock/spawn:
        // Left stays held, and the new piece keeps repeating on the 50ms cadence.
        let mut controller = KeyboardController::new(CONFIG);
        poll(&mut controller, tap_left());
        poll(&mut controller, hold_left(0.3));
        assert!(poll(&mut controller, hold_left(0.05)).left);

        // "Piece boundary" — no special handling, key still held.
        assert!(
            poll(&mut controller, hold_left(0.05)).left,
            "carried-over DAS keeps repeating on the new piece"
        );
        assert!(
            poll(&mut controller, hold_left(0.05)).left,
            "carry-over must not re-arm the initial delay"
        );
    }

    #[test]
    fn action_flags_pass_through_as_edge_triggers() {
        let mut controller = KeyboardController::new(CONFIG);
        let input = RawKeyboardFrame {
            dt_seconds: 0.016,
            soft_drop: true,
            hard_drop_just_pressed: true,
            rotate_cw_just_pressed: true,
            rotate_ccw_just_pressed: false,
            hold_just_pressed: true,
            pause_just_pressed: true,
            ..RawKeyboardFrame::default()
        };
        let frame = poll(&mut controller, input);

        assert!(frame.soft_drop);
        assert!(frame.hard_drop);
        assert!(frame.rotate_clockwise);
        assert!(!frame.rotate_counterclockwise);
        assert!(frame.hold);
        assert!(frame.pause);
        assert_eq!(frame.dt_seconds, 0.016);
    }
}
