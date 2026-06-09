//! Player-side DAS (Delayed Auto Shift) state machine.
//!
//! This is the seconds-based port of the renderer's `DasState`. It is deliberately
//! Bevy-free and engine-free so it can
//! be driven directly in headless unit tests: feed it a held direction plus the
//! frame `dt` and it returns whether a one-cell move pulse fires this frame.
//!
//! Cadence (reference §6.2 / §25.3):
//!   * A tap (just-pressed) fires one pulse immediately.
//!   * Holding waits `delay_seconds` (initial DAS delay) before the first
//!     auto-shift, then repeats every `repeat_seconds`.
//!   * Pressing the opposite direction restarts the full initial delay.
//!   * Charge persists as long as the direction stays held; nothing external
//!     resets it across piece locks/spawns, so auto-repeat carries over.

use crate::engine::MoveDirection;

/// Player-side DAS timings. These live with the player, not the engine — the
/// engine applies one cell per `left`/`right` pulse and never consults them
/// (roadmap ADR-4 / E0.13).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DasConfig {
    /// Initial delay before auto-shift begins, in seconds (~0.3s per spec).
    pub delay_seconds: f32,
    /// Auto-shift repeat interval after the initial delay, in seconds (~0.05s).
    pub repeat_seconds: f32,
}

impl Default for DasConfig {
    fn default() -> Self {
        Self {
            delay_seconds: 0.3,
            repeat_seconds: 0.05,
        }
    }
}

/// The DAS charge state machine. `None` `active_direction` means no horizontal
/// key is currently charging.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct DasState {
    active_direction: Option<MoveDirection>,
    held_seconds: f32,
    repeat_elapsed_seconds: f32,
}

impl DasState {
    /// The direction DAS is currently charging, if any. Used to disambiguate a
    /// simultaneous left+right hold.
    pub fn active_direction(&self) -> Option<MoveDirection> {
        self.active_direction
    }

    /// Advance the DAS machine one frame and return the move pulse (if any).
    ///
    /// * `held_direction`: the resolved horizontal direction held this frame, or
    ///   `None` if no horizontal key is held.
    /// * `just_pressed`: whether that direction was newly pressed this frame.
    /// * `dt_seconds`: elapsed wall-clock time for this frame.
    /// * `config`: the player-side DAS timings.
    ///
    /// Returns `Some(direction)` on the frames a one-cell move should occur.
    pub fn next_pulse(
        &mut self,
        held_direction: Option<MoveDirection>,
        just_pressed: bool,
        dt_seconds: f32,
        config: &DasConfig,
    ) -> Option<MoveDirection> {
        let dt_seconds = dt_seconds.max(0.0);

        let Some(direction) = held_direction else {
            self.reset();
            return None;
        };

        // A new direction (including switching sides): immediate tap, charge resets.
        if self.active_direction != Some(direction) {
            self.active_direction = Some(direction);
            self.held_seconds = 0.0;
            self.repeat_elapsed_seconds = 0.0;
            return just_pressed.then_some(direction);
        }

        // Re-press of the same direction: fire immediately, restart repeat phase.
        if just_pressed {
            self.repeat_elapsed_seconds = 0.0;
            return Some(direction);
        }

        let was_waiting_for_delay = self.held_seconds < config.delay_seconds;
        self.held_seconds += dt_seconds;

        if was_waiting_for_delay {
            if self.held_seconds >= config.delay_seconds {
                // Crossed the initial delay this frame: first auto-shift.
                self.repeat_elapsed_seconds = 0.0;
                return Some(direction);
            }
            return None;
        }

        self.repeat_elapsed_seconds += dt_seconds;
        if self.repeat_elapsed_seconds >= config.repeat_seconds {
            self.repeat_elapsed_seconds -= config.repeat_seconds;
            Some(direction)
        } else {
            None
        }
    }

    fn reset(&mut self) {
        self.active_direction = None;
        self.held_seconds = 0.0;
        self.repeat_elapsed_seconds = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const CONFIG: DasConfig = DasConfig {
        delay_seconds: 0.3,
        repeat_seconds: 0.05,
    };

    #[test]
    fn tap_fires_one_pulse_immediately() {
        let mut das = DasState::default();
        assert_eq!(
            das.next_pulse(Some(MoveDirection::Left), true, 0.0, &CONFIG),
            Some(MoveDirection::Left)
        );
    }

    #[test]
    fn hold_waits_for_initial_delay_then_repeats() {
        // Use a realistic ~16ms frame; thresholds are crossed by accumulation, not
        // by landing exactly on the boundary (f32 sums don't hit it exactly).
        const DT: f32 = 0.016;
        let mut das = DasState::default();
        // Tap consumes the initial press.
        das.next_pulse(Some(MoveDirection::Left), true, 0.0, &CONFIG);

        // Frames short of the 0.3s delay must not auto-shift. 0.3 / 0.016 ≈ 18.75,
        // so 18 frames (= 0.288s) is still before the delay.
        for _ in 0..18 {
            assert_eq!(
                das.next_pulse(Some(MoveDirection::Left), false, DT, &CONFIG),
                None,
                "no auto-shift before the initial delay elapses"
            );
        }
        // The next frame crosses 0.3s: first auto-shift.
        assert_eq!(
            das.next_pulse(Some(MoveDirection::Left), false, DT, &CONFIG),
            Some(MoveDirection::Left),
            "first auto-shift fires once the initial delay is crossed"
        );
        // Now in the repeat phase (repeat interval 0.05s). A 0.04s frame is under
        // one interval -> no pulse...
        assert_eq!(
            das.next_pulse(Some(MoveDirection::Left), false, 0.04, &CONFIG),
            None
        );
        // ...and accumulating past 0.05s yields one repeat pulse.
        assert_eq!(
            das.next_pulse(Some(MoveDirection::Left), false, 0.02, &CONFIG),
            Some(MoveDirection::Left)
        );
    }

    #[test]
    fn opposite_direction_press_restarts_delay() {
        let mut das = DasState::default();
        // Fully charge Left through the initial delay (0.4 > 0.3).
        das.next_pulse(Some(MoveDirection::Left), true, 0.0, &CONFIG);
        das.next_pulse(Some(MoveDirection::Left), false, 0.4, &CONFIG);

        // Opposite press: immediate tap, charge resets to the new direction.
        assert_eq!(
            das.next_pulse(Some(MoveDirection::Right), true, 0.0, &CONFIG),
            Some(MoveDirection::Right)
        );
        // The Left charge must NOT carry into Right: a frame just shy of the
        // restarted delay produces no auto-shift.
        assert_eq!(
            das.next_pulse(Some(MoveDirection::Right), false, 0.29, &CONFIG),
            None
        );
        // Accumulating past the delay yields the first Right auto-shift.
        assert_eq!(
            das.next_pulse(Some(MoveDirection::Right), false, 0.02, &CONFIG),
            Some(MoveDirection::Right)
        );
    }

    #[test]
    fn releasing_direction_clears_charge() {
        let mut das = DasState::default();
        das.next_pulse(Some(MoveDirection::Left), true, 0.0, &CONFIG);
        das.next_pulse(Some(MoveDirection::Left), false, 0.3, &CONFIG);
        assert_eq!(das.active_direction(), Some(MoveDirection::Left));

        assert_eq!(das.next_pulse(None, false, 0.05, &CONFIG), None);
        assert_eq!(das.active_direction(), None);
    }

    #[test]
    fn charge_persists_across_a_simulated_piece_boundary() {
        // The controller is long-lived and nothing resets DAS across a lock/spawn.
        // While Left stays held through a piece lock + the next spawn, the new
        // piece must keep auto-shifting on the repeat cadence, NOT re-arm the
        // initial delay.
        let mut das = DasState::default();
        das.next_pulse(Some(MoveDirection::Left), true, 0.0, &CONFIG);
        das.next_pulse(Some(MoveDirection::Left), false, 0.3, &CONFIG);
        assert_eq!(
            das.next_pulse(Some(MoveDirection::Left), false, 0.05, &CONFIG),
            Some(MoveDirection::Left),
        );

        // --- Piece A locks; Piece B spawns. The driver does NOT touch DasState,
        // and Left is still held (no release, no fresh press). ---
        assert_eq!(
            das.next_pulse(Some(MoveDirection::Left), false, 0.05, &CONFIG),
            Some(MoveDirection::Left),
            "carried-over DAS must keep repeating on the new piece",
        );
        // ...and it must not behave like a fresh hold: a sub-delay frame still
        // fires on the repeat cadence rather than waiting out a new initial delay.
        assert_eq!(
            das.next_pulse(Some(MoveDirection::Left), false, 0.05, &CONFIG),
            Some(MoveDirection::Left),
            "carry-over must not re-arm the initial delay",
        );
    }
}
