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
pub use keyboard::{KeyboardController, RawKeyboardFrame};

/// Produces the next [`InputFrame`] for an `Engine`, given the latest snapshot.
///
/// Per ADR-4 the engine is stepped as `engine.step(controller.poll(&snapshot))`.
/// The snapshot lets stateful controllers (AI, replay) react to what the engine
/// did last frame; simple controllers may ignore it.
pub trait PlayerController {
    fn poll(&mut self, snapshot: &EngineSnapshot) -> InputFrame;
}

/// Step `engine` once for `controller`: poll the controller against the engine's
/// current snapshot and feed the resulting [`InputFrame`] straight to
/// [`Engine::step`](crate::engine::Engine::step). Returns the events that step
/// produced.
///
/// This is the engine-agnostic driving seam. A controller that is **not** the
/// keyboard (the AI, a replay, a remote peer) needs no `PreUpdate` raw-input
/// latching, so it can drive an engine through nothing but `poll → step`. The
/// AI sandbox (AI3.6) and a future N-player local game (M3) own their own
/// `(Engine, Box<dyn PlayerController>)` pair(s) and tick them with this helper;
/// the single-player **keyboard** path keeps its dedicated latch-in-`PreUpdate`
/// pipeline (`src/level`) unchanged, because its controller needs the raw frame
/// staged before `poll`.
///
/// # The controller owns `dt`
///
/// Unlike the keyboard path (where the level driver stamps
/// `input.dt_seconds = time.delta_secs()` per slice), this helper steps the engine
/// with **exactly the frame the controller emitted**, `dt` included. That is
/// load-bearing for the AI: it positions a piece with a burst of `dt == 0`
/// maneuver frames (so gravity does not advance and the piece lands where the
/// planner intended — see [`placement_to_inputs`](crate::ai::placement_to_inputs))
/// and advances real time only on its neutral "thinking"/idle frames. Overwriting
/// `dt` here would apply gravity mid-maneuver and desync the placement.
///
/// Pure (no Bevy).
pub fn drive_engine(
    engine: &mut crate::engine::Engine,
    controller: &mut dyn PlayerController,
) -> Vec<crate::engine::EngineEvent> {
    let snapshot = engine.snapshot();
    let frame = controller.poll(&snapshot);
    engine.step(frame)
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::{AiController, Handicap, DEFAULT_AI_SEED};
    use crate::engine::{Engine, EngineConfig};

    /// The integration seam: an engine can be driven entirely through a
    /// `Box<dyn PlayerController>` (here the AI) with nothing but
    /// [`drive_engine`] — no keyboard, no `PreUpdate` latching. This is the path
    /// the AI sandbox (AI3.6) and future N-player local games (M3) use.
    #[test]
    fn drive_engine_runs_a_boxed_controller_to_place_pieces() {
        let mut controller: Box<dyn PlayerController> =
            Box::new(AiController::new(Handicap::perfect(), DEFAULT_AI_SEED));
        let mut engine = Engine::new(EngineConfig::default(), 7);

        let mut placed = false;
        for _ in 0..1_000 {
            if engine.snapshot().game_over.is_some() {
                break;
            }
            // `&mut *controller` reborrows the box as `&mut dyn PlayerController`.
            drive_engine(&mut engine, &mut *controller);
            if !engine.snapshot().board_cells.is_empty() {
                placed = true;
                break;
            }
        }
        assert!(
            placed,
            "driving a boxed AiController via drive_engine should lock a piece"
        );
    }

    /// `drive_engine` must honour the controller's own `dt`: an AI maneuver frame
    /// (`dt == 0`) is stepped without advancing gravity, so the helper never
    /// overrides it (unlike the keyboard level driver, which stamps the slice dt).
    #[test]
    fn drive_engine_preserves_the_controllers_frame_dt() {
        /// A stub controller that always emits a zero-dt left pulse, and records
        /// nothing — we only assert the engine saw a zero-dt step (no gravity).
        struct ZeroDtLeft;
        impl PlayerController for ZeroDtLeft {
            fn poll(&mut self, _snapshot: &EngineSnapshot) -> InputFrame {
                InputFrame {
                    dt_seconds: 0.0,
                    left: true,
                    ..InputFrame::default()
                }
            }
        }

        let mut engine = Engine::new(EngineConfig::default(), 1);
        engine.step(InputFrame::default()); // spawn a piece
        let before = engine.snapshot().active.unwrap().origin;

        let mut controller = ZeroDtLeft;
        drive_engine(&mut engine, &mut controller);

        let after = engine.snapshot().active.unwrap().origin;
        // The piece shifted one cell left (the pulse applied) but did NOT fall,
        // because the zero-dt frame the controller chose was preserved.
        assert_eq!(after.0, before.0 - 1, "left pulse applied");
        assert_eq!(after.1, before.1, "no gravity on a zero-dt frame");
    }
}
