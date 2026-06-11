//! Engine ↔ Bevy bridge: the configuration seam and the input-edge latch.
//!
//! The session (`src/session/`) owns the engines and steps them; this module
//! owns what a step is built from. [`engine_config_for_game`] folds
//! `LevelConfig`/`GameSettings`/`Variant` into an [`EngineConfig`],
//! [`das_config_from_level`] does the same for the keyboard's DAS timings,
//! and [`PendingEdges`] latches just-pressed input for the fixed slices. The
//! renderer stays a one-way consumer of the engine: no render system mutates
//! simulation state.

use bevy::prelude::*;

use crate::engine::EngineConfig;
use crate::level::common::LevelConfig;
use crate::player::{DasConfig, RawKeyboardFrame};
use crate::settings::GameSettings;
use crate::variant::Variant;

/// Fixed simulation rate. The engine is stepped from Bevy's `FixedUpdate`
/// schedule, which runs as many fixed slices per render frame as the accumulated
/// virtual time allows, so gravity/lock-down advance deterministically
/// regardless of render frame rate. `SIM_HZ` seeds `Time::<Fixed>` in
/// `SessionPlugin::build` (Bevy's default is 64 Hz, which would not match);
/// `SIM_DT_SECONDS` mirrors the per-slice `dt` for the engine steps and tests.
pub const SIM_HZ: f32 = 60.0;
pub const SIM_DT_SECONDS: f32 = 1.0 / SIM_HZ;

/// Just-pressed input edges latched for the fixed sim slices that run this frame.
///
/// The render loop can run faster than [`SIM_HZ`] (e.g. 120fps vs 60Hz sim), so
/// some render frames accumulate less than one [`SIM_DT_SECONDS`] and run **zero**
/// engine steps. Bevy clears `just_pressed` in `PreUpdate` (before `FixedUpdate`
/// runs), so an edge (tap, hard drop, rotate, hold, pause) read directly inside a
/// fixed step would be lost on a frame that runs zero slices — and double-counted
/// on a frame that runs several. This was the cause of "I had to press
/// left/space several times". We latch edges here in `PreUpdate` (where
/// `just_pressed` is still valid) and drain them on the first fixed slice, then
/// [`reset`](Self::reset) so later slices in the same frame can't replay the
/// press. No press is dropped or duplicated.
#[derive(Resource, Default)]
pub struct PendingEdges {
    pub left: bool,
    pub right: bool,
    pub hard_drop: bool,
    pub rotate_cw: bool,
    pub rotate_ccw: bool,
    pub hold: bool,
    pub pause: bool,
}

impl PendingEdges {
    /// OR this frame's just-pressed edges into the latch.
    pub fn latch(&mut self, input: &RawKeyboardFrame) {
        self.left |= input.left_just_pressed;
        self.right |= input.right_just_pressed;
        self.hard_drop |= input.hard_drop_just_pressed;
        self.rotate_cw |= input.rotate_cw_just_pressed;
        self.rotate_ccw |= input.rotate_ccw_just_pressed;
        self.hold |= input.hold_just_pressed;
        self.pause |= input.pause_just_pressed;
    }

    /// Replace `input`'s edge flags with the latched edges, so that once they are
    /// cleared, extra slices in the same frame don't replay a press. A latched
    /// horizontal tap also forces the held flag true so the one-cell move still
    /// fires even if the key was physically released before this slice ran.
    pub fn drain_onto(&self, input: &mut RawKeyboardFrame) {
        input.left_just_pressed = self.left;
        input.right_just_pressed = self.right;
        input.hard_drop_just_pressed = self.hard_drop;
        input.rotate_cw_just_pressed = self.rotate_cw;
        input.rotate_ccw_just_pressed = self.rotate_ccw;
        input.hold_just_pressed = self.hold;
        input.pause_just_pressed = self.pause;
        if self.left {
            input.left_pressed = true;
        }
        if self.right {
            input.right_pressed = true;
        }
    }

    /// Clear all latched edges (after a slice consumes them).
    pub fn reset(&mut self) {
        *self = Self::default();
    }
}

/// Build an [`EngineConfig`] from the renderer's [`LevelConfig`].
///
/// The board is `board_width × board_height` visible with a hidden buffer above
/// (the renderer historically used a 20-row top margin to fake a 10×20 field).
/// DAS timings are intentionally NOT part of `EngineConfig`; DAS is player-side.
pub fn engine_config_from_level(config: &LevelConfig) -> EngineConfig {
    EngineConfig {
        board_width: config.board_width,
        visible_height: config.board_height,
        preview_count: config.preview_count,
        lock_down_mode: config.lock_down_mode,
        lock_down_seconds: config.locking_duration.as_secs_f32(),
        starting_level: crate::engine::MIN_LEVEL,
        goal_system: crate::engine::GoalSystem::Fixed,
        // Single-player: nothing feeds the garbage queue, so the cap is inert
        // until a versus mode arms it. The engine default is the standard 8.
        garbage_cap: EngineConfig::default().garbage_cap,
    }
}

/// Build the [`EngineConfig`] for a concrete game: start from the renderer's
/// [`LevelConfig`], overlay the player's [`GameSettings`] (preview/next count and
/// lock-down rule), then apply the [`Variant`]'s engine overrides (goal system).
///
/// This is the single seam where the shared contracts feed the engine, so the
/// previewer (which reads `LevelConfig.preview_count`), the engine queue, and the
/// variant goal system all stay consistent. Callers also mirror
/// `settings.next_count` into `LevelConfig.preview_count` before building UI.
pub fn engine_config_for_game(
    config: &LevelConfig,
    settings: &GameSettings,
    variant: Variant,
) -> EngineConfig {
    let mut engine_config = engine_config_from_level(config);
    engine_config.preview_count = settings.next_count;
    engine_config.lock_down_mode = settings.lock_down_mode;
    variant.def().apply_engine_overrides(&mut engine_config);
    engine_config
}

/// Build the player-side [`DasConfig`] from the renderer's [`LevelConfig`] DAS
/// durations (these stay on `LevelConfig`, consumed here, never by the engine).
pub fn das_config_from_level(config: &LevelConfig) -> DasConfig {
    DasConfig {
        delay_seconds: config.das_delay.as_secs_f32(),
        repeat_seconds: config.das_repeat_duration.as_secs_f32(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn left_tap() -> RawKeyboardFrame {
        RawKeyboardFrame {
            left_just_pressed: true,
            ..RawKeyboardFrame::default()
        }
    }

    #[test]
    fn latched_tap_survives_zero_step_frame_then_drains_exactly_once() {
        let mut pending = PendingEdges::default();
        // A frame with a left tap but no sim slice (render fps > SIM_HZ): latched.
        pending.latch(&left_tap());
        assert!(pending.left);

        // The next slice drains it: a left move fires, with the held flag forced so
        // the tap still moves even if the key was already released.
        let mut input = RawKeyboardFrame::default();
        pending.drain_onto(&mut input);
        assert!(input.left_just_pressed);
        assert!(input.left_pressed);
        pending.reset();

        // A second slice in the same frame must not replay the press.
        let mut second = RawKeyboardFrame::default();
        pending.drain_onto(&mut second);
        assert!(!second.left_just_pressed);
    }

    #[test]
    fn latch_stays_set_across_empty_frames_until_reset() {
        let mut pending = PendingEdges::default();
        pending.latch(&left_tap());
        pending.latch(&RawKeyboardFrame::default()); // empty frame, edge persists
        assert!(pending.left);
        pending.reset();
        assert!(!pending.left);
    }
}
