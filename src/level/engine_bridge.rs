//! Engine ↔ Bevy bridge (P2.2).
//!
//! Makes the renderer a one-way consumer of the engine: `Engine → Snapshot →
//! Renderer`. The renderer holds a single authoritative [`Engine`] in
//! [`EngineState`], drives it at a fixed sim rate from real frame time, and
//! publishes the per-frame [`EngineSnapshot`] and [`EngineEvent`] list into
//! resources that every render / audio / score / UI system reads. No renderer
//! system mutates simulation state; all state changes flow through
//! `engine.step(controller.poll(&snapshot))`.

use bevy::prelude::*;

use crate::engine::{Engine, EngineConfig, EngineEvent, EngineSnapshot};
use crate::level::common::LevelConfig;
use crate::player::{DasConfig, KeyboardController};

/// Fixed simulation rate. Real frame `dt` is accumulated and the engine is
/// stepped in fixed slices so gravity/lock-down advance deterministically
/// regardless of render frame rate.
pub const SIM_HZ: f32 = 60.0;
pub const SIM_DT_SECONDS: f32 = 1.0 / SIM_HZ;

/// Seed for the engine's piece generator. Fixed for now (matches the renderer's
/// previous `PieceGenerator::with_seed(0)`); a replay/match layer may supply it
/// later.
pub const DEFAULT_SEED: u64 = 0;

/// The single authoritative simulation.
#[derive(Resource)]
pub struct EngineState(pub Engine);

/// Latest snapshot produced by the driver this frame. Every render/UI system
/// reads from here instead of from parallel Bevy state.
#[derive(Resource)]
pub struct LatestSnapshot(pub EngineSnapshot);

/// Events emitted by the engine during the steps that ran this frame. Replaced
/// wholesale each frame by the driver. Read by SFX / score / game-over systems.
/// Stored as a plain resource (not a Bevy message stream) so the canonical
/// consumers can each iterate the same list without double-buffering games.
#[derive(Resource, Default)]
pub struct FrameEvents(pub Vec<EngineEvent>);

/// The player's keyboard controller (owns player-side DAS; P2.1).
#[derive(Resource)]
pub struct PlayerInput(pub KeyboardController);

/// Real-time accumulator for the fixed-timestep driver.
#[derive(Resource, Default)]
pub struct SimClock {
    pub accumulator_seconds: f32,
}

/// Build an [`EngineConfig`] from the renderer's [`LevelConfig`].
///
/// The board is `board_width × board_height` visible with a hidden buffer above
/// (the renderer historically used a 20-row top margin to fake a 10×20 field).
/// DAS timings are intentionally NOT part of `EngineConfig` (player-side, ADR-4).
pub fn engine_config_from_level(config: &LevelConfig) -> EngineConfig {
    EngineConfig {
        board_width: config.board_width,
        visible_height: config.board_height,
        buffer_height: 20,
        preview_count: config.preview_count,
        lock_down_mode: config.lock_down_mode,
        lock_down_seconds: config.locking_duration.as_secs_f32(),
        starting_level: crate::engine::MIN_LEVEL,
        goal_system: crate::engine::GoalSystem::Fixed,
    }
}

/// Build the player-side [`DasConfig`] from the renderer's [`LevelConfig`] DAS
/// durations (these stay on `LevelConfig`, consumed here, never by the engine).
pub fn das_config_from_level(config: &LevelConfig) -> DasConfig {
    DasConfig {
        delay_seconds: config.das_delay.as_secs_f32(),
        repeat_seconds: config.das_repeat_duration.as_secs_f32(),
    }
}
