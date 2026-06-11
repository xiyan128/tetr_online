//! Shared session vocabulary: the renderer-side [`LevelConfig`], the
//! engine-decoupled [`AudioCue`], the gameplay-camera marker, and the
//! colour/translation helpers that turn engine coordinates into world-space
//! sprites. Schedule sets and block markers are the session render's own
//! (`src/session/render.rs`); nothing here touches a schedule.

use crate::engine::{LockDownMode, PieceType};
use bevy::math::{IVec2, Vec3};
use bevy::prelude::{Color, Component, Reflect, ReflectComponent, ReflectResource, Resource};
use std::time::Duration;

/// Audio cue, decoupled from the engine. The engine-bridge maps [`EngineEvent`]s
/// (Rotated/HardDropped/Held/Locked) onto these so the existing
/// `SoundEffectsPlugin` observer keeps working unchanged.
///
/// [`EngineEvent`]: crate::engine::EngineEvent
#[derive(bevy::prelude::Event, Clone, Debug)]
pub enum AudioCue {
    Rotation,
    HardDrop,
    Hold,
    Placed,
    Locked(usize),
}

#[derive(Resource, Debug, Reflect)]
#[reflect(Resource)]
pub struct LevelConfig {
    pub(crate) block_size: f32,
    pub(crate) preview_scale: f32,
    pub(crate) board_width: usize,
    pub(crate) preview_count: usize,

    pub(crate) board_height: usize,
    // DAS timings stay player-side (consumed by KeyboardController via the
    // engine-bridge's das_config_from_level; never read by the engine).
    pub(crate) das_delay: Duration,
    pub(crate) das_repeat_duration: Duration,
    pub(crate) locking_duration: Duration,
    // `LockDownMode` lives in the engine-agnostic `engine/` crate, which must not
    // depend on Bevy (no `Reflect`). Skip it for reflection rather than couple the
    // engine to Bevy; the inspector simply won't surface this one field.
    #[reflect(ignore)]
    pub(crate) lock_down_mode: LockDownMode,
}

impl Default for LevelConfig {
    fn default() -> Self {
        Self {
            block_size: 32.0,
            preview_scale: 0.8,
            board_width: 10,
            board_height: 20,
            preview_count: 6,
            das_delay: Duration::from_millis(300),
            das_repeat_duration: Duration::from_millis(50),
            locking_duration: Duration::from_secs_f32(crate::engine::LOCK_DOWN_SECONDS),
            lock_down_mode: LockDownMode::default(),
        }
    }
}

/// Marker for the in-game camera (the session spawns it). Visual-FX systems that
/// target *gameplay* specifically — screen shake, the optional bloom skin — query
/// this so they never disturb the separate menu cameras.
#[derive(Component, Reflect)]
#[reflect(Component)]
pub struct GameplayCamera;

pub fn to_translation(x: isize, y: isize, block_size: f32) -> Vec3 {
    IVec2::new(x as i32, y as i32).as_vec2().extend(0.0) * block_size
}

/// The Kissaten piece palette: standard guideline hues (trained recognition
/// transfers intact) with saturation compressed ~30% from stock. Mute, never
/// merge — the seven hues stay separable from each other and from garbage at
/// spectator scale.
pub fn piece_color(piece_type: PieceType) -> Color {
    match piece_type {
        PieceType::I => Color::srgb_u8(114, 181, 196), // cyan   #72B5C4
        PieceType::J => Color::srgb_u8(107, 118, 173), // blue   #6B76AD
        PieceType::L => Color::srgb_u8(201, 128, 63),  // orange #C9803F
        PieceType::O => Color::srgb_u8(217, 190, 86),  // yellow #D9BE56
        PieceType::S => Color::srgb_u8(123, 164, 94),  // green  #7BA45E
        PieceType::T => Color::srgb_u8(156, 110, 150), // purple #9C6E96
        PieceType::Z => Color::srgb_u8(194, 85, 76),   // red    #C2554C
    }
}
