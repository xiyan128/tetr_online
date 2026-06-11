//! Shared session vocabulary: the renderer-side [`LevelConfig`], the
//! engine-decoupled [`AudioCue`], the gameplay-camera marker, and the
//! colour/translation helpers that turn engine coordinates into world-space
//! sprites. (The per-frame schedule sets and block markers that lived here
//! died with the single-player pipeline — the session render owns its own.)

use crate::engine::{LockDownMode, PieceType};
#[cfg(feature = "bloom")]
use bevy::color::LinearRgba;
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
/// target *gameplay* specifically — screen shake, neon bloom, the CRT pass — query
/// this so they never disturb the separate menu cameras.
#[derive(Component, Reflect)]
#[reflect(Component)]
pub struct GameplayCamera;

pub fn to_translation(x: isize, y: isize, block_size: f32) -> Vec3 {
    IVec2::new(x as i32, y as i32).as_vec2().extend(0.0) * block_size
}

pub fn piece_color(piece_type: PieceType) -> Color {
    match piece_type {
        PieceType::I => Color::srgb_u8(100, 196, 235), // cyan
        PieceType::J => Color::srgb_u8(90, 99, 165),   // blue
        PieceType::L => Color::srgb_u8(224, 127, 58),  // orange
        PieceType::O => Color::srgb_u8(241, 212, 72),  // yellow
        PieceType::S => Color::srgb_u8(100, 180, 82),  // green
        PieceType::T => Color::srgb_u8(161, 83, 152),  // purple
        PieceType::Z => Color::srgb_u8(216, 57, 52),   // red
    }
}

/// Multiplier that lifts mino colors past the bloom threshold so they glow under
/// the neon pass. Only compiled with the `bloom` feature — on the WebGL2 bundle an
/// over-bright color would merely clamp to a washed-out white, so the plain palette
/// is used instead.
#[cfg(feature = "bloom")]
const MINO_GLOW: f32 = 1.6;

/// On-screen color for a piece's minos: [`piece_color`] lifted into HDR for the
/// bloom glow on capable builds, or the plain palette color otherwise. The hue is
/// preserved (all channels scale together); the brightest channels clip to a
/// neon-white core while bloom carries the color out into the halo.
pub fn mino_render_color(piece_type: PieceType) -> Color {
    let base = piece_color(piece_type);
    #[cfg(feature = "bloom")]
    {
        let c = base.to_linear();
        Color::LinearRgba(LinearRgba::rgb(
            c.red * MINO_GLOW,
            c.green * MINO_GLOW,
            c.blue * MINO_GLOW,
        ))
    }
    #[cfg(not(feature = "bloom"))]
    base
}
