use crate::assets::GameAssets;
use crate::engine::{Cell, LockDownMode, PieceType};
use bevy::color::Alpha;
use bevy::math::{IVec2, Vec2, Vec3};
use bevy::prelude::{
    Color, Commands, Component, Entity, Reflect, ReflectComponent, ReflectResource, Res, Resource,
    Sprite, SubStates, SystemSet, Transform,
};
use bevy::sprite::Anchor;
use bevy::state::state::StateSet;
use std::time::Duration;

use crate::{DespawnOnExit, GameState, InGameplay};

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

/// Falling vs. Locking, derived each frame from `snapshot.active.landed`. Drives
/// the lock-down timer bar's visibility (it only shows while Locking).
#[derive(SubStates, PartialEq, Eq, Debug, Clone, Hash, Default)]
#[source(GameState = GameState::Playing)]
pub enum PlayingState {
    #[default]
    Falling,
    Locking,
}

/// System ordering label: the engine driver runs before everything that reads
/// the snapshot/events it produces.
#[derive(SystemSet, Debug, Hash, PartialEq, Eq, Clone, Copy)]
pub(crate) enum LevelSystems {
    /// Collects input, steps the engine, publishes snapshot + events.
    EngineDriver,
    /// Reconciles render entities / UI / audio from the snapshot + events.
    Reconcile,
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

#[derive(Copy, Clone, PartialEq, Eq)]
pub enum BlockKind {
    Background,
    Falling,
    Static,
    Ghost,
    Preview,
}

/// Anchor entity at the board origin. The lock-down timer bar parents to it so
/// the bar's local transform maps cleanly into board space.
#[derive(Component, Reflect)]
#[reflect(Component)]
pub struct GameField;

#[derive(Component, Clone, Reflect)]
#[reflect(Component)]
pub struct BackgroundBlock;

/// A cell of the active (falling) piece. Reconciled from `snapshot.active`.
#[derive(Component, Clone, Reflect)]
#[reflect(Component)]
pub struct FallingBlock;

/// A locked board mino. Reconciled from `snapshot.board_cells`.
#[derive(Component, Clone, Reflect)]
#[reflect(Component)]
pub struct StaticBlock;

/// A ghost-piece cell. Reconciled from `snapshot.ghost_cells`.
#[derive(Component, Clone, Reflect)]
#[reflect(Component)]
pub struct GhostBlock;

#[derive(Component, Clone, Reflect)]
#[reflect(Component)]
pub struct PreviewBlock;

pub fn to_translation(x: isize, y: isize, block_size: f32) -> Vec3 {
    IVec2::new(x as i32, y as i32).as_vec2().extend(0.0) * block_size
}

/// Build a render block at a single board/ghost/preview cell. Reused verbatim
/// from the pre-migration renderer; the only difference is callers now feed it
/// cells derived from `SnapshotCell`s instead of from a parallel `Board`.
pub fn spawn_free_block(
    commands: &mut Commands,
    config: &LevelConfig,
    texture_assets: &Res<GameAssets>,
    cell: &Cell,
    block_kind: BlockKind,
) -> Entity {
    let (x, y) = cell.coords();

    let color = match block_kind {
        BlockKind::Falling | BlockKind::Preview | BlockKind::Static => {
            piece_color(cell.cell_kind().unwrap())
        }
        BlockKind::Ghost => Color::srgb(0.5, 0.5, 0.5).with_alpha(0.5),
        BlockKind::Background => Color::srgb(0.1, 0.1, 0.1),
    };

    let mut transform = Transform::from_translation(to_translation(x, y, config.block_size));

    let sprite = match block_kind {
        BlockKind::Background => {
            Sprite::from_color(color, Vec2::new(config.block_size, config.block_size))
        }
        _ => {
            let mut sprite = Sprite::from_image(texture_assets.block_texture.clone());
            sprite.custom_size = Some(Vec2::new(config.block_size, config.block_size));
            sprite.color = color;
            sprite
        }
    };

    transform.translation.z = match block_kind {
        BlockKind::Ghost => -0.1,     // ghost is behind falling block
        BlockKind::Background => -1., // background is behind everything
        _ => 0.,
    };

    let entity = commands
        .spawn((sprite, transform, Anchor::BOTTOM_LEFT))
        .id();

    match block_kind {
        BlockKind::Background => {
            commands.entity(entity).insert(BackgroundBlock);
        }
        BlockKind::Falling => {
            commands.entity(entity).insert(FallingBlock);
        }
        BlockKind::Static => {
            commands.entity(entity).insert(StaticBlock);
        }
        BlockKind::Ghost => {
            commands.entity(entity).insert(GhostBlock);
        }
        BlockKind::Preview => {
            commands.entity(entity).insert(PreviewBlock);
        }
    }

    entity
}

/// Spawn a render block for a snapshot mino at absolute board coords with a
/// despawn-on-exit guard. Used by the per-frame reconcilers.
pub fn spawn_snapshot_block(
    commands: &mut Commands,
    config: &LevelConfig,
    texture_assets: &Res<GameAssets>,
    x: isize,
    y: isize,
    piece_type: PieceType,
    block_kind: BlockKind,
) -> Entity {
    let cell = Cell::new(x, y, crate::engine::CellKind::Some(piece_type));
    let entity = spawn_free_block(commands, config, texture_assets, &cell, block_kind);
    commands.entity(entity).insert(DespawnOnExit(InGameplay));
    entity
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
