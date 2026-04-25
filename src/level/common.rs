use crate::assets::GameAssets;
use crate::engine::{
    fall_duration, soft_drop_duration, Board, Cell, LockDownMode, MoveDirection, Piece,
    PieceGenerator, PieceType, LOCK_DOWN_SECONDS, MIN_LEVEL,
};
use bevy::color::Alpha;
use bevy::math::{IVec2, Vec2, Vec3};
use bevy::prelude::{
    Color, Commands, Component, Deref, DerefMut, Entity, Event, Message, Res, Resource, Sprite,
    SubStates, Timer, TimerMode, Transform,
};
use bevy::sprite::Anchor;
use bevy::state::state::StateSet;
use std::time::Duration;

use crate::GameState;

#[derive(Message, Clone, Debug)]
pub enum ActionEvent {
    Rotation(PieceType, usize, u8), // piece type, occupied cells (only for T-Spin), SRS kick number
    Movement(MoveDirection),
    HardDrop(usize),
    Hold,
}

#[derive(Message, Clone)]
pub enum PlacingEvent {
    Locked(usize), // lines cleared
}

#[derive(Event, Clone, Debug)]
pub enum AudioCue {
    Rotation,
    HardDrop,
    Hold,
    Placed,
    Locked(usize),
}

#[derive(SubStates, PartialEq, Eq, Debug, Clone, Hash, Default)]
#[source(GameState = GameState::InGame)]
pub enum PlayingState {
    #[default]
    Falling,
    Locking,
}

#[derive(Resource, Debug)]
pub struct LevelConfig {
    pub(crate) block_size: f32,
    pub(crate) preview_scale: f32,
    pub(crate) board_width: usize,
    pub(crate) preview_count: usize,

    pub(crate) board_height: usize,
    pub(crate) movement_duration: Duration,
    pub(crate) movement_speedup: f64,
    pub(crate) soft_drop_duration: Duration,
    pub(crate) fall_duration: Duration,
    pub(crate) locking_duration: Duration,
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
            movement_duration: Duration::from_millis(200),
            soft_drop_duration: soft_drop_duration(MIN_LEVEL),
            movement_speedup: 1. / 1.0_f64.exp(),
            fall_duration: fall_duration(MIN_LEVEL),
            locking_duration: Duration::from_secs_f32(LOCK_DOWN_SECONDS),
            lock_down_mode: LockDownMode::default(),
        }
    }
}

impl LevelConfig {
    pub(crate) fn spawn_coords(&self, piece: &Piece) -> (isize, isize) {
        piece.spawn_coords(self.board_width, self.board_height)
    }
}

#[derive(Component, PartialEq, Eq, Debug, Clone, Hash)]
pub(crate) struct Coords {
    pub(crate) x: isize,
    pub(crate) y: isize,
}

impl From<(isize, isize)> for Coords {
    fn from((x, y): (isize, isize)) -> Self {
        Self { x, y }
    }
}

impl From<Coords> for (isize, isize) {
    fn from(coords: Coords) -> Self {
        (coords.x, coords.y)
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

#[derive(Component, Clone)]
#[require(Sprite, Transform, Anchor = Anchor::BOTTOM_LEFT)]
pub struct BackgroundBlock;

#[derive(Component, Clone)]
#[require(Sprite, Transform, Anchor = Anchor::BOTTOM_LEFT)]
pub struct FallingBlock;

#[derive(Component, Clone)]
#[require(Sprite, Transform, Anchor = Anchor::BOTTOM_LEFT)]
pub struct StaticBlock;

#[derive(Component, Clone)]
#[require(Sprite, Transform, Anchor = Anchor::BOTTOM_LEFT)]
pub struct GhostBlock;

#[derive(Component, Clone)]
#[require(Sprite, Transform, Anchor = Anchor::BOTTOM_LEFT)]
pub struct PreviewBlock;

#[derive(Component)]
pub struct GhostPiece;

#[derive(Component)]
pub struct PieceController {
    pub(crate) falling_timer: Timer,
    pub(crate) locking_timer: Timer,
    pub(crate) hard_dropped: bool,
    pub(crate) used_hold: bool,
    pub(crate) movement_timer: Timer,
    pub(crate) active_movement: Option<MoveDirection>,
    pub(crate) movement_hold_duration: Duration,
}

impl Default for PieceController {
    fn default() -> Self {
        let config = LevelConfig::default();
        Self::new(&config)
    }
}

impl PieceController {
    pub(crate) fn new(config: &LevelConfig) -> Self {
        Self {
            falling_timer: Timer::new(config.fall_duration, TimerMode::Repeating),
            locking_timer: Timer::new(config.locking_duration, TimerMode::Once),
            movement_timer: Timer::new(config.movement_duration, TimerMode::Repeating),
            hard_dropped: false,
            used_hold: false,
            active_movement: None,
            movement_hold_duration: Duration::ZERO,
        }
    }
}

pub(crate) trait MatchCoords {
    fn from_coords(coords: &Coords, config: &LevelConfig) -> Self;
    fn update_coords(&mut self, coords: &Coords, config: &LevelConfig);
}

pub fn to_translation(x: isize, y: isize, block_size: f32) -> Vec3 {
    IVec2::new(x as i32, y as i32).as_vec2().extend(0.0) * block_size
}

impl MatchCoords for Transform {
    fn from_coords(coords: &Coords, config: &LevelConfig) -> Self {
        Transform::from_translation(to_translation(coords.x, coords.y, config.block_size))
    }
    fn update_coords(&mut self, coords: &Coords, config: &LevelConfig) {
        self.translation = to_translation(coords.x, coords.y, config.block_size);
    }
}

#[derive(Component, Default)]
pub struct PieceHolder {
    pub(crate) piece: Option<Piece>,
}

#[derive(Component, Deref, DerefMut)]
pub struct BoardState(pub(crate) Board);

#[derive(Component, Clone, Debug, Deref, DerefMut)]
pub struct PieceState(pub(crate) Piece);

#[derive(Component, Deref, DerefMut)]
pub struct PieceGeneratorState(pub(crate) PieceGenerator);

// spawn a block that is yet a part of a piece at the given cell
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

    let coords = Coords::from((x, y));
    let mut transform = Transform::from_coords(&coords, config);

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
        .spawn((sprite, transform, Anchor::BOTTOM_LEFT, coords))
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

pub fn spawn_piece_blocks(
    commands: &mut Commands,
    config: &LevelConfig,
    texture_assets: &Res<GameAssets>,
    piece: &Piece,
    block_kind: BlockKind,
) -> Vec<Entity> {
    let board = piece.board();

    board
        .cells()
        .into_iter()
        .map(|cell| spawn_free_block(commands, config, texture_assets, cell, block_kind))
        .collect()
}

pub fn piece_color(piece_type: PieceType) -> Color {
    match piece_type {
        PieceType::I => Color::srgb_u8(100, 196, 235), // cyan
        PieceType::J => Color::srgb_u8(90, 99, 165),   // orange
        PieceType::L => Color::srgb_u8(224, 127, 58),  // blue
        PieceType::O => Color::srgb_u8(241, 212, 72),  // yellow
        PieceType::S => Color::srgb_u8(100, 180, 82),  // green
        PieceType::T => Color::srgb_u8(161, 83, 152),  // purple
        PieceType::Z => Color::srgb_u8(216, 57, 52),   // red
    }
}
