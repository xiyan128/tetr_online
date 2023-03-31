use crate::assets::GameAssets;
use crate::core::{Board, Cell, Piece, PieceType};
use bevy::asset::Handle;
use bevy::math::{IVec2, Vec2, Vec3};
use bevy::prelude::{
    default, Bundle, Color, Commands, Component, Entity, Image, Res, Resource, SpatialBundle,
    Sprite, SpriteBundle, States, Timer, TimerMode, Transform,
};
use bevy::render::texture::DEFAULT_IMAGE_HANDLE;
use bevy::sprite::Anchor;
use std::time::Duration;

#[derive(States, PartialEq, Eq, Debug, Clone, Hash, Default)]
pub enum LevelState {
    #[default]
    Idle,
    Setup,
    Playing,
    GameOver,
}

#[derive(States, PartialEq, Eq, Debug, Clone, Hash, Default)]
pub enum PlayingState {
    #[default]
    Falling,
    Placing,
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
    pub(crate) placing_duration: Duration,
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
            soft_drop_duration: Duration::from_millis(50),
            movement_speedup: 1. / 1.0_f64.exp(),
            fall_duration: Duration::from_millis(500),
            placing_duration: Duration::from_millis(500),
        }
    }
}

impl LevelConfig {
    pub(crate) fn spawn_coords(&self, piece: &Piece) -> (isize, isize) {
        let (offset_x, offset_y) = piece.board_size();
        (
            self.board_width as isize / 2 - offset_x as isize / 2,
            self.board_height as isize,
        )
    }
}

#[derive(Component)]
pub struct LevelCleanup;

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

impl Into<(isize, isize)> for Coords {
    fn into(self) -> (isize, isize) {
        (self.x, self.y)
    }
}

#[derive(PartialEq)]
pub enum BlockComponentKind {
    Background,
    Falling,
    Static,
    Ghost,
    Preview,
}

pub trait BlockComponent: Component {
    fn kind(&self) -> BlockComponentKind;
}

#[derive(Component, Clone)]
pub struct BackgroundBlock;

impl BlockComponent for BackgroundBlock {
    fn kind(&self) -> BlockComponentKind {
        BlockComponentKind::Background
    }
}

#[derive(Component, Clone)]
pub struct FallingBlock;

impl BlockComponent for FallingBlock {
    fn kind(&self) -> BlockComponentKind {
        BlockComponentKind::Falling
    }
}

#[derive(Component, Clone)]
pub struct StaticBlock;

impl BlockComponent for StaticBlock {
    fn kind(&self) -> BlockComponentKind {
        BlockComponentKind::Static
    }
}

#[derive(Component, Clone)]
pub struct GhostBlock;

#[derive(Component)]
pub struct GhostPiece;

impl BlockComponent for GhostBlock {
    fn kind(&self) -> BlockComponentKind {
        BlockComponentKind::Ghost
    }
}

#[derive(Bundle)]
pub struct BlockBundle {
    #[bundle]
    pub(crate) sprite_bundle: SpriteBundle,
}

#[derive(Component)]
pub struct PieceController {
    pub(crate) falling_timer: Timer,
    pub(crate) placing_timer: Timer,
    pub(crate) hard_dropped: bool,
    pub(crate) used_hold: bool,
    pub(crate) movement_timer: Timer,
}

impl Default for PieceController {
    fn default() -> Self {
        let config = LevelConfig::default();

        Self {
            falling_timer: Timer::new(config.fall_duration, TimerMode::Repeating),
            placing_timer: Timer::new(config.placing_duration, TimerMode::Once),
            movement_timer: Timer::new(config.movement_duration, TimerMode::Repeating),
            hard_dropped: false,
            used_hold: false,
        }
    }
}

#[derive(Bundle)]
pub struct BoardBundle {
    pub(crate) board: Board,
    #[bundle]
    pub(crate) spatial_bundle: SpatialBundle,
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

impl BlockBundle {
    pub(crate) fn with_texture(
        config: &LevelConfig,
        texture_assets: &GameAssets,
        transform: Transform,
        color: Color,
    ) -> Self {
        return BlockBundle::new(
            config,
            transform,
            color,
            texture_assets.block_texture.clone(),
        );
    }

    pub(crate) fn transparent(config: &LevelConfig, transform: Transform, color: Color) -> Self {
        return BlockBundle::new(config, transform, color, DEFAULT_IMAGE_HANDLE.typed());
    }

    fn new(
        config: &LevelConfig,
        transform: Transform,
        color: Color,
        texture: Handle<Image>,
    ) -> Self {
        let sprite = Sprite {
            custom_size: Some(Vec2::new(config.block_size, config.block_size)),
            color,
            anchor: Anchor::BottomLeft,
            ..default()
        };

        Self {
            sprite_bundle: SpriteBundle {
                texture,
                sprite,
                transform,
                ..default()
            },
        }
    }
}

#[derive(Component, Default)]
pub struct PieceHolder {
    pub(crate) piece: Option<Piece>,
}

// spawn a block that is yet a part of a piece at the given cell
pub fn spawn_free_block(
    commands: &mut Commands,
    config: &LevelConfig,
    texture_assets: &Res<GameAssets>,
    cell: &Cell,
    block_component: impl BlockComponent,
) -> Entity {
    let (x, y) = cell.coords();

    let block_kind = block_component.kind();

    let color = match block_kind {
        BlockComponentKind::Falling | BlockComponentKind::Preview | BlockComponentKind::Static => {
            piece_color(cell.cell_kind().unwrap())
        }
        BlockComponentKind::Ghost => Color::GRAY.set_a(0.5).as_rgba(),
        BlockComponentKind::Background => Color::rgb(0.1, 0.1, 0.1),
    };

    let coords = Coords::from((x, y));
    let transform = Transform::from_coords(&coords, &config);

    let mut bundle = match block_kind {
        BlockComponentKind::Background => BlockBundle::transparent(&config, transform, color),
        _ => BlockBundle::with_texture(&config, &texture_assets, transform, color),
    };

    bundle.sprite_bundle.transform.translation.z = match block_kind {
        BlockComponentKind::Ghost => -0.1, // ghost is behind falling block
        BlockComponentKind::Background => -1., // background is behind everything
        _ => 0.,
    };

    // spawn falling block
    let entity = commands
        .spawn(bundle)
        .insert(coords)
        .insert(block_component)
        .id();
    entity
}

pub fn spawn_piece_blocks(
    mut commands: &mut Commands,
    config: &LevelConfig,
    texture_assets: &Res<GameAssets>,
    piece: &Piece,
    block_component: impl BlockComponent + Clone,
) -> Vec<Entity> {
    piece
        .board()
        .cells()
        .iter()
        .map(|&cell| {
            spawn_free_block(
                &mut commands,
                &config,
                &texture_assets,
                cell,
                block_component.clone(),
            )
        })
        .collect()
}

pub fn piece_color(piece_type: PieceType) -> Color {
    let color = match piece_type {
        PieceType::I => Color::rgb_u8(100, 196, 235), // cyan
        PieceType::J => Color::rgb_u8(90, 99, 165),   // orange
        PieceType::L => Color::rgb_u8(224, 127, 58),  // blue
        PieceType::O => Color::rgb_u8(241, 212, 72),  // yellow
        PieceType::S => Color::rgb_u8(100, 180, 82),  // green
        PieceType::T => Color::rgb_u8(161, 83, 152),  // purple
        PieceType::Z => Color::rgb_u8(216, 57, 52),   // red
    };
    color
}
