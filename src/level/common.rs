use crate::assets::GameAssets;
use crate::engine::{
    fall_duration, is_block_out, soft_drop_duration, Board, Cell, LockDownMode, MoveDirection,
    Piece, PieceGenerator, PieceType, LOCK_DOWN_SECONDS, MIN_LEVEL,
};
use bevy::color::Alpha;
use bevy::math::{IVec2, Vec2, Vec3};
use bevy::prelude::{
    Color, Commands, Component, Deref, DerefMut, Entity, Event, Message, Res, Resource, Sprite,
    SubStates, SystemSet, Timer, TimerMode, Transform,
};
use bevy::sprite::Anchor;
use bevy::state::state::StateSet;
use std::time::Duration;

use crate::{DespawnOnExit, GameState};

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

#[derive(SystemSet, Debug, Hash, PartialEq, Eq, Clone)]
pub(crate) enum LevelSystems {
    PieceSetup,
    PlayerInput,
}

#[derive(Resource, Debug)]
pub struct LevelConfig {
    pub(crate) block_size: f32,
    pub(crate) preview_scale: f32,
    pub(crate) board_width: usize,
    pub(crate) preview_count: usize,

    pub(crate) board_height: usize,
    pub(crate) das_delay: Duration,
    pub(crate) das_repeat_duration: Duration,
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
            das_delay: Duration::from_millis(300),
            das_repeat_duration: Duration::from_millis(50),
            soft_drop_duration: soft_drop_duration(MIN_LEVEL),
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
    pub(crate) soft_drop_timer: Timer,
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
            soft_drop_timer: Timer::new(config.soft_drop_duration, TimerMode::Repeating),
            hard_dropped: false,
            used_hold: false,
        }
    }
}

#[derive(Resource, Debug, Default)]
pub(crate) struct DasState {
    active_direction: Option<MoveDirection>,
    held_duration: Duration,
    repeat_elapsed: Duration,
}

impl DasState {
    pub(crate) fn active_direction(&self) -> Option<MoveDirection> {
        self.active_direction
    }

    pub(crate) fn next_action(
        &mut self,
        held_direction: Option<MoveDirection>,
        just_pressed: bool,
        delta: Duration,
        config: &LevelConfig,
    ) -> Option<MoveDirection> {
        let Some(direction) = held_direction else {
            self.active_direction = None;
            self.held_duration = Duration::ZERO;
            self.repeat_elapsed = Duration::ZERO;
            return None;
        };

        if self.active_direction != Some(direction) {
            self.active_direction = Some(direction);
            self.held_duration = Duration::ZERO;
            self.repeat_elapsed = Duration::ZERO;
            return just_pressed.then_some(direction);
        }

        if just_pressed {
            self.repeat_elapsed = Duration::ZERO;
            return Some(direction);
        }

        let was_waiting_for_delay = self.held_duration < config.das_delay;
        self.held_duration += delta;

        if was_waiting_for_delay {
            if self.held_duration >= config.das_delay {
                self.repeat_elapsed = Duration::ZERO;
                return Some(direction);
            }
            return None;
        }

        self.repeat_elapsed += delta;
        if self.repeat_elapsed >= config.das_repeat_duration {
            self.repeat_elapsed = self
                .repeat_elapsed
                .saturating_sub(config.das_repeat_duration);
            Some(direction)
        } else {
            None
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

pub(crate) fn spawn_coords_after_generation_rules(
    config: &LevelConfig,
    board: &Board,
    piece: &Piece,
) -> Option<(isize, isize)> {
    let spawn_coords = config.spawn_coords(piece);
    if is_block_out(piece, board, spawn_coords) {
        return None;
    }

    Some(
        piece
            .try_move(board, spawn_coords, MoveDirection::Down)
            .unwrap_or(spawn_coords),
    )
}

pub(crate) fn spawn_falling_piece(
    commands: &mut Commands,
    config: &LevelConfig,
    texture_assets: &Res<GameAssets>,
    piece: Piece,
    spawn_coords: (isize, isize),
    used_hold: bool,
) -> Entity {
    let block_ids =
        spawn_piece_blocks(commands, config, texture_assets, &piece, BlockKind::Falling);
    let piece_entity = commands
        .spawn((PieceState(piece), DespawnOnExit(GameState::InGame)))
        .id();

    commands
        .entity(piece_entity)
        .insert(Coords::from(spawn_coords))
        .insert(PieceController {
            used_hold,
            ..PieceController::new(config)
        })
        .insert(Transform::from_translation(to_translation(
            spawn_coords.0,
            spawn_coords.1,
            config.block_size,
        )));
    commands.entity(piece_entity).add_children(&block_ids);

    piece_entity
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::CellKind;

    #[test]
    fn das_tap_moves_once_immediately() {
        let config = LevelConfig::default();
        let mut das = DasState::default();

        assert_eq!(
            das.next_action(Some(MoveDirection::Left), true, Duration::ZERO, &config),
            Some(MoveDirection::Left)
        );
    }

    #[test]
    fn das_waits_for_initial_delay_before_repeating() {
        let config = LevelConfig::default();
        let mut das = DasState::default();
        das.next_action(Some(MoveDirection::Left), true, Duration::ZERO, &config);

        assert_eq!(
            das.next_action(
                Some(MoveDirection::Left),
                false,
                Duration::from_millis(299),
                &config
            ),
            None
        );
        assert_eq!(
            das.next_action(
                Some(MoveDirection::Left),
                false,
                Duration::from_millis(1),
                &config
            ),
            Some(MoveDirection::Left)
        );
    }

    #[test]
    fn das_repeats_at_repeat_interval_after_delay() {
        let config = LevelConfig::default();
        let mut das = DasState::default();
        das.next_action(Some(MoveDirection::Right), true, Duration::ZERO, &config);
        das.next_action(Some(MoveDirection::Right), false, config.das_delay, &config);

        assert_eq!(
            das.next_action(
                Some(MoveDirection::Right),
                false,
                Duration::from_millis(49),
                &config
            ),
            None
        );
        assert_eq!(
            das.next_action(
                Some(MoveDirection::Right),
                false,
                Duration::from_millis(1),
                &config
            ),
            Some(MoveDirection::Right)
        );
    }

    #[test]
    fn das_opposite_direction_press_restarts_delay_after_tap() {
        let config = LevelConfig::default();
        let mut das = DasState::default();
        das.next_action(Some(MoveDirection::Left), true, Duration::ZERO, &config);
        das.next_action(Some(MoveDirection::Left), false, config.das_delay, &config);

        assert_eq!(
            das.next_action(Some(MoveDirection::Right), true, Duration::ZERO, &config),
            Some(MoveDirection::Right)
        );
        assert_eq!(
            das.next_action(
                Some(MoveDirection::Right),
                false,
                Duration::from_millis(299),
                &config
            ),
            None
        );
    }

    #[test]
    fn das_releasing_one_of_two_directions_reapplies_delay() {
        let config = LevelConfig::default();
        let mut das = DasState::default();
        das.next_action(Some(MoveDirection::Left), true, Duration::ZERO, &config);

        assert_eq!(
            das.next_action(Some(MoveDirection::Right), false, Duration::ZERO, &config),
            None
        );
        assert_eq!(
            das.next_action(Some(MoveDirection::Right), false, config.das_delay, &config),
            Some(MoveDirection::Right)
        );
    }

    #[test]
    fn generation_rules_apply_immediate_drop_when_free() {
        let config = LevelConfig::default();
        let board = Board::with_top_margin(10, 20, 20);
        let piece = Piece::from(PieceType::T);

        assert_eq!(
            spawn_coords_after_generation_rules(&config, &board, &piece),
            Some((3, 18))
        );
    }

    #[test]
    fn generation_rules_keep_spawn_position_when_blocked_below() {
        let config = LevelConfig::default();
        let mut board = Board::with_top_margin(10, 20, 20);
        let piece = Piece::from(PieceType::T);
        assert!(board.set(4, 19, CellKind::Some(PieceType::O)));

        assert_eq!(
            spawn_coords_after_generation_rules(&config, &board, &piece),
            Some((3, 19))
        );
    }

    #[test]
    fn generation_rules_return_none_on_block_out() {
        let config = LevelConfig::default();
        let mut board = Board::with_top_margin(10, 20, 20);
        let piece = Piece::from(PieceType::T);
        assert!(board.set(4, 20, CellKind::Some(PieceType::O)));

        assert_eq!(
            spawn_coords_after_generation_rules(&config, &board, &piece),
            None
        );
    }
}
