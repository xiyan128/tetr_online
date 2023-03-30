use std::default::Default;
use std::time::Duration;
use bevy::app::{App, Plugin};
use bevy::core_pipeline::bloom::BloomSettings;
use bevy::core_pipeline::tonemapping::Tonemapping;
use bevy::hierarchy::Children;
use bevy::prelude::*;
use bevy::render::texture::DEFAULT_IMAGE_HANDLE;
use bevy::sprite::{Anchor, Sprite};
use bevy::time::{Timer, TimerMode};
use bevy::utils::default;
use crate::{GameState};


use leafwing_input_manager::prelude::*;
use crate::core::{Board, Cell, CellKind, MoveDirection, Piece, PieceGenerator, PieceType};
use bevy_asset_loader::prelude::*;
use crate::assets::GameAssets;
use crate::level::actions::ActionsPlugin;
use crate::level::game_over::GameOverPlugin;
use crate::level::ui::UIPlugin;

mod actions;
mod game_over;
mod ui;

#[derive(States, PartialEq, Eq, Debug, Clone, Hash, Default)]
enum LevelState {
    // #[default]
    #[default]
    Ready,
    Falling,
    Placing,
    GameOver,
}

#[derive(Resource, Debug)]
pub struct LevelConfig {
    pub(crate) block_size: f32,
    pub(crate) preview_size: f32,
    pub(crate) board_width: usize,
    pub(crate) preview_count: usize,

    board_height: usize,
    movement_duration: Duration,
    movement_speedup: f64,
    soft_drop_duration: Duration,
    fall_duration: Duration,
    placing_duration: Duration,
}

impl Default for LevelConfig {
    fn default() -> Self {
        Self {
            block_size: 32.0,
            preview_size: 32.0 * 3.0,
            board_width: 10,
            board_height: 20,
            preview_count: 3,
            movement_duration: Duration::from_millis(200),
            soft_drop_duration: Duration::from_millis(50),
            movement_speedup: 1. / 1.0_f64.exp(),
            fall_duration: Duration::from_millis(500),
            placing_duration: Duration::from_millis(500),
        }
    }
}

impl LevelConfig {
    fn spawn_coords(&self, piece: &Piece) -> (isize, isize) {
        let (offset_x, offset_y) = piece.board_size();
        (
            self.board_width as isize / 2 - offset_x as isize / 2,
            self.board_height as isize
        )
    }
}


pub struct LevelPlugin;

#[derive(Actionlike, PartialEq, Eq, Clone, Copy, Hash, Debug)]
pub enum GameControl {
    Up,
    Down,
    Left,
    Right,
    HardDrop,
    RotateClockwise,
    RotateCounterClockwise,
    Hold,
}

impl From<MoveDirection> for GameControl {
    fn from(direction: MoveDirection) -> Self {
        match direction {
            MoveDirection::Down => Self::Down,
            MoveDirection::Left => Self::Left,
            MoveDirection::Right => Self::Right,
        }
    }
}

#[derive(SystemSet, Debug, Hash, PartialEq, Eq, Clone)]
enum LevelSystemSet {
    Control,
    Logic,
    Detection,
}

#[derive(Component)]
struct LevelCleanup;


impl Plugin for LevelPlugin {
    fn build(&self, app: &mut App) {
        app
            .add_state::<LevelState>()
            .add_plugin(ActionsPlugin)
            .add_plugin(GameOverPlugin)
            .add_plugin(UIPlugin)
            .add_collection_to_loading_state::<_, GameAssets>(GameState::Loading)

            .configure_set(LevelSystemSet::Control.before(LevelSystemSet::Logic))
            .configure_set(LevelSystemSet::Logic.before(LevelSystemSet::Detection))

            .add_plugin(InputManagerPlugin::<GameControl>::default())

            // setup
            .add_system(level_setup.in_schedule(OnEnter(GameState::InGame)))

            .add_system(piece_setup.in_schedule(OnEnter(LevelState::Falling)))
            .add_system(level_cleanup.in_schedule(OnEnter(LevelState::GameOver)))
            // updates
            .add_systems((piece_fall, detect_placement, ghost_blocks)
                .in_set(OnUpdate(LevelState::Falling))
            )

            .add_systems(
                (piece_place, detect_placement)
                    .in_set(OnUpdate(LevelState::Placing))
            )


            // resources
            .init_resource::<LevelConfig>()
            .init_resource::<PieceGenerator>();
    }
}

#[derive(Component, PartialEq, Eq, Debug, Clone, Hash)]
pub(crate) struct Coords {
    x: isize,
    y: isize,
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
enum BlockComponentKind {
    Background,
    Falling,
    Static,
    Ghost,
    Preview,
}

trait BlockComponent: Component {
    fn kind(&self) -> BlockComponentKind;
}

#[derive(Component)]
struct BackgroundBlock;

impl BlockComponent for BackgroundBlock {
    fn kind(&self) -> BlockComponentKind {
        BlockComponentKind::Background
    }
}

#[derive(Component)]
struct FallingBlock;

impl BlockComponent for FallingBlock {
    fn kind(&self) -> BlockComponentKind {
        BlockComponentKind::Falling
    }
}

#[derive(Component)]
struct StaticBlock;

impl BlockComponent for StaticBlock {
    fn kind(&self) -> BlockComponentKind {
        BlockComponentKind::Static
    }
}

#[derive(Component)]
struct GhostBlock;

#[derive(Component)]
struct GhostPiece;

impl BlockComponent for GhostBlock {
    fn kind(&self) -> BlockComponentKind {
        BlockComponentKind::Ghost
    }
}

#[derive(Bundle)]
struct BlockBundle {
    #[bundle]
    sprite_bundle: SpriteBundle,

    // level_cleanup: LevelCleanup,
}


#[derive(Component)]
pub struct PieceController {
    falling_timer: Timer,
    pub(crate) placing_timer: Timer,
    hard_dropped: bool,
    used_hold: bool,
    movement_timer: Timer,
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
struct BoardBundle {
    board: Board,
    #[bundle]
    spatial_bundle: SpatialBundle,
}

trait MatchCoords {
    fn from_coords(coords: &Coords, config: &LevelConfig) -> Self;
    fn update_coords(&mut self, coords: &Coords, config: &LevelConfig);
}

fn to_translation(x: isize, y: isize, block_size: f32) -> Vec3 {
    IVec2::new(
        x as i32,
        y as i32,
    ).as_vec2().extend(0.0) * block_size
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
    fn with_texture(config: &LevelConfig, texture_assets: &GameAssets, transform: Transform, color: Color) -> Self {
        return BlockBundle::new(config, transform, color, texture_assets.block_texture.clone());
    }

    fn transparent(config: &LevelConfig, color: Color, transform: Transform) -> Self {
        return BlockBundle::new(config, transform, color, DEFAULT_IMAGE_HANDLE.typed());
    }

    fn new(config: &LevelConfig, transform: Transform, color: Color, texture: Handle<Image>) -> Self {
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

fn level_cleanup(
    mut commands: Commands,
    mut query: Query<Entity, With<LevelCleanup>>,
) {
    info!("level_cleanup");

    for entity in query.iter_mut() {
        commands.entity(entity).despawn_recursive();
    }
}


fn level_setup(
    mut commands: Commands,
    config: Res<LevelConfig>,
    mut next_state: ResMut<NextState<LevelState>>,
    texture_assets: Res<GameAssets>,
) {
    info!("level_setup");

    let board = Board::with_top_margin(config.board_width, config.board_height, 10);


    let mut block_ids = Vec::new();
    for (x, y) in board.coords()
    {
        block_ids.push(if let Some(cell) = board.get(x, y) {
            spawn_static_block(&mut commands, &config, &texture_assets, cell)
        } else {
            let coords = Coords::from((x, y));
            let mut transform = Transform::from_coords(&coords, &config);

            transform.translation.z = -1.0; // background block should be behind the board

            // background block
            let bundle = BlockBundle::transparent(&config, Color::rgb(0.1, 0.1, 0.1), transform);

            // spawn background block
            commands.spawn((bundle, BackgroundBlock)).id()
        });
    }

    info!("config: {:?}", config);

    let board_entity = commands.spawn(BoardBundle {
        board,
        spatial_bundle: SpatialBundle {
            transform: Transform::from_translation(Vec3::new(0., 0., 0.)),
            ..default()
        },
    }).insert(
        InputManagerBundle::<GameControl> {
            // Stores "which actions are currently pressed"
            action_state: ActionState::default(),
            // Describes how to convert from player inputs into those actions
            input_map: InputMap::new([
                (KeyCode::Space, GameControl::HardDrop),
                (KeyCode::Left, GameControl::Left),
                (KeyCode::Right, GameControl::Right),
                (KeyCode::Up, GameControl::RotateClockwise),
                (KeyCode::Z, GameControl::RotateCounterClockwise),
                (KeyCode::X, GameControl::RotateClockwise),
                (KeyCode::Down, GameControl::Down),
                (KeyCode::LShift, GameControl::Hold),
            ]),
            ..default()
        })
        .insert(LevelCleanup)
        .insert(PieceHolder::default())
        .id();

    commands.entity(board_entity).push_children(&block_ids);

    // look at center of the board
    commands.spawn((Camera2dBundle {
        transform: Transform::from_translation(Vec3::new(
            config.block_size * config.board_width as f32 / 2.,
            config.block_size * config.board_height as f32 / 2.,
            1.0,
        )),
        ..default()
    }, LevelCleanup)
    );

    // switch to falling state
    next_state.set(LevelState::Falling);
}

fn spawn_static_block(commands: &mut Commands, config: &Res<LevelConfig>, assets: &Res<GameAssets>, cell: &Cell) -> Entity {
    let piece_type = cell.cell_kind();
    let color = piece_color(piece_type.as_some().unwrap());

    let coords = Coords::from(cell.coords());
    let transform = Transform::from_coords(&coords, &config);

    let bundle = BlockBundle::with_texture(&config, &assets, transform, color);

    // spawn static block
    commands.spawn((bundle, StaticBlock)).id()
}

#[derive(Component, Default)]
pub struct PieceHolder {
    piece: Option<Piece>,
}

fn piece_setup(mut commands: Commands,
               config: Res<LevelConfig>,
               texture_assets: Res<GameAssets>,
               piece_query: Query<(Entity, &Piece, &PieceController)>,
               board_query: Query<&Board>,
               mut next_state: ResMut<NextState<LevelState>>,
               mut next_game_state: ResMut<NextState<GameState>>,
               mut generator: ResMut<PieceGenerator>) {

    // if there is already a piece, don't spawn a new one
    if piece_query.get_single().is_ok() {
        info!("piece already exists");
        return;
    }

    info!("piece_setup");
    let next_piece_type = generator.next();
    info!("preview: {:?}", generator.preview());
    let piece = Piece::from(next_piece_type.unwrap());
    let spawn_coords = config.spawn_coords(&piece);

    let board = board_query.single();

    // test collision and game over
    if piece.collide_with(&board, spawn_coords) {
        info!("game over");
        next_state.set(LevelState::GameOver);
        next_game_state.set(GameState::GameOver);
        return;
    }

    let block_ids: Vec<Entity> = piece.board().cells().iter().map(|&cell| {
        spawn_free_block(&mut commands, &config, &texture_assets, &piece, cell, FallingBlock)
    }).collect();


    let piece_entity = commands
        .spawn(piece)
        .id();


    commands.entity(piece_entity)
        .insert(Coords::from(spawn_coords))
        .insert(PieceController::default())
        .insert(SpatialBundle {
            transform: Transform::from_translation(to_translation(spawn_coords.0, spawn_coords.1, config.block_size)),
            ..default()
        });

    commands.entity(piece_entity).push_children(&block_ids);
}

fn spawn_free_block(commands: &mut Commands,
                    config: &LevelConfig,
                    texture_assets: &Res<GameAssets>,
                    piece: &Piece,
                    cell: &Cell,
                    block_component: impl BlockComponent,
) -> Entity {
    let (x, y) = cell.coords();
    let piece_type = cell.cell_kind().unwrap();

    let color = match block_component.kind() {
        BlockComponentKind::Falling | BlockComponentKind::Preview => piece_color(piece_type),
        BlockComponentKind::Ghost => Color::GRAY.set_a(0.5).as_rgba(),
        BlockComponentKind::Static => piece_color(piece_type),
        _ => Color::rgb(0.2, 0.2, 0.2),
    };

    let coords = Coords::from((x, y));
    let transform = Transform::from_coords(&coords, &config);

    let mut bundle = BlockBundle::with_texture(&config, &texture_assets, transform, color);

    if block_component.kind() == BlockComponentKind::Ghost {
        bundle.sprite_bundle.transform.translation.z = -0.1;
    }

    // spawn falling block
    let entity = commands.spawn(bundle)
        .insert(coords)
        .insert(block_component)
        .id();
    entity
}

pub fn piece_color(piece_type: PieceType) -> Color {
    let color = match piece_type {
        PieceType::I => Color::rgb_u8(100, 196, 235), // cyan
        PieceType::J => Color::rgb_u8(90, 99, 165), // orange
        PieceType::L => Color::rgb_u8(224, 127, 58), // blue
        PieceType::O => Color::rgb_u8(241, 212, 72), // yellow
        PieceType::S => Color::rgb_u8(100, 180, 82), // green
        PieceType::T => Color::rgb_u8(161, 83, 152), // purple
        PieceType::Z => Color::rgb_u8(216, 57, 52), // red
    };
    color
}

fn piece_fall(
    mut query: Query<(&mut Piece, &mut PieceController, &mut Coords, &mut Transform)>,
    mut board_query: Query<&mut Board>,
    config: Res<LevelConfig>,
    time: Res<Time>,
) {
    if query.get_single_mut().is_err() {
        return;
    }

    let (mut piece, mut piece_controller, mut coords, mut transform) = query.single_mut();
    let board = board_query.single_mut();

    piece_controller.falling_timer.tick(time.delta());

    if piece_controller.falling_timer.finished() {
        if let Ok(new_coords) = piece.try_move(&board, (coords.x, coords.y), MoveDirection::Down) {
            (coords.x, coords.y) = new_coords;
            transform.update_coords(coords.as_ref(), &config);
        }
    }
}

fn piece_place(
    mut commands: Commands,
    mut query: Query<(Entity, &Piece, &Coords, &mut PieceController, &Children)>,
    mut board_query: Query<(Entity, &mut Board, &Children)>,
    mut query_children: Query<(&mut Coords, &mut Transform), (With<FallingBlock>, Without<StaticBlock>, Without<Piece>)>,
    mut query_static_blocks: Query<Entity, With<StaticBlock>>,
    time: Res<Time>,
    config: Res<LevelConfig>,
    assets: Res<GameAssets>,
    audio: Res<Audio>,
    mut next_state: ResMut<NextState<LevelState>>,
) {
    let (piece_entity, piece, coords, mut piece_controller, children) = query.single_mut();
    let (board_entity, mut board, board_children) = board_query.single_mut();

    piece_controller.placing_timer.tick(time.delta());

    if piece_controller.placing_timer.finished()
        || piece_controller.hard_dropped // after hard drop, place immediately
    {
        info!("piece_place");
        piece_controller.hard_dropped = false; // reset hard drop flag
        audio.play(assets.hard_drop_sound.clone());
        // switch to piece setup state and finalize piece
        next_state.set(LevelState::Falling);

        // hand over children to board
        commands.entity(board_entity).push_children(children);

        commands.entity(piece_entity).despawn_recursive();

        for child in children.iter() {
            // convert the coordinates of the child to board coordinates
            let (mut child_coords, mut child_transform) = query_children.get_mut(*child).unwrap();
            child_coords.x += coords.x;
            child_coords.y += coords.y;

            board.set(child_coords.x, child_coords.y, CellKind::Some(piece.piece_type()));

            commands.entity(*child).insert(StaticBlock {}).remove::<FallingBlock>();
            child_transform.update_coords(child_coords.as_ref(), &config);
        }

        // check for line clears
        let line_clear_count = board.clear_lines();

        if line_clear_count > 0 {
            // remove all static blocks
            for entity in query_static_blocks.iter_mut() {
                commands.entity(entity).despawn_recursive();
            }

            // remove the current piece's children (which are added to the board but yet to be updated)
            for entity in children.iter() {
                commands.entity(*entity).despawn_recursive();
            }


            for cell in board.cells() {
                spawn_static_block(&mut commands, &config, &assets, cell);
            }

            piece_controller.placing_timer.reset();
        }
    }
}

fn ghost_blocks(
    mut commands: Commands,
    mut query: Query<(&Coords, &Children, &Piece), (With<PieceController>, Or<(Changed<Coords>, Changed<Piece>)>)>, // either the piece or its coords changed
    ghost_query: Query<Entity, With<GhostPiece>>,
    mut board_query: Query<&mut Board>,
    config: Res<LevelConfig>,
    texture_assets: Res<GameAssets>,
) {
    if query.is_empty() {
        return;
    }

    for entity in ghost_query.iter() {
        commands.entity(entity).despawn_recursive();
    }

    let (coords, children, piece) = query.single_mut();
    let board = board_query.single_mut();

    let mut ghost_coords = coords.clone();
    let mut ghost_transform = Transform::default();
    let mut ghost_piece = piece.clone();

    let mut can_move = false;

    while let Ok(new_coords) = ghost_piece.try_move(&board, ghost_coords.into(), MoveDirection::Down) {
        ghost_coords = Coords::from(new_coords);

        ghost_transform.update_coords(&ghost_coords, &config);
        can_move = true;
    }

    if !can_move {
        return;
    }

    let block_entities = ghost_piece.board().cells().iter().map(|cell| {
        spawn_free_block(&mut commands, &config, &texture_assets, &ghost_piece, cell, GhostBlock)
    }).collect::<Vec<_>>();

    let piece_entity = commands
        .spawn(ghost_piece)
        .insert(GhostPiece)
        .insert(SpatialBundle {
            transform: ghost_transform,
            ..Default::default()
        })
        .id();
    commands.entity(piece_entity).push_children(&block_entities);
}


fn detect_placement(
    mut query: Query<(&Coords, &Children, &Piece), Or<(Changed<Coords>, Changed<Piece>)>>, // either the piece or its coords changed
    mut board_query: Query<&mut Board>,
    mut next_state: ResMut<NextState<LevelState>>,
    mut current_state: Res<State<LevelState>>,
) {
    if query.is_empty() {
        return;
    }

    let (coords, children, piece) = query.single_mut();
    let board = board_query.single_mut();

    let current_state = &current_state.0;

    if piece.try_move(&board, (coords.x, coords.y), MoveDirection::Down).is_err() {
        if current_state == &LevelState::Falling {
            next_state.set(LevelState::Placing);
            info!("Transitioning to Placing state.");
        }
    } else if current_state == &LevelState::Placing {
        next_state.set(LevelState::Falling);
        info!("Transitioning to Falling state.");
    }
}
