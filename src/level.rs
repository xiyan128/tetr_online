use std::iter::zip;
use bevy::app::{App, Plugin};
use bevy::hierarchy::Children;
use bevy::log::debug;
use bevy::prelude::{BuildChildren, Bundle, Camera2dBundle, Color, Commands, Component, DespawnRecursiveExt, DetectChangesMut, Entity, IntoSystemAppConfig, IntoSystemConfig, IVec2, KeyCode, Mut, NextState, OnEnter, OnUpdate, Query, Res, ResMut, Resource, SpatialBundle, SpriteBundle, States, Time, Transform, Vec2, Vec3, With, Without};
use bevy::sprite::Sprite;
use bevy::time::{Timer, TimerMode};
use bevy::utils::default;
use crate::GameState;


use leafwing_input_manager::{Actionlike, InputManagerBundle};
use leafwing_input_manager::input_map::InputMap;
use leafwing_input_manager::prelude::{ActionState, InputManagerPlugin};
use crate::core::{Board, Cell, CellKind, MoveDirection, Piece, PieceGenerator, PieceType};

#[derive(States, PartialEq, Eq, Debug, Clone, Hash, Default)]
enum LevelState {
    // #[default]
    #[default]
    Idle,
    Falling,
    Placing,
}

#[derive(Resource)]
struct LevelConfig {
    block_size: f32,
    board_width: usize,
    board_height: usize,
    spawn_coords: (i32, i32),
}

impl Default for LevelConfig {
    fn default() -> Self {
        Self {
            block_size: 32.0,
            board_width: 10,
            board_height: 20,
            spawn_coords: (5, 20),
        }
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
}

impl Plugin for LevelPlugin {
    fn build(&self, app: &mut App) {
        app
            .add_state::<LevelState>()
            .add_plugin(InputManagerPlugin::<GameControl>::default())
            // setup
            .add_system(level_setup.in_schedule(OnEnter(GameState::InGame)))
            .add_system(piece_setup.in_schedule(OnEnter(LevelState::Falling)))

            // updates
            .add_system(piece_fall.in_set(OnUpdate(LevelState::Falling)))
            .add_system(piece_place.in_set(OnUpdate(LevelState::Placing)))

            // controls (TODO: also allow actions when placing)
            .add_system(handle_movements.in_set(OnUpdate(LevelState::Falling)))
            .add_system(handle_rotations.in_set(OnUpdate(LevelState::Falling)))

            // resources
            .init_resource::<LevelConfig>()
            .init_resource::<PieceGenerator>();
    }
}

#[derive(Component, PartialEq, Eq, Debug, Clone, Hash)]
struct Coords {
    x: i32,
    y: i32,
}

impl From<(i32, i32)> for Coords {
    fn from((x, y): (i32, i32)) -> Self {
        Self { x, y }
    }
}

impl Into<(i32, i32)> for Coords {
    fn into(self) -> (i32, i32) {
        (self.x, self.y)
    }
}


#[derive(Component)]
struct BackgroundBlock;

#[derive(Component)]
struct FallingBlock;

#[derive(Component)]
struct StaticBlock;

#[derive(Bundle)]
struct BlockBundle {
    #[bundle]
    sprite_bundle: SpriteBundle,
}


#[derive(Component)]
struct PieceController {
    falling_timer: Timer,
    placing_timer: Timer,
    hard_dropped: bool,
    movement_timer: Timer,
}

impl Default for PieceController {
    fn default() -> Self {
        Self {
            falling_timer: Timer::from_seconds(0.5, TimerMode::Repeating),
            placing_timer: Timer::from_seconds(0.5, TimerMode::Once),
            movement_timer: Timer::from_seconds(0.1, TimerMode::Repeating),
            hard_dropped: false,
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

fn to_translation(x: i32, y: i32, config: &LevelConfig) -> Vec3 {
    IVec2::new(
        x,
        y,
    ).as_vec2().extend(0.0) * config.block_size
}

impl MatchCoords for Transform {
    fn from_coords(coords: &Coords, config: &LevelConfig) -> Self {
        Transform::from_translation(to_translation(coords.x, coords.y, &config))
    }
    fn update_coords(&mut self, coords: &Coords, config: &LevelConfig) {
        self.translation = to_translation(coords.x, coords.y, &config);
    }
}


impl BlockBundle {
    fn new(config: &LevelConfig, transform: Transform, color: Color) -> Self {
        let sprite = Sprite {
            custom_size: Some(Vec2::new(config.block_size, config.block_size)),
            color,
            ..default()
        };

        Self {
            sprite_bundle: SpriteBundle {
                sprite,
                transform,
                ..default()
            },
        }
    }
}

fn level_setup(
    mut commands: Commands,
    config: Res<LevelConfig>,
    mut next_state: ResMut<NextState<LevelState>>,
) {
    debug!("level_setup");

    let board = Board::new(config.board_width, config.board_height);


    let mut block_ids = Vec::new();
    for (x, y) in board.coords()
    {
        block_ids.push(if let Some(cell) = board.get(x, y) {
            spawn_static_block(&mut commands, &config, cell)
        } else {
            let coords = Coords::from((x, y));
            let transform = Transform::from_coords(&coords, &config);

            let bundle = BlockBundle::new(&config, transform, Color::rgb(0.1, 0.1, 0.1));

            // spawn background block
            commands.spawn((bundle, BackgroundBlock)).id()
        });
    }

    let board_entity = commands.spawn(BoardBundle {
        board,
        spatial_bundle: SpatialBundle {
            transform: Transform::from_translation(Vec3::new(0., 0., 0.)),
            ..default()
        },
    }).id();

    commands.entity(board_entity).push_children(&block_ids);

    // look at center of the board
    commands.spawn(Camera2dBundle {
        transform: Transform::from_translation(Vec3::new(
            config.block_size * config.board_width as f32 / 2. - config.block_size / 2.,
            config.block_size * config.board_height as f32 / 2. - config.block_size / 2.,
            1.0,
        )),
        ..default()
    });

    // switch to falling state
    next_state.set(LevelState::Falling);
}

fn spawn_static_block(commands: &mut Commands, config: &Res<LevelConfig>, cell: &Cell) -> Entity {
    let piece_type = cell.cell_kind();
    let color = piece_color(piece_type.as_some().unwrap());

    let coords = Coords::from(cell.coords());
    let transform = Transform::from_coords(&coords, &config);

    let bundle = BlockBundle::new(&config, transform, color);

    // spawn static block
    commands.spawn((bundle, StaticBlock)).id()
}

fn piece_setup(mut commands: Commands,
               config: Res<LevelConfig>,
               mut generator: ResMut<PieceGenerator>) {
    debug!("piece_setup");
    let next_piece_type = generator.next();
    let piece = Piece::from(next_piece_type.unwrap());

    let block_ids: Vec<Entity> = piece.board().cells().iter().map(|cell| {
        spawn_falling_block(&mut commands, &config, &piece, cell)
    }).collect();

    let piece_entity = commands
        .spawn(piece)
        .insert(
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
                ]),
                ..default()
            }).id();

    commands.entity(piece_entity)
        .insert(Coords::from(config.spawn_coords))
        .insert(PieceController::default())
        .insert(SpatialBundle {
            transform: Transform::from_translation(to_translation(config.spawn_coords.0, config.spawn_coords.1, &config)),
            ..default()
        });

    commands.entity(piece_entity).push_children(&block_ids);
}

fn spawn_falling_block(commands: &mut Commands, config: &Res<LevelConfig>, piece: &Piece, cell: &&Cell) -> Entity {
    let (x, y) = cell.coords();
    let piece_type = piece.board().get_cell_kind(x, y).unwrap();

    let color = piece_color(piece_type);

    let coords = Coords::from((x, y));
    let transform = Transform::from_coords(&coords, &config);

    let bundle = BlockBundle::new(&config, transform, color);
    // spawn falling block
    let entity = commands.spawn(bundle)
        .insert(coords)
        .insert(FallingBlock)
        .id();
    entity
}

pub fn piece_color(piece_type: PieceType) -> Color {
    let color = match piece_type {
        PieceType::I => Color::BLUE,
        PieceType::J => Color::GREEN,
        PieceType::O => Color::YELLOW,
        PieceType::L => Color::ORANGE,
        PieceType::S => Color::RED,
        PieceType::T => Color::PURPLE,
        PieceType::Z => Color::PINK,
    };
    color
}

fn piece_fall(
    mut query: Query<(&mut Piece, &mut PieceController, &mut Coords, &mut Transform)>,
    mut board_query: Query<&mut Board>,
    config: Res<LevelConfig>,
    time: Res<Time>,
    mut next_state: ResMut<NextState<LevelState>>,
) {
    let (mut piece, mut piece_controller, mut coords, mut transform) = query.single_mut();
    let board = board_query.single_mut();

    piece_controller.falling_timer.tick(time.delta());

    if piece_controller.falling_timer.finished() {
        if let Ok(new_coords) = piece.try_move(&board, (coords.x, coords.y), MoveDirection::Down) {
            (coords.x, coords.y) = new_coords;
            transform.update_coords(coords.as_ref(), &config);
        } else {
            // switch to piece placement state
            next_state.set(LevelState::Placing);
        }
    }
}


fn piece_place(
    mut commands: Commands,
    mut query: Query<(Entity, &Piece, &Coords, &mut PieceController, &Children)>,
    mut board_query: Query<(Entity, &mut Board, &Children)>,
    mut query_children: Query<(&mut Coords, &mut Transform), (With<FallingBlock>, Without<StaticBlock>, Without<Piece>)>,
    mut query_static_blocks: Query<Entity, With<StaticBlock>>,
    // mut query_children: Query<(&mut Coords, &mut Transform), Without<PieceController>>,
    time: Res<Time>,
    config: Res<LevelConfig>,
    mut next_state: ResMut<NextState<LevelState>>,
) {
    let (piece_entity, piece, coords, mut piece_controller, children) = query.single_mut();
    let (board_entity, mut board, board_children) = board_query.single_mut();

    piece_controller.placing_timer.tick(time.delta());

    if piece_controller.placing_timer.finished()
        || piece_controller.hard_dropped // after hard drop, place immediately
    {
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

        for child in board_children.iter() {
            commands.entity(*child).log_components();
        }

        if line_clear_count > 0 {
            // remove all static blocks
            for entity in query_static_blocks.iter_mut() {
                commands.entity(entity).despawn_recursive();
            }
            // also remove the falling piece children
            for child in children.iter() {
                commands.entity(*child).despawn_recursive();
            }

            for cell in board.cells() {
                spawn_static_block(&mut commands, &config, cell); // redraw all static blocks
            }

            piece_controller.placing_timer.reset();
        }
    }
}


fn handle_movements(
    action_query: Query<&ActionState<GameControl>, With<Piece>>,
    mut query: Query<(&mut Piece, &mut Coords, &mut Transform, &mut PieceController)>,
    mut board_query: Query<&mut Board>,
    config: Res<LevelConfig>,
    time: Res<Time>,
) {
    let action_state = action_query.single();


    let movement = IVec2::new(
        action_state.pressed(GameControl::Right) as i32
            - action_state.pressed(GameControl::Left) as i32,
        action_state.pressed(GameControl::Down) as i32);

    let hard_drop = action_state.just_pressed(GameControl::HardDrop);

    if hard_drop {
        debug!("hard drop")
    }

    let movement =
        match movement {
            IVec2 { x: 0, y: 1 } => Some(MoveDirection::Down),
            IVec2 { x: 1, y: 0 } => Some(MoveDirection::Right),
            IVec2 { x: -1, y: 0 } => Some(MoveDirection::Left),
            _ => hard_drop.then(|| MoveDirection::Down),
        };

    let (mut piece, mut coords, mut transform, mut piece_controller) = query.single_mut();
    let board = board_query.single_mut();

    piece_controller.movement_timer.tick(time.delta());

    if piece_controller.movement_timer.finished() || hard_drop {
        if let Some(movement) = movement {
            if hard_drop {
                while move_piece_and_update(&config, &mut piece, &mut coords, &mut transform, &board, movement) {}
                piece_controller.hard_dropped = true;
            } else {
                move_piece_and_update(&config, &mut piece, &mut coords, &mut transform, &board, movement);
            }
            // move_piece_and_update(&config, &mut piece, coords, &mut transform, &board, movement);
        }
        piece_controller.movement_timer.reset();
    }
}

fn move_piece_and_update(config: &Res<LevelConfig>,
                         piece: &mut Mut<Piece>,
                         coords: &mut Mut<'_, Coords>,
                         transform: &mut Mut<Transform>,
                         board: &Mut<Board>,
                         movement: MoveDirection) -> bool {
    if let Ok(new_coords) = piece.try_move(&board, (coords.x, coords.y), movement) {
        (coords.x, coords.y) = new_coords;
        transform.update_coords(coords.as_ref(), &config);
        true
    } else {
        false
    }
}

fn handle_rotations(
    action_query: Query<&ActionState<GameControl>, With<Piece>>,
    mut query: Query<(&mut Piece, &mut Coords, &mut Transform, &Children)>,
    mut children_query: Query<(&mut Coords, &mut Transform), Without<Piece>>,
    mut board_query: Query<&mut Board>,
    config: Res<LevelConfig>,
) {
    let action_state = action_query.single();


    let rotation = action_state.just_pressed(GameControl::RotateClockwise) as i32
        - action_state.just_pressed(GameControl::RotateCounterClockwise) as i32;

    let rotation = match rotation {
        1 => Some(1),
        -1 => Some(3),
        _ => None,
    };


    let (mut piece, coords, mut transform, children) = query.single_mut();
    let board = board_query.single_mut();

    if let Some(rotation_n) = rotation {
        if let Ok(rotation) = piece.try_rotate(&board, (coords.x, coords.y), rotation_n) {
            piece.rotate_to(rotation);

            for (child_entity, cell) in zip(children.iter(), piece.board().cells().iter()) {
                let (mut child_coords, mut child_transform) = children_query.get_mut(*child_entity).unwrap();
                child_coords.set_if_neq(Coords::from(cell.coords()));
                child_transform.update_coords(child_coords.as_ref(), &config);
            }

            transform.update_coords(coords.as_ref(), &config);
        }
    }
}