use bevy::prelude::*;

use crate::core::*;
use crate::utils::*;
use common::*;

use crate::assets::GameAssets;
use crate::level::actions::ActionsPlugin;
use crate::level::common::PlayingState::Falling;
use crate::level::game_over::GameOverPlugin;
use crate::level::score::ScorePlugin;
use crate::level::sound_effects::SoundEffectsPlugin;
use crate::level::ui::UIPlugin;
use crate::GameState;

mod actions;
mod common;
mod game_over;
mod score;
mod setup;
mod sound_effects;
mod ui;

pub struct LevelPlugin;

impl Plugin for LevelPlugin {
    fn build(&self, app: &mut App) {
        app
            // states
            .init_state::<LevelState>()
            .init_state::<PlayingState>()
            // events
            .add_message::<ActionEvent>()
            .add_message::<PlacingEvent>()
            // plugins
            .add_plugins(ActionsPlugin)
            .add_plugins(GameOverPlugin)
            .add_plugins(SoundEffectsPlugin)
            .add_plugins(ScorePlugin)
            .add_plugins(UIPlugin)
            // resources
            .init_resource::<LevelConfig>()
            // enter setup immediately in game
            .add_systems(
                OnEnter(GameState::InGame),
                continue_to_state(LevelState::Setup),
            )
            // setup
            .add_systems(
                OnEnter(LevelState::Setup),
                (level_cleanup, level_setup).chain(),
            )
            .add_systems(OnEnter(LevelState::Playing), piece_setup)
            // updates
            .add_systems(
                Update,
                (piece_fall, detect_placement, ghost_blocks).run_if(in_state(LevelState::Playing)),
            )
            .add_systems(Update, piece_lock.run_if(in_state(PlayingState::Locking)));
    }
}

// despawn recursively all entities with the LevelCleanup component
fn level_cleanup(mut commands: Commands, mut query: Query<Entity, With<LevelCleanup>>) {
    for entity in query.iter_mut() {
        commands.entity(entity).despawn();
    }
}

// setup board and camera
fn level_setup(
    mut commands: Commands,
    config: Res<LevelConfig>,
    mut next_state: ResMut<NextState<LevelState>>,
    texture_assets: Res<GameAssets>,
) {
    info!("level_setup");

    // Play field is 10×40, where rows above 20 are hidden or obstructed by the field frame to trick
    // the player into thinking it's 10×20.
    let board = Board::with_top_margin(config.board_width, config.board_height, 20);

    let mut block_ids = Vec::new();
    for (x, y) in board.coords() {
        let cell = Cell::new(x, y, CellKind::None);

        // spawn background block
        let block_id = spawn_free_block(
            &mut commands,
            &config,
            &texture_assets,
            &cell,
            BackgroundBlock,
        );
        block_ids.push(block_id);
    }

    let board_entity = commands
        .spawn(BoardBundle {
            board,
            transform: Transform::default(),
        })
        .insert(LevelCleanup)
        .insert(PieceHolder::default())
        .insert(PieceGenerator::default())
        .id();

    commands.entity(board_entity).add_children(&block_ids);

    // look at center of the board
    commands.spawn((
        Camera2d,
        Transform::from_translation(Vec3::new(
            config.block_size * config.board_width as f32 / 2.,
            config.block_size * config.board_height as f32 / 2.,
            1.0,
        )),
        LevelCleanup,
    ));

    // switch to falling state
    next_state.set(LevelState::Playing);
}

fn spawn_static_block(
    commands: &mut Commands,
    config: &Res<LevelConfig>,
    assets: &Res<GameAssets>,
    cell: &Cell,
) -> Entity {
    let piece_type = cell.cell_kind();
    let color = piece_color(piece_type.as_some().unwrap());

    let coords = Coords::from(cell.coords());
    let transform = Transform::from_coords(&coords, &config);

    let bundle = BlockBundle::with_texture(&config, &assets, transform, color);

    // spawn static block
    commands.spawn((bundle, StaticBlock)).id()
}

fn piece_setup(
    mut commands: Commands,
    config: Res<LevelConfig>,
    texture_assets: Res<GameAssets>,
    piece_query: Query<(Entity, &Piece, &PieceController)>,
    board_query: Query<&Board>,
    mut next_level_state: ResMut<NextState<LevelState>>,
    mut next_game_state: ResMut<NextState<GameState>>,
    mut next_playing_state: ResMut<NextState<PlayingState>>,
    mut generator_query: Query<&mut PieceGenerator>,
) {
    // if there is already a piece, don't spawn a new one
    if piece_query.single().is_ok() {
        info!("piece already exists");
        return;
    }
    let Ok(mut generator) = generator_query.single_mut() else {
        return;
    };

    info!("piece_setup");
    let next_piece_type = generator.next();
    info!("preview: {:?}", generator.preview());
    let piece = Piece::from(next_piece_type.unwrap());
    let spawn_coords = config.spawn_coords(&piece);

    let Ok(board) = board_query.single() else {
        return;
    };

    // test collision and game over
    if piece.collide_with(&board, spawn_coords) {
        info!("game over");
        next_level_state.set(LevelState::GameOver);
        next_playing_state.set(PlayingState::default());
        next_game_state.set(GameState::GameOver);
        return;
    }

    let block_ids: Vec<Entity> = spawn_piece_blocks(
        &mut commands,
        &config,
        &texture_assets,
        &piece,
        FallingBlock,
    );

    let piece_entity = commands.spawn(piece).id();

    commands
        .entity(piece_entity)
        .insert(Coords::from(spawn_coords))
        .insert(PieceController::default())
        .insert(Transform::from_translation(common::to_translation(
            spawn_coords.0,
            spawn_coords.1,
            config.block_size,
        )));

    commands.entity(piece_entity).add_children(&block_ids);
}

fn piece_fall(
    mut query: Query<(
        &mut Piece,
        &mut PieceController,
        &mut Coords,
        &mut Transform,
    )>,
    mut board_query: Query<&mut Board>,
    config: Res<LevelConfig>,
    time: Res<Time>,
) {
    let Ok((piece, mut piece_controller, mut coords, mut transform)) = query.single_mut() else {
        return;
    };
    let Ok(board) = board_query.single_mut() else {
        return;
    };

    piece_controller.falling_timer.tick(time.delta());

    if piece_controller.falling_timer.is_finished() {
        if let Ok(new_coords) = piece.try_move(&board, (coords.x, coords.y), MoveDirection::Down) {
            (coords.x, coords.y) = new_coords;
            transform.update_coords(coords.as_ref(), &config);
        }
    }
}

fn piece_lock(
    mut commands: Commands,
    mut query: Query<(Entity, &Piece, &Coords, &mut PieceController, &Children)>,
    mut board_query: Query<(Entity, &mut Board)>,
    mut query_children: Query<
        (&mut Coords, &mut Transform),
        (With<FallingBlock>, Without<StaticBlock>, Without<Piece>),
    >,
    mut query_static_blocks: Query<Entity, With<StaticBlock>>,
    time: Res<Time>,
    config: Res<LevelConfig>,
    assets: Res<GameAssets>,
    mut next_state: ResMut<NextState<LevelState>>,
    mut ev_placing: MessageWriter<PlacingEvent>,
) {
    let Ok((piece_entity, piece, coords, mut piece_controller, children)) = query.single_mut()
    else {
        return;
    };
    let Ok((board_entity, mut board)) = board_query.single_mut() else {
        return;
    };

    piece_controller.locking_timer.tick(time.delta());

    if piece_controller.locking_timer.is_finished() || piece_controller.hard_dropped
    // after hard drop, place immediately
    {
        info!("piece_place");
        piece_controller.hard_dropped = false; // reset hard drop flag
                                               // switch to piece setup state and finalize piece
        next_state.set(LevelState::Playing); // enter the next playing loop

        // hand over children to board
        commands.entity(board_entity).add_children(children);

        commands.entity(piece_entity).despawn();

        for child in children.iter() {
            // convert the coordinates of the child to board coordinates
            let (mut child_coords, mut child_transform) = query_children.get_mut(child).unwrap();
            child_coords.x += coords.x;
            child_coords.y += coords.y;

            board.set(
                child_coords.x,
                child_coords.y,
                CellKind::Some(piece.piece_type()),
            );

            commands
                .entity(child)
                .insert(StaticBlock {})
                .remove::<FallingBlock>();
            child_transform.update_coords(child_coords.as_ref(), &config);
        }

        // check for line clears
        let lines_cleared = board.clear_lines();

        if lines_cleared > 0 {
            // remove all static blocks
            for entity in query_static_blocks.iter_mut() {
                commands.entity(entity).despawn();
            }

            // remove the current piece's children (which are added to the board but yet to be updated)
            for entity in children.iter() {
                commands.entity(entity).despawn();
            }

            // redraw the board
            for cell in board.cells() {
                let block_entity = spawn_static_block(&mut commands, &config, &assets, cell);
                commands.entity(board_entity).add_child(block_entity);
            }

            piece_controller.locking_timer.reset();
        }

        ev_placing.write(PlacingEvent::Locked(lines_cleared));
    }
}

fn ghost_blocks(
    mut commands: Commands,
    mut piece_query: Query<
        (&Coords, &Piece),
        (With<PieceController>, Or<(Changed<Coords>, Changed<Piece>)>),
    >, // either the piece or its coords changed
    ghost_query: Query<Entity, With<GhostPiece>>,
    mut board_query: Query<&mut Board>,
    config: Res<LevelConfig>,
    texture_assets: Res<GameAssets>,
) {
    if piece_query.is_empty() {
        return;
    }

    for entity in ghost_query.iter() {
        commands.entity(entity).despawn();
    }

    let Ok((coords, piece)) = piece_query.single_mut() else {
        return;
    };
    let Ok(board) = board_query.single_mut() else {
        return;
    };

    let mut ghost_coords = coords.clone();
    let mut ghost_transform = Transform::default();
    let ghost_piece = piece.clone();

    let mut can_move = false;

    while let Ok(new_coords) =
        ghost_piece.try_move(&board, ghost_coords.into(), MoveDirection::Down)
    {
        ghost_coords = Coords::from(new_coords);

        ghost_transform.update_coords(&ghost_coords, &config);
        can_move = true;
    }

    if !can_move {
        return;
    }

    let block_entities = ghost_piece
        .board()
        .cells()
        .iter()
        .map(|cell| spawn_free_block(&mut commands, &config, &texture_assets, cell, GhostBlock))
        .collect::<Vec<_>>();

    let piece_entity = commands
        .spawn(ghost_piece)
        .insert(GhostPiece)
        .insert(ghost_transform)
        .id();
    commands.entity(piece_entity).add_children(&block_entities);
}

fn detect_placement(
    mut piece_query: Query<(&Coords, &Piece), Or<(Changed<Coords>, Changed<Piece>)>>, // either the piece or its coords changed
    mut board_query: Query<&mut Board>,
    mut next_state: ResMut<NextState<PlayingState>>,
    current_state: Res<State<PlayingState>>,
    mut ev_placing: MessageWriter<PlacingEvent>,
) {
    if piece_query.is_empty() {
        return;
    }

    let Ok((coords, piece)) = piece_query.single_mut() else {
        return;
    };
    let Ok(board) = board_query.single_mut() else {
        return;
    };

    let current_state = current_state.get();

    if piece
        .try_move(&board, (coords.x, coords.y), MoveDirection::Down)
        .is_err()
    {
        if current_state == &Falling {
            next_state.set(PlayingState::Locking);
            info!("Transitioning to Placing state.");

            ev_placing.write(PlacingEvent::Placed);
        }
    } else if current_state == &PlayingState::Locking {
        next_state.set(Falling);
        info!("Transitioning to Falling state.");
    }
}
