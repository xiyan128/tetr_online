use bevy::ecs::system::SystemParam;
use bevy::prelude::*;

use crate::core::*;
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
mod sound_effects;
mod ui;

pub struct LevelPlugin;

type FallingBlockQuery<'w, 's> = Query<
    'w,
    's,
    (&'static mut Coords, &'static mut Transform),
    (With<FallingBlock>, Without<StaticBlock>, Without<Piece>),
>;

type ActivePieceQuery<'w, 's> = Query<
    'w,
    's,
    (&'static Coords, &'static Piece),
    (
        With<PieceController>,
        Without<GhostPiece>,
        Without<GhostBlock>,
        Or<(Changed<Coords>, Changed<Piece>)>,
    ),
>;

type GhostBlockQuery<'w, 's> = Query<
    'w,
    's,
    (&'static mut Coords, &'static mut Transform),
    (
        With<GhostBlock>,
        Without<GhostPiece>,
        Without<PieceController>,
    ),
>;

impl Plugin for LevelPlugin {
    fn build(&self, app: &mut App) {
        app
            // states
            .add_sub_state::<PlayingState>()
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
            // setup
            .add_systems(OnEnter(GameState::InGame), level_setup)
            // updates
            .add_systems(
                Update,
                (piece_setup, piece_fall, detect_placement, ghost_blocks)
                    .run_if(in_state(GameState::InGame)),
            )
            .add_systems(Update, piece_lock.run_if(in_state(PlayingState::Locking)));
    }
}

#[derive(SystemParam)]
struct LockQueries<'w, 's> {
    piece: Query<
        'w,
        's,
        (
            Entity,
            &'static Piece,
            &'static Coords,
            &'static mut PieceController,
            &'static Children,
        ),
    >,
    board: Query<'w, 's, (Entity, &'static mut Board)>,
    falling_blocks: FallingBlockQuery<'w, 's>,
    static_blocks: Query<'w, 's, Entity, With<StaticBlock>>,
}

#[derive(SystemParam)]
struct GhostQueries<'w, 's> {
    active_piece: ActivePieceQuery<'w, 's>,
    ghosts: Query<
        'w,
        's,
        (
            Entity,
            &'static mut Piece,
            &'static mut Transform,
            Option<&'static Children>,
        ),
        With<GhostPiece>,
    >,
    ghost_blocks: GhostBlockQuery<'w, 's>,
    board: Query<'w, 's, &'static Board>,
}

type ChangedPieceQuery<'w, 's> =
    Query<'w, 's, (&'static Coords, &'static Piece), Or<(Changed<Coords>, Changed<Piece>)>>;

// setup board and camera
fn level_setup(mut commands: Commands, config: Res<LevelConfig>, texture_assets: Res<GameAssets>) {
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
            BlockKind::Background,
        );
        block_ids.push(block_id);
    }

    let board_entity = commands
        .spawn((board, Transform::default()))
        .insert(DespawnOnExit(GameState::InGame))
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
        DespawnOnExit(GameState::InGame),
    ));
}

fn spawn_static_block(
    commands: &mut Commands,
    config: &Res<LevelConfig>,
    assets: &Res<GameAssets>,
    cell: &Cell,
) -> Entity {
    spawn_free_block(commands, config, assets, cell, BlockKind::Static)
}

fn piece_setup(
    mut commands: Commands,
    config: Res<LevelConfig>,
    texture_assets: Res<GameAssets>,
    piece_query: Query<(Entity, &Piece, &PieceController)>,
    board_query: Query<&Board>,
    mut next_game_state: ResMut<NextState<GameState>>,
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
    if piece.collide_with(board, spawn_coords) {
        info!("game over");
        next_game_state.set(GameState::GameOver);
        return;
    }

    let block_ids: Vec<Entity> = spawn_piece_blocks(
        &mut commands,
        &config,
        &texture_assets,
        &piece,
        BlockKind::Falling,
    );

    let piece_entity = commands
        .spawn((piece, DespawnOnExit(GameState::InGame)))
        .id();

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
    board_query: Query<&Board>,
    config: Res<LevelConfig>,
    time: Res<Time>,
) {
    let Ok((piece, mut piece_controller, mut coords, mut transform)) = query.single_mut() else {
        return;
    };
    let Ok(board) = board_query.single() else {
        return;
    };

    piece_controller.falling_timer.tick(time.delta());

    if piece_controller.falling_timer.is_finished() {
        if let Some(new_coords) = piece.try_move(board, (coords.x, coords.y), MoveDirection::Down) {
            (coords.x, coords.y) = new_coords;
            transform.update_coords(coords.as_ref(), &config);
        }
    }
}

fn piece_lock(
    mut commands: Commands,
    mut queries: LockQueries,
    time: Res<Time>,
    config: Res<LevelConfig>,
    assets: Res<GameAssets>,
    mut next_playing_state: ResMut<NextState<PlayingState>>,
    mut ev_placing: MessageWriter<PlacingEvent>,
) {
    let Ok((piece_entity, piece, coords, mut piece_controller, children)) =
        queries.piece.single_mut()
    else {
        return;
    };
    let Ok((board_entity, mut board)) = queries.board.single_mut() else {
        return;
    };

    piece_controller.locking_timer.tick(time.delta());

    if piece_controller.locking_timer.is_finished() || piece_controller.hard_dropped
    // after hard drop, place immediately
    {
        info!("piece_place");
        piece_controller.hard_dropped = false; // reset hard drop flag
        next_playing_state.set(PlayingState::Falling);

        // hand over children to board
        commands.entity(board_entity).add_children(children);

        commands.entity(piece_entity).despawn();

        for child in children.iter() {
            // convert the coordinates of the child to board coordinates
            let (mut child_coords, mut child_transform) =
                queries.falling_blocks.get_mut(child).unwrap();
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
            for entity in queries.static_blocks.iter_mut() {
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
        commands.trigger(AudioCue::Locked(lines_cleared));
    }
}

fn ghost_blocks(
    mut commands: Commands,
    mut queries: GhostQueries,
    config: Res<LevelConfig>,
    texture_assets: Res<GameAssets>,
) {
    if queries.active_piece.is_empty() {
        return;
    }

    let Ok((coords, piece)) = queries.active_piece.single_mut() else {
        return;
    };
    let Ok(board) = queries.board.single() else {
        return;
    };

    let mut ghost_coords = coords.clone();
    let mut ghost_transform = Transform::default();
    let ghost_piece = piece.clone();

    let mut can_move = false;

    while let Some(new_coords) =
        ghost_piece.try_move(board, ghost_coords.into(), MoveDirection::Down)
    {
        ghost_coords = Coords::from(new_coords);

        ghost_transform.update_coords(&ghost_coords, &config);
        can_move = true;
    }

    if !can_move {
        for (entity, _, _, _) in queries.ghosts.iter_mut() {
            commands.entity(entity).despawn();
        }
        return;
    }

    let ghost_board = ghost_piece.board();
    let ghost_cells = ghost_board.cells();

    let Ok((piece_entity, mut existing_piece, mut transform, children)) =
        queries.ghosts.single_mut()
    else {
        let block_entities = ghost_cells
            .iter()
            .map(|cell| {
                spawn_free_block(
                    &mut commands,
                    &config,
                    &texture_assets,
                    cell,
                    BlockKind::Ghost,
                )
            })
            .collect::<Vec<_>>();

        let piece_entity = commands
            .spawn((
                ghost_piece,
                ghost_transform,
                DespawnOnExit(GameState::InGame),
            ))
            .insert(GhostPiece)
            .id();
        commands.entity(piece_entity).add_children(&block_entities);
        return;
    };

    *existing_piece = ghost_piece;
    *transform = ghost_transform;

    let Some(children) = children else {
        return;
    };

    if children.len() != ghost_cells.len() {
        commands.entity(piece_entity).despawn_related::<Children>();
        let block_entities = ghost_cells
            .iter()
            .map(|cell| {
                spawn_free_block(
                    &mut commands,
                    &config,
                    &texture_assets,
                    cell,
                    BlockKind::Ghost,
                )
            })
            .collect::<Vec<_>>();
        commands.entity(piece_entity).add_children(&block_entities);
        return;
    }

    for (child_entity, cell) in children.iter().zip(ghost_cells.iter()) {
        let Ok((mut child_coords, mut child_transform)) =
            queries.ghost_blocks.get_mut(child_entity)
        else {
            continue;
        };
        child_coords.set_if_neq(Coords::from(cell.coords()));
        child_transform.update_coords(child_coords.as_ref(), &config);
    }
}

fn detect_placement(
    mut commands: Commands,
    mut piece_query: ChangedPieceQuery, // either the piece or its coords changed
    board_query: Query<&Board>,
    mut next_state: ResMut<NextState<PlayingState>>,
    current_state: Res<State<PlayingState>>,
) {
    if piece_query.is_empty() {
        return;
    }

    let Ok((coords, piece)) = piece_query.single_mut() else {
        return;
    };
    let Ok(board) = board_query.single() else {
        return;
    };

    let current_state = current_state.get();

    if piece
        .try_move(board, (coords.x, coords.y), MoveDirection::Down)
        .is_none()
    {
        if current_state == &Falling {
            next_state.set(PlayingState::Locking);
            info!("Transitioning to Placing state.");
            commands.trigger(AudioCue::Placed);
        }
    } else if current_state == &PlayingState::Locking {
        next_state.set(Falling);
        info!("Transitioning to Falling state.");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn level_plugin_systems_initialize_without_query_conflicts() {
        let mut app = App::new();
        app.add_plugins((MinimalPlugins, bevy::state::app::StatesPlugin))
            .init_state::<GameState>()
            .insert_resource(ButtonInput::<KeyCode>::default())
            .insert_resource(GameAssets {
                block_texture: default(),
                hard_drop_sound: default(),
                placed_sound: default(),
                line_clear_1: default(),
                line_clear_2: default(),
                line_clear_3: default(),
                line_clear_4: default(),
                locked_sound: default(),
                hold_sound: default(),
                rotation_sound: default(),
                font: default(),
            })
            .add_plugins(LevelPlugin);

        app.world_mut()
            .resource_mut::<NextState<GameState>>()
            .set(GameState::InGame);
        app.update();
    }
}
