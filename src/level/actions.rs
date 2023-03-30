use bevy::prelude::*;
use leafwing_input_manager::prelude::*;

use std::iter::zip;
use crate::assets::GameAssets;
use crate::core::{Board, MoveDirection, Piece, PieceGenerator, PieceRotation};
use crate::level;
use crate::level::{Coords, FallingBlock, GameControl, PieceHolder, LevelConfig, LevelState, MatchCoords, PieceController};

pub struct ActionsPlugin;

impl Plugin for ActionsPlugin {
    fn build(&self, app: &mut App) {
        app
            .add_systems(
                (handle_movements, handle_rotations, handle_hard_drop, swap_hold)
                    .in_set(OnUpdate(LevelState::Falling))
            )

            .add_systems(
                (handle_movements, handle_rotations, handle_hard_drop, swap_hold)
                    .in_set(OnUpdate(LevelState::Placing))
            );
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

pub(crate) fn handle_movements(
    mut action_query: Query<&mut ActionState<GameControl>, With<Board>>,
    mut query: Query<(&mut Piece, &mut Coords, &mut Transform, &mut PieceController)>,
    mut board_query: Query<&mut Board>,
    config: Res<LevelConfig>,
    time: Res<Time>,
) {
    if query.get_single_mut().is_err() {
        return;
    }

    let mut action_state = action_query.single_mut();

    let (mut piece, mut coords, mut transform, mut piece_controller) = query.single_mut();
    let board = board_query.single_mut();

    piece_controller.movement_timer.tick(time.delta());

    let mut movement = None;
    let default_duration = config.movement_duration;
    let mut duration = default_duration;
    let mut just_pressed = false;

    for move_dir in [MoveDirection::Left, MoveDirection::Right, MoveDirection::Down] {
        if action_state.pressed(move_dir.into()) {
            movement = Some(move_dir);
            just_pressed = action_state.just_pressed(move_dir.into());

            if move_dir == MoveDirection::Down {
                duration = config.soft_drop_duration;
                break;
            }

            let press_duration = action_state.current_duration(move_dir.into());
            if press_duration > duration {
                duration = default_duration.mul_f64((config.movement_speedup)
                    .powf(press_duration.as_secs_f64() / default_duration.as_secs_f64()));
            }
            break;
        }
    }

    if piece_controller.movement_timer.finished() || just_pressed {
        if let Some(movement) = movement {
            if move_piece_and_update(&config, &mut piece, &mut coords, &mut transform, &board, movement) {
                piece_controller.placing_timer.reset();
                piece_controller.movement_timer.set_duration(duration);
            }
            if just_pressed {
                piece_controller.movement_timer.reset();
            }
        }
    }
}

pub(crate) fn handle_hard_drop(
    mut action_query: Query<&ActionState<GameControl>, With<Board>>,
    mut query: Query<(&mut Piece, &mut Coords, &mut Transform, &mut PieceController)>,
    mut board_query: Query<&mut Board>,
    config: Res<LevelConfig>,
) {
    if query.get_single_mut().is_err() {
        return;
    }

    let action_state = action_query.single_mut();

    let (mut piece, mut coords, mut transform, mut piece_controller) = query.single_mut();
    let board = board_query.single_mut();

    let hard_drop = action_state.just_pressed(GameControl::HardDrop);

    if hard_drop {
        while move_piece_and_update(&config, &mut piece, &mut coords, &mut transform, &board, MoveDirection::Down) {
            piece_controller.placing_timer.reset();
        }
        piece_controller.hard_dropped = true;
    }
}


pub(crate) fn handle_rotations(
    action_query: Query<&ActionState<GameControl>, With<Board>>,
    mut query: Query<(&mut Piece, &mut Coords, &mut Transform, &Children)>,
    mut children_query: Query<(&mut Coords, &mut Transform), Without<Piece>>,
    mut board_query: Query<&mut Board>,
    mut piece_controller_query: Query<&mut PieceController>,
    config: Res<LevelConfig>,
) {

    if query.get_single_mut().is_err() {
        return;
    }

    let action_state = action_query.single();
    let mut piece_controller = piece_controller_query.single_mut();


    let (mut piece, mut coords, mut transform, children) = query.single_mut();
    let board = board_query.single_mut();

    let rotation = action_state.just_pressed(GameControl::RotateClockwise) as isize - action_state.just_pressed(GameControl::RotateCounterClockwise) as isize;

    let rotation = match rotation {
        1 => Some(PieceRotation::R90 + piece.rotation()),
        -1 => Some(PieceRotation::R270 + piece.rotation()),
        _ => None
    };


    if let Some(rotation_n) = rotation {
        if let Ok((rotation, new_coords)) = piece.try_rotate_with_kicks(&board, (coords.x, coords.y), rotation_n) {
            piece.rotate_to(rotation);

            (coords.x, coords.y) = new_coords;

            for (child_entity, cell) in zip(children.iter(), piece.board().cells().iter()) {
                let (mut child_coords, mut child_transform) = children_query.get_mut(*child_entity).unwrap();
                child_coords.set_if_neq(Coords::from(cell.coords()));
                child_transform.update_coords(child_coords.as_ref(), &config);
            }

            transform.update_coords(coords.as_ref(), &config);

            piece_controller.placing_timer.reset();
        }
    }
}

fn swap_hold(mut commands: Commands,
             mut holder_query: Query<&mut PieceHolder>,
             mut piece_query: Query<(Entity, &mut Piece, &mut PieceController)>,
             config: Res<LevelConfig>,
             texture_assets: Res<GameAssets>,
             mut generator: ResMut<PieceGenerator>,
             action_query: Query<&ActionState<GameControl>, With<Board>>,
) {
    if piece_query.get_single_mut().is_err() {
        return;
    }

    let action_state = action_query.single();
    let (current_piece_entity, current_piece, mut piece_controller) = piece_query.single_mut();
    if !action_state.just_pressed(GameControl::Hold) || piece_controller.used_hold {
        return;
    }

    let mut holder = holder_query.single_mut();


    let next_piece = holder.piece.take().unwrap_or(Piece::from(generator.next().unwrap()));


    // spawn the blocks
    let block_ids: Vec<Entity> = next_piece.board().cells().iter().map(|&cell| {
        level::spawn_free_block(&mut commands, &config, &texture_assets, &next_piece, cell, FallingBlock)
    }).collect();


    let spawn_coords = config.spawn_coords(&next_piece);
    let piece_entity = commands.spawn(next_piece).id();

    // spawn new piece
    commands.entity(piece_entity)
        .insert(Coords::from(spawn_coords))
        .insert(PieceController {
            used_hold: true, // prevent hold from being used again
            ..default()
        })
        .insert(SpatialBundle {
            transform: Transform::from_translation(level::to_translation(spawn_coords.0, spawn_coords.1, config.block_size)),
            ..default()
        });

    commands.entity(piece_entity).push_children(&block_ids);

    holder.piece = Some(current_piece.clone());

    // remove old piece
    commands.entity(current_piece_entity).despawn_recursive();

    piece_controller.used_hold = true;
}
