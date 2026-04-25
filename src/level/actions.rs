use bevy::ecs::system::SystemParam;
use bevy::prelude::*;

use crate::assets::GameAssets;
use crate::core::{Board, MoveDirection, Piece, PieceGenerator, PieceRotation, PieceType};
use crate::level::common;
use crate::level::common::{
    spawn_piece_blocks, ActionEvent, AudioCue, BlockKind, Coords, LevelConfig, MatchCoords,
    PieceController, PieceHolder,
};
use crate::GameState;
use itertools::iproduct;
use std::iter::zip;

pub struct ActionsPlugin;

impl Plugin for ActionsPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(
            Update,
            (
                handle_movements,
                handle_rotations,
                handle_hard_drop,
                swap_hold,
            )
                .run_if(in_state(GameState::InGame)),
        );
    }
}

#[derive(SystemParam)]
struct RotationQueries<'w, 's> {
    piece: Query<
        'w,
        's,
        (
            &'static mut Piece,
            &'static mut Coords,
            &'static mut Transform,
            &'static Children,
        ),
    >,
    children: Query<'w, 's, (&'static mut Coords, &'static mut Transform), Without<Piece>>,
    board: Query<'w, 's, &'static Board>,
    controller: Query<'w, 's, &'static mut PieceController>,
}

#[derive(SystemParam)]
struct HoldQueries<'w, 's> {
    holder: Query<'w, 's, &'static mut PieceHolder>,
    piece: Query<'w, 's, (Entity, &'static Piece, &'static mut PieceController)>,
    generator: Query<'w, 's, &'static mut PieceGenerator>,
}

fn move_piece_and_update(
    config: &LevelConfig,
    piece: &Piece,
    coords: &mut Coords,
    transform: &mut Transform,
    board: &Board,
    movement: MoveDirection,
) -> bool {
    if let Some(new_coords) = piece.try_move(board, (coords.x, coords.y), movement) {
        (coords.x, coords.y) = new_coords;
        transform.update_coords(coords, config);
        true
    } else {
        false
    }
}

fn handle_movements(
    mut query: Query<(&Piece, &mut Coords, &mut Transform, &mut PieceController)>,
    board_query: Query<&Board>,
    keyboard: Res<ButtonInput<KeyCode>>,
    config: Res<LevelConfig>,
    time: Res<Time>,
    mut ev_action: MessageWriter<ActionEvent>,
) {
    let Ok((piece, mut coords, mut transform, mut piece_controller)) = query.single_mut() else {
        return;
    };
    let Ok(board) = board_query.single() else {
        return;
    };

    piece_controller.movement_timer.tick(time.delta());

    let mut movement = None;
    let default_duration = config.movement_duration;
    let mut duration = default_duration;
    let mut just_pressed = false;

    for move_dir in [
        MoveDirection::Left,
        MoveDirection::Right,
        MoveDirection::Down,
    ] {
        let key = key_for_movement(move_dir);
        if keyboard.pressed(key) {
            movement = Some(move_dir);
            just_pressed = keyboard.just_pressed(key);

            if move_dir == MoveDirection::Down {
                duration = config.soft_drop_duration;
                break;
            }

            if piece_controller.active_movement != Some(move_dir) {
                piece_controller.active_movement = Some(move_dir);
                piece_controller.movement_hold_duration = Default::default();
            }
            piece_controller.movement_hold_duration += time.delta();
            let press_duration = piece_controller.movement_hold_duration;
            if press_duration > duration {
                duration = default_duration.mul_f64(
                    (config.movement_speedup)
                        .powf(press_duration.as_secs_f64() / default_duration.as_secs_f64()),
                );
            }
            break;
        }
    }

    if movement.is_none() {
        piece_controller.active_movement = None;
        piece_controller.movement_hold_duration = Default::default();
    }

    if piece_controller.movement_timer.is_finished() || just_pressed {
        if let Some(movement) = movement {
            if move_piece_and_update(&config, piece, &mut coords, &mut transform, board, movement) {
                ev_action.write(ActionEvent::Movement(movement));
                piece_controller.locking_timer.reset();
                piece_controller.movement_timer.set_duration(duration);
            }
            if just_pressed {
                piece_controller.movement_timer.reset();
            }
        }
    }
}

fn handle_hard_drop(
    mut commands: Commands,
    mut query: Query<(&Piece, &mut Coords, &mut Transform, &mut PieceController)>,
    board_query: Query<&Board>,
    keyboard: Res<ButtonInput<KeyCode>>,
    config: Res<LevelConfig>,
    mut ev_action: MessageWriter<ActionEvent>,
) {
    let Ok((piece, mut coords, mut transform, mut piece_controller)) = query.single_mut() else {
        return;
    };
    let Ok(board) = board_query.single() else {
        return;
    };

    let hard_drop = keyboard.just_pressed(KeyCode::Space);

    if hard_drop {
        let mut lines = 0;
        while move_piece_and_update(
            &config,
            piece,
            &mut coords,
            &mut transform,
            board,
            MoveDirection::Down,
        ) {
            piece_controller.locking_timer.reset();
            lines += 1;
        }
        piece_controller.hard_dropped = true;

        ev_action.write(ActionEvent::HardDrop(lines));
        commands.trigger(AudioCue::HardDrop);
    }
}

fn handle_rotations(
    mut commands: Commands,
    mut queries: RotationQueries,
    keyboard: Res<ButtonInput<KeyCode>>,
    config: Res<LevelConfig>,
    mut ev_action: MessageWriter<ActionEvent>,
) {
    let Ok((mut piece, mut coords, mut transform, children)) = queries.piece.single_mut() else {
        return;
    };
    let Ok(mut piece_controller) = queries.controller.single_mut() else {
        return;
    };
    let Ok(board) = queries.board.single() else {
        return;
    };

    let rotation = (keyboard.just_pressed(KeyCode::ArrowUp) || keyboard.just_pressed(KeyCode::KeyX))
        as isize
        - keyboard.just_pressed(KeyCode::KeyZ) as isize;

    let rotation = match rotation {
        1 => Some(PieceRotation::R90 + piece.rotation()),
        -1 => Some(PieceRotation::R270 + piece.rotation()),
        _ => None,
    };

    if let Some(rotation_n) = rotation {
        if let Some((rotation, new_coords, wall_kick_set)) =
            piece.try_rotate_with_kicks(board, (coords.x, coords.y), rotation_n)
        {
            piece.rotate_to(rotation);

            (coords.x, coords.y) = new_coords;

            for (child_entity, cell) in zip(children.iter(), piece.board().cells().iter()) {
                let (mut child_coords, mut child_transform) =
                    queries.children.get_mut(child_entity).unwrap();
                child_coords.set_if_neq(Coords::from(cell.coords()));
                child_transform.update_coords(child_coords.as_ref(), &config);
            }

            transform.update_coords(coords.as_ref(), &config);
            piece_controller.locking_timer.reset();

            // count t-spin corners
            let mut t_spin_corners = 0;
            if piece.piece_type() == PieceType::T {
                for (x, y) in iproduct!([0, 2], [0, 2]) {
                    if board.get(coords.x + x, coords.y + y).is_some() {
                        t_spin_corners += 1;
                    }
                }
            }

            ev_action.write(ActionEvent::Rotation(
                piece.piece_type(), // type of piece that was rotated
                t_spin_corners,     // (only for t-spin) number of corners occupied by other blocks
                wall_kick_set != 0, // if wall kick was performed
            ));
            commands.trigger(AudioCue::Rotation);
        }
    }
}

fn swap_hold(
    mut commands: Commands,
    mut queries: HoldQueries,
    config: Res<LevelConfig>,
    texture_assets: Res<GameAssets>,
    keyboard: Res<ButtonInput<KeyCode>>,
    mut ev_action: MessageWriter<ActionEvent>,
) {
    let Ok((current_piece_entity, current_piece, mut piece_controller)) =
        queries.piece.single_mut()
    else {
        return;
    };
    if !keyboard.just_pressed(KeyCode::ShiftLeft) || piece_controller.used_hold {
        return;
    }

    let Ok(mut holder) = queries.holder.single_mut() else {
        return;
    };
    let Ok(mut generator) = queries.generator.single_mut() else {
        return;
    };

    let next_piece = holder
        .piece
        .take()
        .unwrap_or(Piece::from(generator.next().unwrap()));

    let block_ids = spawn_piece_blocks(
        &mut commands,
        &config,
        &texture_assets,
        &next_piece,
        BlockKind::Falling,
    );

    let spawn_coords = config.spawn_coords(&next_piece);
    let piece_entity = commands
        .spawn((next_piece, DespawnOnExit(GameState::InGame)))
        .id();

    // spawn new piece
    commands
        .entity(piece_entity)
        .insert(Coords::from(spawn_coords))
        .insert(PieceController {
            used_hold: true, // prevent hold from being used again
            ..default()
        })
        .insert(Transform::from_translation(common::to_translation(
            spawn_coords.0,
            spawn_coords.1,
            config.block_size,
        )));

    commands.entity(piece_entity).add_children(&block_ids);

    holder.piece = Some(current_piece.clone());

    // remove old piece
    commands.entity(current_piece_entity).despawn();

    piece_controller.used_hold = true;
    ev_action.write(ActionEvent::Hold);
    commands.trigger(AudioCue::Hold);
}

fn key_for_movement(direction: MoveDirection) -> KeyCode {
    match direction {
        MoveDirection::Down => KeyCode::ArrowDown,
        MoveDirection::Left => KeyCode::ArrowLeft,
        MoveDirection::Right => KeyCode::ArrowRight,
    }
}
