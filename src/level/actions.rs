use bevy::ecs::system::SystemParam;
use bevy::prelude::*;

use crate::assets::GameAssets;
use crate::engine::{Board, LockDownMode, MoveDirection, Piece, PieceRotation, PieceType};
use crate::level::common::{
    spawn_coords_after_generation_rules, spawn_falling_piece, ActionEvent, AudioCue, BoardState,
    Coords, DasState, LevelConfig, LevelSystems, MatchCoords, PieceController, PieceGeneratorState,
    PieceHolder, PieceState,
};
use crate::GameState;
use itertools::iproduct;
use std::iter::zip;
use std::time::Duration;

pub struct ActionsPlugin;

impl Plugin for ActionsPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<DasState>()
            .add_systems(OnEnter(GameState::InGame), reset_das_state)
            .add_systems(
                Update,
                (
                    handle_movements,
                    handle_rotations,
                    handle_hard_drop,
                    swap_hold,
                )
                    .in_set(LevelSystems::PlayerInput)
                    .run_if(in_state(GameState::InGame)),
            );
    }
}

fn reset_das_state(mut das_state: ResMut<DasState>) {
    *das_state = DasState::default();
}

#[derive(SystemParam)]
struct RotationQueries<'w, 's> {
    piece: Query<
        'w,
        's,
        (
            &'static mut PieceState,
            &'static mut Coords,
            &'static mut Transform,
            &'static Children,
        ),
    >,
    children: Query<'w, 's, (&'static mut Coords, &'static mut Transform), Without<PieceState>>,
    board: Query<'w, 's, &'static BoardState>,
    controller: Query<'w, 's, &'static mut PieceController>,
}

#[derive(SystemParam)]
struct HoldQueries<'w, 's> {
    holder: Query<'w, 's, &'static mut PieceHolder>,
    piece: Query<'w, 's, (Entity, &'static PieceState, &'static PieceController)>,
    generator: Query<'w, 's, &'static mut PieceGeneratorState>,
    board: Query<'w, 's, &'static BoardState>,
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

fn reset_locking_timer_after_successful_move_or_rotation(
    piece_controller: &mut PieceController,
    config: &LevelConfig,
) {
    if config.lock_down_mode != LockDownMode::Classic {
        piece_controller.locking_timer.reset();
    }
}

fn horizontal_input(
    keyboard: &ButtonInput<KeyCode>,
    das_state: &DasState,
) -> (Option<MoveDirection>, bool) {
    let left_pressed = keyboard.pressed(KeyCode::ArrowLeft);
    let right_pressed = keyboard.pressed(KeyCode::ArrowRight);
    let left_just_pressed = keyboard.just_pressed(KeyCode::ArrowLeft);
    let right_just_pressed = keyboard.just_pressed(KeyCode::ArrowRight);

    match (left_pressed, right_pressed) {
        (true, false) => (Some(MoveDirection::Left), left_just_pressed),
        (false, true) => (Some(MoveDirection::Right), right_just_pressed),
        (true, true) if left_just_pressed => (Some(MoveDirection::Left), true),
        (true, true) if right_just_pressed => (Some(MoveDirection::Right), true),
        (true, true) => (das_state.active_direction(), false),
        (false, false) => (None, false),
    }
}

fn soft_drop_action(
    piece_controller: &mut PieceController,
    keyboard: &ButtonInput<KeyCode>,
    delta: Duration,
) -> bool {
    if !keyboard.pressed(KeyCode::ArrowDown) {
        piece_controller.soft_drop_timer.reset();
        return false;
    }

    piece_controller.soft_drop_timer.tick(delta);
    keyboard.just_pressed(KeyCode::ArrowDown) || piece_controller.soft_drop_timer.is_finished()
}

fn reset_piece_to_north(piece: &Piece) -> Piece {
    Piece::from(piece.piece_type())
}

fn handle_movements(
    mut query: Query<(
        &PieceState,
        &mut Coords,
        &mut Transform,
        &mut PieceController,
    )>,
    board_query: Query<&BoardState>,
    keyboard: Res<ButtonInput<KeyCode>>,
    config: Res<LevelConfig>,
    time: Res<Time>,
    mut das_state: ResMut<DasState>,
    mut ev_action: MessageWriter<ActionEvent>,
) {
    let (horizontal_movement, horizontal_just_pressed) = horizontal_input(&keyboard, &das_state);
    let movement = das_state.next_action(
        horizontal_movement,
        horizontal_just_pressed,
        time.delta(),
        &config,
    );

    let Ok((piece, mut coords, mut transform, mut piece_controller)) = query.single_mut() else {
        return;
    };
    let Ok(board) = board_query.single() else {
        return;
    };

    let soft_drop = soft_drop_action(&mut piece_controller, &keyboard, time.delta());

    if let Some(movement) = movement {
        if move_piece_and_update(&config, piece, &mut coords, &mut transform, board, movement) {
            ev_action.write(ActionEvent::Movement(movement));
            reset_locking_timer_after_successful_move_or_rotation(&mut piece_controller, &config);
            return;
        }
    }

    if soft_drop
        && move_piece_and_update(
            &config,
            piece,
            &mut coords,
            &mut transform,
            board,
            MoveDirection::Down,
        )
    {
        ev_action.write(ActionEvent::Movement(MoveDirection::Down));
        reset_locking_timer_after_successful_move_or_rotation(&mut piece_controller, &config);
    }
}

fn handle_hard_drop(
    mut commands: Commands,
    mut query: Query<(
        &PieceState,
        &mut Coords,
        &mut Transform,
        &mut PieceController,
    )>,
    board_query: Query<&BoardState>,
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
            reset_locking_timer_after_successful_move_or_rotation(&mut piece_controller, &config);

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
                wall_kick_set,      // successful SRS kick test number
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
    mut next_game_state: ResMut<NextState<GameState>>,
    mut ev_action: MessageWriter<ActionEvent>,
) {
    let Ok((current_piece_entity, current_piece, piece_controller)) = queries.piece.single() else {
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
    let Ok(board) = queries.board.single() else {
        return;
    };

    let next_piece = holder
        .piece
        .take()
        .map(|piece| reset_piece_to_north(&piece))
        .unwrap_or(Piece::from(generator.next().unwrap()));

    let Some(spawn_coords) = spawn_coords_after_generation_rules(&config, board, &next_piece)
    else {
        next_game_state.set(GameState::GameOver);
        return;
    };

    spawn_falling_piece(
        &mut commands,
        &config,
        &texture_assets,
        next_piece,
        spawn_coords,
        true,
    );

    holder.piece = Some(reset_piece_to_north(current_piece));

    // remove old piece
    commands.entity(current_piece_entity).despawn();

    ev_action.write(ActionEvent::Hold);
    commands.trigger(AudioCue::Hold);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reset_piece_to_north_preserves_type_and_clears_rotation() {
        let mut piece = Piece::from(PieceType::T);
        piece.rotate_to(PieceRotation::R90);

        let reset = reset_piece_to_north(&piece);

        assert_eq!(reset.piece_type(), PieceType::T);
        assert_eq!(reset.rotation(), PieceRotation::R0);
    }
}
