use crate::level::common::{BoardState, LevelConfig, PieceController};
use bevy::prelude::*;
use bevy::sprite::Anchor;

#[derive(Component)]
pub struct LockingTimerBar;

pub fn spawn_locking_timer_bar(
    mut commands: Commands,
    config: Res<LevelConfig>,
    board_entity_query: Query<Entity, With<BoardState>>,
) {
    let Ok(board_entity) = board_entity_query.single() else {
        return;
    };

    let bar_height = config.block_size * 0.2;

    // spawn the timer bar
    let timer_bar_entity = commands
        .spawn((
            Sprite {
                custom_size: Some(Vec2::new(
                    config.block_size * config.board_width as f32,
                    bar_height,
                )),
                color: Color::srgb(0.5, 0.5, 0.5),
                ..Default::default()
            },
            Anchor::BOTTOM_LEFT,
            Transform::from_translation(Vec3::new(0., -bar_height, 1.)),
        ))
        .insert(LockingTimerBar)
        .id();

    commands.entity(board_entity).add_child(timer_bar_entity);
}

pub fn update_locking_timer_bar(
    piece_controller_query: Query<&PieceController>,
    mut bar_query: Query<&mut Sprite, With<LockingTimerBar>>,
    config: Res<LevelConfig>,
) {
    let Ok(piece_controller) = piece_controller_query.single() else {
        return;
    };

    let timer = &piece_controller.locking_timer;
    let Ok(mut bar) = bar_query.single_mut() else {
        return;
    };

    let progress = timer.fraction();
    let width = config.block_size * config.board_width as f32 * progress;

    bar.custom_size = Some(Vec2::new(width, bar.custom_size.unwrap().y));
}

pub fn despawn_locking_timer_bar(
    bar_query: Query<Entity, With<LockingTimerBar>>,
    mut commands: Commands,
) {
    let Ok(bar) = bar_query.single() else {
        return;
    };
    commands.entity(bar).despawn();
}
