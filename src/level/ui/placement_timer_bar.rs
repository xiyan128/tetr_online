use crate::level::common::{LevelConfig, PieceController};
use crate::level::Board;
use bevy::prelude::*;
use bevy::sprite::Anchor;

#[derive(Component)]
pub struct LockingTimerBar;

pub fn spawn_locking_timer_bar(
    mut commands: Commands,
    config: Res<LevelConfig>,
    board_entity_query: Query<Entity, With<Board>>,
) {
    let board_entity = board_entity_query.single();

    let bar_height = config.block_size * 0.2;

    // spawn the timer bar
    let timer_bar_entity = commands
        .spawn(SpriteBundle {
            transform: Transform::from_translation(Vec3::new(0., -bar_height, 1.)),
            sprite: Sprite {
                custom_size: Some(Vec2::new(
                    config.block_size * config.board_width as f32,
                    bar_height,
                )),
                color: Color::GRAY,
                anchor: Anchor::BottomLeft,
                ..Default::default()
            },
            ..Default::default()
        })
        .insert(LockingTimerBar)
        .id();

    commands.entity(board_entity).add_child(timer_bar_entity);
}

// update_locking_timer_bar
pub fn update_locking_timer_bar(
    mut piece_controller_query: Query<&mut PieceController>,
    mut bar_query: Query<&mut Sprite, With<LockingTimerBar>>,
    config: Res<LevelConfig>,
) {
    let mut piece_controller = piece_controller_query.single_mut();

    let timer = &mut piece_controller.locking_timer;
    let mut bar = bar_query.single_mut();

    let progress = timer.percent();
    let width = config.block_size * config.board_width as f32 * progress;

    bar.custom_size = Some(Vec2::new(width, bar.custom_size.unwrap().y));
}

// remove_locking_timer_bar
pub fn despawn_locking_timer_bar(
    mut bar_query: Query<Entity, With<LockingTimerBar>>,
    mut commands: Commands,
) {
    info!("despawning locking timer bar");
    let bar = bar_query.single();
    commands.entity(bar).despawn_recursive();
}
