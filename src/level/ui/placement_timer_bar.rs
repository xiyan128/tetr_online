use bevy::prelude::*;
use bevy::sprite::Anchor;
use crate::level::{LevelConfig, PieceController, Board};

#[derive(Component)]
pub struct PlacementTimerBar;

pub fn spawn_placement_timer_bar(
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
                custom_size: Some(Vec2::new(config.block_size * config.board_width as f32, bar_height)),
                color: Color::GRAY,
                anchor: Anchor::BottomLeft,
                ..Default::default()
            },
            visibility: Visibility::Hidden,
            ..Default::default()
        })
        .insert(PlacementTimerBar).id();

    commands.entity(board_entity).add_child(timer_bar_entity);
}

pub fn show_placement_timer_bar(
    mut bar_query: Query<&mut Visibility, With<PlacementTimerBar>>,
) {
    let mut bar = bar_query.single_mut();
    bar.set_if_neq(Visibility::Inherited);
}

// update_placement_timer_bar
pub fn update_placement_timer_bar(
    mut piece_controller_query: Query<&mut PieceController>,
    mut bar_query: Query<&mut Sprite, With<PlacementTimerBar>>,
    config: Res<LevelConfig>,
) {
    let mut piece_controller = piece_controller_query.single_mut();

    let timer = &mut piece_controller.placing_timer;
    let mut bar = bar_query.single_mut();

    let progress = timer.percent();
    let width = config.block_size * config.board_width as f32 * progress;

    bar.custom_size = Some(Vec2::new(width, bar.custom_size.unwrap().y));
}

// remove_placement_timer_bar
pub fn hide_placement_timer_bar(
    mut bar_query: Query<&mut Visibility, With<PlacementTimerBar>>,
) {
    let mut bar = bar_query.single_mut();
    bar.set_if_neq(Visibility::Hidden);
}
