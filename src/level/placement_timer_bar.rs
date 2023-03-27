use bevy::prelude::*;
use crate::level::{LevelConfig, PieceController};

#[derive(Component)]
pub struct PlacementTimerBar;

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
