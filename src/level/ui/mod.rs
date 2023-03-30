mod placement_timer_bar;
mod piece_previewer;

use bevy::prelude::*;
use crate::GameState;
use crate::level::LevelState;
use crate::level::ui::piece_previewer::{spawn_hold_viewer, spawn_piece_previewer, update_hold_viewer, update_piece_previewer};
use crate::level::ui::placement_timer_bar::spawn_placement_timer_bar;

pub(crate) struct UIPlugin;

impl Plugin for UIPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems((update_piece_previewer, update_hold_viewer).in_set(OnUpdate(GameState::InGame)))
            .add_systems((spawn_placement_timer_bar, spawn_piece_previewer, spawn_hold_viewer).in_schedule(OnExit(LevelState::Ready)))
            .add_system(placement_timer_bar::show_placement_timer_bar.in_schedule(OnEnter(LevelState::Placing)))
            .add_system(placement_timer_bar::update_placement_timer_bar.in_set(OnUpdate(LevelState::Placing)))
            .add_system(placement_timer_bar::hide_placement_timer_bar.in_schedule(OnExit(LevelState::Placing)));
    }
}
