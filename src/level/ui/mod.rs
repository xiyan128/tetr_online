mod piece_previewer;
mod placement_timer_bar;

use bevy::prelude::*;

use crate::core::PieceGenerator;
use crate::level::common::{LevelState, PieceHolder, PlayingState};
use crate::level::ui::piece_previewer::*;
use crate::level::ui::placement_timer_bar::*;

pub(crate) struct UIPlugin;

impl Plugin for UIPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(
            (spawn_piece_previewer, spawn_hold_viewer).in_schedule(OnEnter(LevelState::Setup)),
        )
        // update piece previewer
        .add_system(update_piece_previewer.run_if(
            any_with_component::<PiecePreviewer>().and_then(any_with_component::<PieceGenerator>()),
        ))
        // update holder viewer
        .add_system(update_hold_viewer.run_if(
            any_with_component::<HoldViewer>().and_then(any_with_component::<PieceHolder>()),
        ))
        .add_system(spawn_placement_timer_bar.in_schedule(OnEnter(PlayingState::Placing)))
        .add_system(update_placement_timer_bar.run_if(any_with_component::<PlacementTimerBar>()))
        .add_system(despawn_placement_timer_bar.in_schedule(OnExit(PlayingState::Placing)));
    }
}
