mod piece_previewer;
mod placement_timer_bar;
mod score_views;

use bevy::prelude::*;

use crate::core::PieceGenerator;
use crate::level::common::{LevelConfig, LevelState, PieceHolder, PlayingState};
use crate::level::score::Scorer;
use crate::level::ui::piece_previewer::*;
use crate::level::ui::placement_timer_bar::*;
use crate::level::ui::score_views::*;

pub(crate) struct UIPlugin;

impl Plugin for UIPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(
            OnEnter(LevelState::Setup),
            (
                spawn_piece_previewer,
                spawn_hold_viewer,
                spawn_score_text,
                spawn_line_count_text,
                spawn_score_type_text,
            ),
        )
        // update piece previewer
        .add_systems(
            Update,
            update_piece_previewer.run_if(
                any_with_component::<PiecePreviewer>.and(any_with_component::<PieceGenerator>),
            ),
        )
        // update holder viewer
        .add_systems(
            Update,
            update_hold_viewer
                .run_if(any_with_component::<HoldViewer>.and(any_with_component::<PieceHolder>)),
        )
        .add_systems(
            Update,
            update_score_text
                .run_if(any_with_component::<ScoreText>.and(resource_exists_and_changed::<Scorer>)),
        )
        .add_systems(
            Update,
            update_line_count_text.run_if(
                any_with_component::<LineCountText>.and(resource_exists_and_changed::<Scorer>),
            ),
        )
        .add_systems(
            Update,
            update_score_type_text.run_if(any_with_component::<ScoreTypeText>),
        )
        .add_systems(
            Update,
            fade_out_score_type_text.run_if(any_with_component::<ScoreTypeText>),
        )
        .add_systems(OnEnter(PlayingState::Locking), spawn_locking_timer_bar)
        .add_systems(
            Update,
            update_locking_timer_bar.run_if(any_with_component::<LockingTimerBar>),
        )
        .add_systems(OnExit(PlayingState::Locking), despawn_locking_timer_bar);
    }
}

pub fn calc_ui_offset(config: &LevelConfig) -> f32 {
    config.block_size * 0.5
}
