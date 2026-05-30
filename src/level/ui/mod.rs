//! In-game HUD: piece previews, score readouts, and the lock-down bar.
//!
//! [`UIPlugin`] spawns these on entering the session and updates them from the
//! published engine snapshot each frame. Layout offsets are shared via
//! [`calc_ui_offset`] so the widgets stay aligned around the playfield.

mod piece_previewer;
mod placement_timer_bar;
mod score_views;

use bevy::prelude::*;

use crate::level::common::{LevelConfig, PlayingState};
use crate::level::score::Scorer;
use crate::level::ui::piece_previewer::*;
use crate::level::ui::placement_timer_bar::*;
use crate::level::ui::score_views::*;
use crate::{GameState, PauseState};

pub(crate) struct UIPlugin;

impl Plugin for UIPlugin {
    fn build(&self, app: &mut App) {
        // Inspector/scene registration for the in-game UI markers owned here.
        app.register_type::<PiecePreviewer>()
            .register_type::<HoldViewer>()
            .register_type::<PreviewHolder>()
            .register_type::<LockingTimerBar>()
            .register_type::<ScoreText>()
            .register_type::<LineCountText>()
            .register_type::<ScoreTypeText>();
        // Spawned once per session (OnEnter Playing). Pause is a sub-state of
        // Playing, so these survive a pause/resume round-trip.
        app.add_systems(
            OnEnter(GameState::Playing),
            (
                spawn_piece_previewer,
                spawn_hold_viewer,
                spawn_score_text,
                spawn_line_count_text,
                spawn_score_type_text,
            ),
        )
        // Previews + hold read the engine snapshot directly.
        .add_systems(
            Update,
            update_piece_previewer
                .run_if(in_state(PauseState::Running).and(any_with_component::<PiecePreviewer>)),
        )
        .add_systems(
            Update,
            update_hold_viewer
                .run_if(in_state(PauseState::Running).and(any_with_component::<HoldViewer>)),
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
            update_locking_timer_bar
                .run_if(in_state(PauseState::Running).and(any_with_component::<LockingTimerBar>)),
        )
        .add_systems(OnExit(PlayingState::Locking), despawn_locking_timer_bar);
    }
}

pub fn calc_ui_offset(config: &LevelConfig) -> f32 {
    config.block_size * 0.5
}
