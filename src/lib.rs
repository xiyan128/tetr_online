use bevy::app::App;
#[cfg(debug_assertions)]
use bevy::diagnostic::{FrameTimeDiagnosticsPlugin, LogDiagnosticsPlugin};
use bevy::prelude::*;
use bevy_asset_loader::prelude::*;

mod assets;
mod engine;
mod level;

pub use crate::engine::{
    apply_grounded_move_or_rotation, breaks_back_to_back, classify_t_spin, fall_duration,
    fall_speed_seconds, fixed_goal_for_level, goal_for_level, is_block_out, is_lock_out,
    is_top_out, qualifies_for_back_to_back, soft_drop_duration, soft_drop_speed_seconds,
    t_spin_corners, variable_goal_for_level, variable_goal_units, ActivePiece, ActivePieceSnapshot,
    Engine, EngineConfig, EngineEvent, EngineSnapshot, GameOverStatus, GoalProgress, GoalSystem,
    InputFrame, LockDownMode, PieceAction, PieceRotation, PieceType, RotationDirection,
    SnapshotCell, TSpinCorners, TSpinKind, EXTENDED_LOCK_RESET_BUDGET, LOCK_DOWN_SECONDS,
    MAX_LEVEL, MIN_LEVEL,
};
use crate::level::LevelPlugin;

#[derive(States, PartialEq, Eq, Debug, Clone, Hash, Default)]
pub enum GameState {
    #[default]
    Loading,
    MainMenu,
    InGame,
    GameOver,
}

pub struct GamePlugin;

impl Plugin for GamePlugin {
    fn build(&self, app: &mut App) {
        app.init_state::<GameState>()
            .add_loading_state(
                LoadingState::new(GameState::Loading)
                    .load_collection::<crate::assets::GameAssets>()
                    .continue_to_state(GameState::InGame),
            )
            .add_plugins(LevelPlugin);

        #[cfg(debug_assertions)]
        {
            app.add_plugins(FrameTimeDiagnosticsPlugin::default())
                .add_plugins(LogDiagnosticsPlugin::default());
        }
    }
}
