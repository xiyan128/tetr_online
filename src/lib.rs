use bevy::app::App;
#[cfg(debug_assertions)]
use bevy::diagnostic::{FrameTimeDiagnosticsPlugin, LogDiagnosticsPlugin};
use bevy::prelude::*;
use bevy_asset_loader::prelude::*;

mod assets;
mod engine;
mod level;

pub use crate::engine::{
    apply_grounded_move_or_rotation, fall_duration, fall_speed_seconds, soft_drop_duration,
    soft_drop_speed_seconds, ActivePiece, Engine, EngineConfig, EngineEvent, EngineSnapshot,
    InputFrame, LockDownMode, PieceAction, PieceRotation, PieceType, RotationDirection,
    EXTENDED_LOCK_RESET_BUDGET, LOCK_DOWN_SECONDS, MAX_LEVEL, MIN_LEVEL,
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
