use bevy::app::App;
#[cfg(debug_assertions)]
use bevy::diagnostic::{FrameTimeDiagnosticsPlugin, LogDiagnosticsPlugin};
use bevy::prelude::*;
use bevy_asset_loader::prelude::*;

mod assets;
pub mod engine;
mod features;
pub mod high_scores;
pub(crate) mod level;
pub mod player;
mod screens;
pub mod settings;
pub(crate) mod ui;
pub mod variant;

pub use crate::engine::{
    apply_grounded_move_or_rotation, breaks_back_to_back, classify_t_spin, fall_duration,
    fall_speed_seconds, fixed_goal_for_level, goal_for_level, is_block_out, is_lock_out,
    is_top_out, qualifies_for_back_to_back, soft_drop_duration, soft_drop_speed_seconds,
    t_spin_corners, variable_goal_for_level, variable_goal_units, ActivePiece, ActivePieceSnapshot,
    Engine, EngineConfig, EngineEvent, EngineScoreAction, EngineSnapshot, GameOverStatus,
    GoalProgress, GoalSystem, InputFrame, LockDownMode, PieceAction, PieceRotation, PieceType,
    RotationDirection, SnapshotCell, TSpinCorners, TSpinKind, EXTENDED_LOCK_RESET_BUDGET,
    LOCK_DOWN_SECONDS, MAX_LEVEL, MIN_LEVEL,
};
use crate::level::LevelPlugin;

/// Top-level screen the app is on. Drives which plugins' systems run and which
/// UI is spawned. Flow: `Loading` (asset load) -> `Title` -> `MainMenu`, with
/// `ModeSelect`/`Options`/`Help`/`HighScores` reachable from the menu, `Playing`
/// the active game, and `Paused`/`GameOver` layered over/after it.
#[derive(States, PartialEq, Eq, Debug, Clone, Hash, Default)]
pub enum GameState {
    /// Asset loading; advances to [`GameState::Title`] when assets are ready.
    #[default]
    Loading,
    /// Splash/title screen; any key advances to the main menu.
    Title,
    /// Root navigation menu (Play / Options / Help / High Scores).
    MainMenu,
    /// Choose a [`variant::Variant`] (Marathon/Sprint/Ultra) before playing.
    ModeSelect,
    /// Settings screen (filled by the options feature).
    Options,
    /// Controls/about screen (filled by the help feature).
    Help,
    /// Leaderboards (filled by the high-scores feature).
    HighScores,
    /// Active gameplay (formerly `InGame`). The engine is authoritative here.
    Playing,
    /// Gameplay paused; overlay shown by the pause feature, engine frozen.
    Paused,
    /// Post-game results; offers restart / back to menu.
    GameOver,
}

pub struct GamePlugin;

impl Plugin for GamePlugin {
    fn build(&self, app: &mut App) {
        app.init_state::<GameState>()
            .add_loading_state(
                LoadingState::new(GameState::Loading)
                    .load_collection::<crate::assets::GameAssets>()
                    .continue_to_state(GameState::Title),
            )
            // Shared M1 contracts (defined once, read everywhere).
            .init_resource::<crate::settings::GameSettings>()
            .init_resource::<crate::variant::ActiveVariant>()
            .init_resource::<crate::variant::VariantProgress>()
            .init_resource::<crate::high_scores::HighScores>()
            // Gameplay + screen-shell + feature plugins.
            .add_plugins(LevelPlugin)
            .add_plugins(crate::screens::ScreensPlugin)
            .add_plugins(crate::features::FeaturesPlugin);

        #[cfg(debug_assertions)]
        {
            app.add_plugins(FrameTimeDiagnosticsPlugin::default())
                .add_plugins(LogDiagnosticsPlugin::default());
        }
    }
}
