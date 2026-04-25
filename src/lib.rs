use bevy::app::App;
#[cfg(debug_assertions)]
use bevy::diagnostic::{FrameTimeDiagnosticsPlugin, LogDiagnosticsPlugin};
use bevy::prelude::*;
use bevy_asset_loader::prelude::*;

mod assets;
mod core;
mod level;
mod utils;

use crate::level::LevelPlugin;

#[derive(States, PartialEq, Eq, Debug, Clone, Hash, Default)]
enum GameState {
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
