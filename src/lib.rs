use bevy::app::App;
#[cfg(debug_assertions)]
use bevy::diagnostic::{FrameTimeDiagnosticsPlugin, LogDiagnosticsPlugin};
use bevy::prelude::*;
use bevy_asset_loader::prelude::*;

mod level;
mod core;
mod assets;

use crate::level::LevelPlugin;

#[derive(States, PartialEq, Eq, Debug, Clone, Hash, Default)]
enum GameState {
    #[default]
    Loading,
    MainMenu,
    // #[default]
    InGame,
}


pub struct GamePlugin;

impl Plugin for GamePlugin {
    fn build(&self, app: &mut App) {
        app.add_state::<GameState>()
            .add_loading_state(
                LoadingState::new(GameState::Loading)
                    .continue_to_state(GameState::InGame))
            .add_plugin(LevelPlugin);
        // .add_plugin(LoadingPlugin)
        // .add_plugin(MenuPlugin)
        // .add_plugin(ActionsPlugin)
        // .add_plugin(InternalAudioPlugin)
        // .add_plugin(PlayerPlugin);

        #[cfg(debug_assertions)]
        {
            app.add_plugin(FrameTimeDiagnosticsPlugin::default())
                .add_plugin(LogDiagnosticsPlugin::default());
        }
    }
}
