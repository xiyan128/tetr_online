//! Options screen shell. Navigation + back only; the actual settings widgets
//! (next-count, toggles, volumes, rebinds reading [`GameSettings`]) are added by
//! the options feature plugin (`src/features/options.rs`) onto this same state.
//!
//! [`GameSettings`]: crate::settings::GameSettings

use bevy::prelude::*;

use crate::GameState;
use crate::assets::GameAssets;
use crate::ui::widgets::{label_text, screen_root, title_text};

pub struct OptionsScreenPlugin;

impl Plugin for OptionsScreenPlugin {
    fn build(&self, app: &mut App) {
        app.register_type::<OptionsRoot>()
            .add_systems(OnEnter(GameState::Options), setup)
            .add_systems(Update, back.run_if(in_state(GameState::Options)));
    }
}

/// Screen-root marker. The options feature queries this to attach its widgets.
#[derive(Component, Reflect)]
#[reflect(Component)]
pub struct OptionsRoot;

fn setup(mut commands: Commands, assets: Res<GameAssets>) {
    commands.spawn((Camera2d, DespawnOnExit(GameState::Options)));
    commands.spawn((
        OptionsRoot,
        screen_root(),
        DespawnOnExit(GameState::Options),
        children![
            title_text("Options", assets.font.clone()),
            label_text("Esc to go back", assets.font.clone()),
        ],
    ));
}

fn back(keys: Res<ButtonInput<KeyCode>>, mut next: ResMut<NextState<GameState>>) {
    if keys.just_pressed(KeyCode::Escape) {
        next.set(GameState::MainMenu);
    }
}
