//! Help screen shell. Navigation + back only; the controls/about content is
//! added by the help feature plugin (`src/features/help.rs`) onto this state.

use bevy::prelude::*;

use crate::GameState;
use crate::assets::GameAssets;
use crate::ui::widgets::{label_text, screen_root, title_text};

pub struct HelpScreenPlugin;

impl Plugin for HelpScreenPlugin {
    fn build(&self, app: &mut App) {
        app.register_type::<HelpRoot>()
            .add_systems(OnEnter(GameState::Help), setup)
            .add_systems(Update, back.run_if(in_state(GameState::Help)));
    }
}

/// Screen-root marker. The help feature queries this to attach its content.
#[derive(Component, Reflect)]
#[reflect(Component)]
pub struct HelpRoot;

fn setup(mut commands: Commands, assets: Res<GameAssets>) {
    commands.spawn((Camera2d, DespawnOnExit(GameState::Help)));
    commands.spawn((
        HelpRoot,
        screen_root(),
        DespawnOnExit(GameState::Help),
        children![
            title_text("Help", assets.font.clone()),
            label_text("Esc to go back", assets.font.clone()),
        ],
    ));
}

fn back(keys: Res<ButtonInput<KeyCode>>, mut next: ResMut<NextState<GameState>>) {
    if keys.just_pressed(KeyCode::Escape) {
        next.set(GameState::MainMenu);
    }
}
