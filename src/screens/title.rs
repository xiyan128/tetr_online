//! Title screen: a splash that advances to the main menu on any key.

use bevy::prelude::*;

use crate::assets::GameAssets;
use crate::ui::widgets::{label_text, screen_root, title_text};
use crate::GameState;

pub struct TitleScreenPlugin;

impl Plugin for TitleScreenPlugin {
    fn build(&self, app: &mut App) {
        app.register_type::<TitleUi>()
            .add_systems(OnEnter(GameState::Title), setup)
            .add_systems(Update, advance.run_if(in_state(GameState::Title)));
    }
}

#[derive(Component, Reflect)]
#[reflect(Component)]
struct TitleUi;

fn setup(mut commands: Commands, assets: Res<GameAssets>) {
    commands.spawn((Camera2d, DespawnOnExit(GameState::Title)));
    commands.spawn((
        TitleUi,
        screen_root(),
        DespawnOnExit(GameState::Title),
        children![
            title_text("TETR ONLINE", assets.font.clone()),
            label_text("Press any key to start", assets.font.clone()),
        ],
    ));
}

/// Advance to the main menu on any keyboard press.
fn advance(keys: Res<ButtonInput<KeyCode>>, mut next: ResMut<NextState<GameState>>) {
    if keys.get_just_pressed().next().is_some() {
        next.set(GameState::MainMenu);
    }
}
