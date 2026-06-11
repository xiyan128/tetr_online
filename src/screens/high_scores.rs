//! High-scores screen shell. Navigation + back only; the leaderboard tables
//! (reading [`HighScores`] per [`Variant`]) are rendered by the high-scores
//! feature plugin (`src/features/high_scores.rs`) onto this state.
//!
//! [`HighScores`]: crate::high_scores::HighScores
//! [`Variant`]: crate::variant::Variant

use bevy::prelude::*;

use crate::GameState;
use crate::assets::GameAssets;
use crate::ui::widgets::{label_text, screen_root, title_text};

pub struct HighScoresScreenPlugin;

impl Plugin for HighScoresScreenPlugin {
    fn build(&self, app: &mut App) {
        app.register_type::<HighScoresRoot>()
            .add_systems(OnEnter(GameState::HighScores), setup)
            .add_systems(Update, back.run_if(in_state(GameState::HighScores)));
    }
}

/// Screen-root marker. The high-scores feature queries this to attach tables.
#[derive(Component, Reflect)]
#[reflect(Component)]
pub struct HighScoresRoot;

fn setup(mut commands: Commands, assets: Res<GameAssets>) {
    commands.spawn((Camera2d, DespawnOnExit(GameState::HighScores)));
    commands.spawn((
        HighScoresRoot,
        screen_root(),
        DespawnOnExit(GameState::HighScores),
        children![
            title_text("High Scores", assets.font.clone()),
            label_text("Esc to go back", assets.font.clone()),
        ],
    ));
}

fn back(keys: Res<ButtonInput<KeyCode>>, mut next: ResMut<NextState<GameState>>) {
    if keys.just_pressed(KeyCode::Escape) {
        next.set(GameState::MainMenu);
    }
}
