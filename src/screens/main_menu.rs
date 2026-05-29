//! Main menu: Play / Options / Help / High Scores, keyboard-navigable.
//!
//! Reference implementation of the shared focus-navigation pattern: a
//! [`FocusList`] on the root, [`menu_button`] rows each tagged with a
//! [`MainMenuAction`], [`focus_navigation`] for Up/Down + highlight, and a
//! handler that reads Enter (activate focused) / Esc.

use bevy::prelude::*;

use crate::assets::GameAssets;
use crate::ui::focus::{focus_navigation, read_nav_action, FocusList, Focusable, NavAction};
use crate::ui::widgets::{menu_button, screen_root, title_text};
use crate::GameState;

pub struct MainMenuPlugin;

impl Plugin for MainMenuPlugin {
    fn build(&self, app: &mut App) {
        app.register_type::<MainMenuRoot>()
            .register_type::<MainMenuAction>()
            .add_systems(OnEnter(GameState::MainMenu), setup)
            .add_systems(
                Update,
                (focus_navigation::<MainMenuRoot>, activate)
                    .chain()
                    .run_if(in_state(GameState::MainMenu)),
            );
    }
}

#[derive(Component, Reflect)]
#[reflect(Component)]
struct MainMenuRoot;

#[derive(Component, Clone, Copy, Reflect)]
#[reflect(Component)]
enum MainMenuAction {
    Play,
    Options,
    Help,
    HighScores,
}

const ITEMS: [(MainMenuAction, &str); 4] = [
    (MainMenuAction::Play, "Play"),
    (MainMenuAction::Options, "Options"),
    (MainMenuAction::Help, "Help"),
    (MainMenuAction::HighScores, "High Scores"),
];

fn setup(mut commands: Commands, assets: Res<GameAssets>) {
    commands.spawn((Camera2d, DespawnOnExit(GameState::MainMenu)));
    let root = commands
        .spawn((
            MainMenuRoot,
            FocusList::new(ITEMS.len()),
            screen_root(),
            DespawnOnExit(GameState::MainMenu),
            children![title_text("TETR ONLINE", assets.font.clone())],
        ))
        .id();

    for (index, (action, label)) in ITEMS.into_iter().enumerate() {
        let button = commands
            .spawn((menu_button(index, label, assets.font.clone()), action))
            .id();
        commands.entity(root).add_child(button);
    }
}

/// On Enter, route to the focused item's screen. Esc is a no-op here (the main
/// menu is the root). Also paints the focused row's "pressed" color briefly via
/// the focus helper's normal restyle on the next frame.
fn activate(
    keys: Res<ButtonInput<KeyCode>>,
    lists: Query<&FocusList, With<MainMenuRoot>>,
    actions: Query<(&Focusable, &MainMenuAction)>,
    mut next: ResMut<NextState<GameState>>,
) {
    let Ok(list) = lists.single() else {
        return;
    };
    let Some(NavAction::Select(index)) = read_nav_action(&keys, list) else {
        return;
    };

    for (focusable, action) in &actions {
        if focusable.index != index {
            continue;
        }
        match action {
            MainMenuAction::Play => next.set(GameState::ModeSelect),
            MainMenuAction::Options => next.set(GameState::Options),
            MainMenuAction::Help => next.set(GameState::Help),
            MainMenuAction::HighScores => next.set(GameState::HighScores),
        }
    }
}
