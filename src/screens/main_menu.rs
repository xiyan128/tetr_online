//! Main menu: Play / Options / Help / High Scores, keyboard-navigable.
//!
//! Reference implementation of the shared focus-navigation pattern: a
//! [`FocusList`] on the root, [`menu_button`] rows each tagged with a
//! [`MainMenuAction`], [`focus_navigation`] for Up/Down + highlight, and a
//! handler that reads Enter (activate focused) / Esc.

use bevy::prelude::*;

use crate::GameState;
use crate::assets::GameAssets;
use crate::ui::focus::{
    FocusList, Focusable, NavAction, clicked_focusable, focus_navigation, read_nav_action,
};
use crate::ui::widgets::{menu_button, screen_root, title_text};

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
    Versus,
    WatchAi,
    Options,
    Help,
    HighScores,
}

const ITEMS: [(MainMenuAction, &str); 6] = [
    (MainMenuAction::Play, "Play"),
    (MainMenuAction::Versus, "Versus"),
    (MainMenuAction::WatchAi, "Watch AI"),
    (MainMenuAction::Options, "Options"),
    (MainMenuAction::Help, "Help"),
    (MainMenuAction::HighScores, "High Scores"),
];

fn setup(mut commands: Commands, assets: Res<GameAssets>) {
    commands.spawn((
        crate::ui::widgets::menu_camera(),
        DespawnOnExit(GameState::MainMenu),
    ));
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
///
/// **Play** seats you and goes to mode select; **Watch AI** picks its bot in
/// the seat picker first, then the mode — both land in a one-seat Solo
/// session, differing only in who occupies the seat. **Versus** configures
/// two seats. The seats are written HERE (and by the pickers), so a previous
/// run's configuration can never leak into the next.
#[allow(clippy::too_many_arguments)] // a Bevy system's params are its dependency list
fn activate(
    keys: Res<ButtonInput<KeyCode>>,
    list: Single<&FocusList, With<MainMenuRoot>>,
    actions: Query<(&Focusable, &MainMenuAction)>,
    clicks: Query<(&Focusable, &Interaction), Changed<Interaction>>,
    mut session: ResMut<crate::session::SessionConfig>,
    mut setup_kind: ResMut<crate::screens::session_setup::SetupKind>,
    mut next: ResMut<NextState<GameState>>,
) {
    // Select via keyboard (Enter/Space on the focused row) or a mouse click on a
    // row. Esc is a no-op here (the main menu is the root).
    let Some(index) = read_nav_action(&keys, *list)
        .and_then(|nav| match nav {
            NavAction::Select(index) => Some(index),
            NavAction::Back => None,
        })
        .or_else(|| clicked_focusable(&clicks))
    else {
        return;
    };

    for (focusable, action) in &actions {
        if focusable.index != index {
            continue;
        }
        match action {
            MainMenuAction::Play => {
                session.seats[0] = crate::session::Participant::Human;
                next.set(GameState::ModeSelect);
            }
            MainMenuAction::Versus => {
                *setup_kind = crate::screens::session_setup::SetupKind::Versus;
                next.set(GameState::SessionSetup);
            }
            MainMenuAction::WatchAi => {
                *setup_kind = crate::screens::session_setup::SetupKind::WatchAi;
                // Watch-AI picks its bot in the seat picker, then the mode.
                next.set(GameState::SessionSetup);
            }
            MainMenuAction::Options => next.set(GameState::Options),
            MainMenuAction::Help => next.set(GameState::Help),
            MainMenuAction::HighScores => next.set(GameState::HighScores),
        }
    }
}
