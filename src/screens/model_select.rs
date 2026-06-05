//! Model select: choose which AI model drives the Watch-AI session.
//!
//! Reached from the main menu's "Watch AI" entry, *before* mode select. Each row
//! is a model from the [`ModelRegistry`]; selecting it writes the registry's
//! current selection (read by [`crate::ai::sandbox`] when the session starts) and
//! advances to [`GameState::ModeSelect`]. Esc returns to the main menu.
//!
//! Mirrors [`mode_select`](super::mode_select) — same `FocusList` + focus-nav +
//! activate shape — so it navigates identically to every other menu screen.

use bevy::prelude::*;

use crate::ai::ModelRegistry;
use crate::assets::GameAssets;
use crate::ui::focus::{
    clicked_focusable, focus_navigation, read_nav_action, FocusList, Focusable, NavAction,
};
use crate::ui::widgets::{label_text, menu_button, screen_root, title_text};
use crate::GameState;

pub struct ModelSelectPlugin;

impl Plugin for ModelSelectPlugin {
    fn build(&self, app: &mut App) {
        app.register_type::<ModelSelectRoot>()
            .add_systems(OnEnter(GameState::ModelSelect), setup)
            .add_systems(
                Update,
                (focus_navigation::<ModelSelectRoot>, activate)
                    .chain()
                    .run_if(in_state(GameState::ModelSelect)),
            );
    }
}

#[derive(Component, Reflect)]
#[reflect(Component)]
struct ModelSelectRoot;

fn setup(mut commands: Commands, assets: Res<GameAssets>, registry: Res<ModelRegistry>) {
    commands.spawn((Camera2d, DespawnOnExit(GameState::ModelSelect)));
    let labels = registry.labels();
    let root = commands
        .spawn((
            ModelSelectRoot,
            FocusList::new(labels.len()),
            screen_root(),
            DespawnOnExit(GameState::ModelSelect),
            children![
                title_text("Select Model", assets.font.clone()),
                label_text("Enter to choose  -  Esc to go back", assets.font.clone()),
            ],
        ))
        .id();

    for (index, label) in labels.into_iter().enumerate() {
        let button = commands
            .spawn(menu_button(index, label, assets.font.clone()))
            .id();
        commands.entity(root).add_child(button);
    }
}

fn activate(
    keys: Res<ButtonInput<KeyCode>>,
    list: Single<&FocusList, With<ModelSelectRoot>>,
    clicks: Query<(&Focusable, &Interaction)>,
    mut registry: ResMut<ModelRegistry>,
    mut next: ResMut<NextState<GameState>>,
) {
    // Keyboard (Enter/Space) or a mouse click both select the focused model.
    let nav =
        read_nav_action(&keys, *list).or_else(|| clicked_focusable(&clicks).map(NavAction::Select));
    match nav {
        Some(NavAction::Select(index)) => {
            registry.select(index);
            info!("Watch AI model: {}", registry.selected_label());
            next.set(GameState::ModeSelect);
        }
        Some(NavAction::Back) => next.set(GameState::MainMenu),
        None => {}
    }
}
