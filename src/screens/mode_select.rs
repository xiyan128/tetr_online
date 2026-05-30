//! Mode select: choose a [`Variant`] then start the game.
//!
//! Each row corresponds to a [`Variant`]; selecting it writes [`ActiveVariant`]
//! (which the engine bridge reads when building the engine) and transitions to
//! [`GameState::Playing`]. Esc returns to the main menu.

use bevy::prelude::*;

use crate::assets::GameAssets;
use crate::ui::focus::{
    clicked_focusable, focus_navigation, read_nav_action, FocusList, Focusable, NavAction,
};
use crate::ui::widgets::{label_text, menu_button, screen_root, title_text};
use crate::variant::{ActiveVariant, Variant};
use crate::GameState;

pub struct ModeSelectPlugin;

impl Plugin for ModeSelectPlugin {
    fn build(&self, app: &mut App) {
        app.register_type::<ModeSelectRoot>()
            .add_systems(OnEnter(GameState::ModeSelect), setup)
            .add_systems(
                Update,
                (focus_navigation::<ModeSelectRoot>, activate)
                    .chain()
                    .run_if(in_state(GameState::ModeSelect)),
            );
    }
}

#[derive(Component, Reflect)]
#[reflect(Component)]
struct ModeSelectRoot;

fn setup(mut commands: Commands, assets: Res<GameAssets>) {
    commands.spawn((Camera2d, DespawnOnExit(GameState::ModeSelect)));
    let root = commands
        .spawn((
            ModeSelectRoot,
            FocusList::new(Variant::ALL.len()),
            screen_root(),
            DespawnOnExit(GameState::ModeSelect),
            children![
                title_text("Select Mode", assets.font.clone()),
                label_text("Enter to play  -  Esc to go back", assets.font.clone()),
            ],
        ))
        .id();

    for (index, variant) in Variant::ALL.into_iter().enumerate() {
        let button = commands
            .spawn(menu_button(
                index,
                variant.display_name(),
                assets.font.clone(),
            ))
            .id();
        commands.entity(root).add_child(button);
    }
}

fn activate(
    keys: Res<ButtonInput<KeyCode>>,
    list: Single<&FocusList, With<ModeSelectRoot>>,
    clicks: Query<(&Focusable, &Interaction)>,
    mut active: ResMut<ActiveVariant>,
    mut next: ResMut<NextState<GameState>>,
) {
    // Keyboard (Enter/Space) or a mouse click both select the focused variant.
    let nav =
        read_nav_action(&keys, *list).or_else(|| clicked_focusable(&clicks).map(NavAction::Select));
    match nav {
        Some(NavAction::Select(index)) => {
            if let Some(&variant) = Variant::ALL.get(index) {
                *active = ActiveVariant(variant);
                next.set(GameState::Playing);
            }
        }
        Some(NavAction::Back) => next.set(GameState::MainMenu),
        None => {}
    }
}
