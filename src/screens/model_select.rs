//! Model select: choose which AI model drives the Watch-AI session.
//!
//! Reached from the main menu's "Watch AI" entry, *before* mode select. Each row
//! is a model from the [`ModelRegistry`] under its **short** label (sized to the
//! shared `menu_button` row); the focused row's one-line description renders in a
//! fixed **detail pane** under the list, updating as the cursor moves — long
//! model descriptions live there instead of overflowing the buttons. Selecting a
//! row writes the registry's current selection (read by [`crate::ai::sandbox`]
//! when the session starts) and advances to [`GameState::ModeSelect`]; the screen
//! opens focused on the current selection. Esc returns to the main menu.
//!
//! Mirrors [`mode_select`](super::mode_select) — same `FocusList` + focus-nav +
//! activate shape — so it navigates identically to every other menu screen.

use bevy::prelude::*;

use crate::ai::ModelRegistry;
use crate::assets::GameAssets;
use crate::ui::focus::{
    clicked_focusable, focus_navigation, read_nav_action, FocusList, Focusable, NavAction,
};
use crate::ui::widgets::{label_text, menu_button_sized, screen_root, theme, title_text};
use crate::GameState;

pub struct ModelSelectPlugin;

impl Plugin for ModelSelectPlugin {
    fn build(&self, app: &mut App) {
        app.register_type::<ModelSelectRoot>()
            .add_systems(OnEnter(GameState::ModelSelect), setup)
            .add_systems(
                Update,
                (focus_navigation::<ModelSelectRoot>, update_detail, activate)
                    .chain()
                    .run_if(in_state(GameState::ModelSelect)),
            );
    }
}

#[derive(Component, Reflect)]
#[reflect(Component)]
struct ModelSelectRoot;

/// Marks the detail pane's text — the focused model's one-line description.
#[derive(Component)]
struct ModelDetail;

fn setup(mut commands: Commands, assets: Res<GameAssets>, registry: Res<ModelRegistry>) {
    commands.spawn((Camera2d, DespawnOnExit(GameState::ModelSelect)));
    let labels = registry.labels();
    let focused = registry.selected_index();
    let root = commands
        .spawn((
            ModelSelectRoot,
            // Open with the cursor on the current selection, not row 0.
            FocusList {
                index: focused,
                count: labels.len(),
            },
            screen_root(),
            DespawnOnExit(GameState::ModelSelect),
            children![title_text("Select Model", assets.font.clone())],
        ))
        .id();

    for (index, label) in labels.into_iter().enumerate() {
        // Wider than the default 220 px row: model names run to 17 characters and
        // the pixel font needs ~15 px per glyph (`labels_fit_a_menu_row` pins the
        // length budget this width buys).
        let button = commands
            .spawn(menu_button_sized(index, label, assets.font.clone(), 320.0))
            .id();
        commands.entity(root).add_child(button);
    }

    // The detail pane (below the list) and the nav hint (below the pane). The pane
    // reserves three lines of height so the hint doesn't jump as the focused
    // model's description wraps to a different line count.
    let detail = commands
        .spawn((
            ModelDetail,
            Text::new(registry.detail(focused)),
            TextFont {
                font: assets.font.clone(),
                font_size: theme::LABEL_FONT_SIZE,
                ..default()
            },
            TextColor(theme::TEXT_DIM),
            TextLayout::new_with_justify(Justify::Center),
            Node {
                max_width: px(560),
                min_height: px(64),
                margin: UiRect::top(px(8)),
                ..default()
            },
        ))
        .id();
    let hint = commands
        .spawn(label_text(
            "Enter to choose  -  Esc to go back",
            assets.font.clone(),
        ))
        .id();
    commands.entity(root).add_child(detail);
    commands.entity(root).add_child(hint);
}

/// Keep the detail pane on the focused row: whenever the cursor moves (keyboard
/// or hover — anything that mutates the [`FocusList`]), swap in that model's
/// description.
fn update_detail(
    list: Single<&FocusList, (With<ModelSelectRoot>, Changed<FocusList>)>,
    registry: Res<ModelRegistry>,
    mut detail: Single<&mut Text, With<ModelDetail>>,
) {
    detail.0 = registry.detail(list.index).to_string();
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
