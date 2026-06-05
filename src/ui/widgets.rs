//! Shared menu widgets + theme (M1 shared UI).
//!
//! Small Bevy-UI bundle builders so every screen (Title/MainMenu/ModeSelect/
//! Options/Help/HighScores) and the menu-related feature agents share one look:
//!
//! * [`theme`] — colors + sizes.
//! * [`screen_root`] — a full-window centered column to parent a menu into.
//! * [`title_text`] / [`label_text`] — headings and body labels.
//! * [`menu_button`] — a focusable row carrying a [`Focusable`] index (styled by
//!   the [`focus_navigation`](super::focus::focus_navigation) helper).
//!
//! These take a [`Handle<Font>`] (from [`GameAssets`](crate::assets::GameAssets))
//! so callers don't depend on asset loading details.

use bevy::prelude::*;

use super::focus::Focusable;

/// Shared palette + sizing. Tuned to read on the dark `ClearColor` background.
pub mod theme {
    use bevy::prelude::*;

    pub const TEXT: Color = Color::srgb(0.92, 0.92, 0.92);
    pub const TEXT_DIM: Color = Color::srgb(0.6, 0.6, 0.6);
    pub const ACCENT: Color = Color::srgb(0.35, 0.75, 0.35);

    /// Idle menu-row background.
    pub const BUTTON_NORMAL: Color = Color::srgb(0.15, 0.15, 0.15);
    /// Keyboard-focused (or hovered) menu-row background.
    pub const BUTTON_FOCUSED: Color = Color::srgb(0.30, 0.30, 0.30);
    /// Pressed/activated menu-row background.
    pub const BUTTON_PRESSED: Color = Color::srgb(0.35, 0.75, 0.35);

    pub const TITLE_FONT_SIZE: f32 = 28.0;
    pub const LABEL_FONT_SIZE: f32 = 14.0;
    pub const BUTTON_FONT_SIZE: f32 = 16.0;
}

/// A full-window, centered vertical column to parent a screen's content into.
/// Pair with a [`DespawnOnExit`] for the screen's state on the caller side.
pub fn screen_root() -> impl Bundle {
    (Node {
        width: percent(100),
        height: percent(100),
        flex_direction: FlexDirection::Column,
        align_items: AlignItems::Center,
        justify_content: JustifyContent::Center,
        row_gap: px(12),
        ..default()
    },)
}

/// A large heading.
pub fn title_text(text: impl Into<String>, font: Handle<Font>) -> impl Bundle {
    (
        Text::new(text),
        TextFont {
            font,
            font_size: theme::TITLE_FONT_SIZE,
            ..default()
        },
        TextColor(theme::TEXT),
        Node {
            margin: UiRect::all(px(12)),
            ..default()
        },
    )
}

/// A body / hint label.
pub fn label_text(text: impl Into<String>, font: Handle<Font>) -> impl Bundle {
    (
        Text::new(text),
        TextFont {
            font,
            font_size: theme::LABEL_FONT_SIZE,
            ..default()
        },
        TextColor(theme::TEXT_DIM),
    )
}

/// A focusable menu row. `index` ties it to the screen's
/// [`FocusList`](super::focus::FocusList); the focus helper restyles it when
/// focused. The returned bundle includes a child text label.
///
/// Callers usually also insert an action marker component on the same entity to
/// identify which row was selected, e.g.
/// `commands.spawn(menu_button(0, "Play", font)).insert(MyAction::Play);`.
pub fn menu_button(index: usize, label: impl Into<String>, font: Handle<Font>) -> impl Bundle {
    (
        Button,
        Focusable::new(index),
        Node {
            width: px(220),
            height: px(34),
            margin: UiRect::all(px(4)),
            justify_content: JustifyContent::Center,
            align_items: AlignItems::Center,
            ..default()
        },
        BackgroundColor(theme::BUTTON_NORMAL),
        children![(
            Text::new(label),
            TextFont {
                font,
                font_size: theme::BUTTON_FONT_SIZE,
                ..default()
            },
            TextColor(theme::TEXT),
        )],
    )
}
