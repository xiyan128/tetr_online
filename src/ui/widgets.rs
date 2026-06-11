//! Shared menu widgets + theme.
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

use super::focus::{FocusLabel, Focusable};

/// The Kissaten palette: a warm neutral ramp plus two saturated signals.
/// Pure black and pure white are banned; `GRID` (near-black brown) and `TEXT`
/// (aged-paper cream) take their roles. `ATTACK` is deliberately the most
/// saturated value in the system — nothing else may approach its chroma.
pub mod theme {
    use bevy::prelude::*;

    /// Warm charcoal: the field and panel ground (also the clear color).
    pub const BG: Color = Color::srgb(0.1804, 0.1686, 0.1569); // #2E2B28
    /// Gridlines, meter tracks at rest, overlay scrims.
    pub const GRID: Color = Color::srgb(0.1373, 0.1255, 0.1137); // #23201D
    /// Board border, dividers, resting button borders.
    pub const FRAME: Color = Color::srgb(0.3333, 0.3137, 0.2902); // #55504A
    /// Garbage minos: warm gray, zero chroma — dead weight next to live pieces.
    pub const GARBAGE: Color = Color::srgb(0.4314, 0.4118, 0.3882); // #6E6963
    /// Primary type and numerals (aged paper cream).
    pub const TEXT: Color = Color::srgb(0.9176, 0.8902, 0.8235); // #EAE3D2
    /// Labels, hints, inactive states. Chrome only — anything the player must
    /// read mid-match uses `TEXT`.
    pub const TEXT_DIM: Color = Color::srgb(0.6039, 0.5765, 0.5412); // #9A938A
    /// Amber: B2B / spin callouts, hover, focus, selection. Chrome and gutters
    /// only, never on field cells (it would read as the O piece).
    pub const ACCENT: Color = Color::srgb(0.8510, 0.6510, 0.2824); // #D9A648
    /// Incoming damage and danger. The saturation ceiling of the whole system.
    pub const ATTACK: Color = Color::srgb(0.8196, 0.2941, 0.2588); // #D14B42

    /// Dogica at native multiples only (8 px grid): 32 / 24 / 16.
    pub const TITLE_FONT_SIZE: f32 = 32.0;
    pub const NUMERAL_FONT_SIZE: f32 = 24.0;
    pub const BUTTON_FONT_SIZE: f32 = 16.0;
    /// Departure Mono, the working voice: body 14, micro 12.
    pub const LABEL_FONT_SIZE: f32 = 14.0;
    pub const MICRO_FONT_SIZE: f32 = 12.0;
}

/// The camera every menu screen spawns: it composites over the ambient
/// background pass (`features::ambient_wave`) instead of clearing it, so the
/// Kissaten ground stays alive behind the chrome. Pair with a
/// [`DespawnOnExit`] for the screen's state on the caller side.
pub fn menu_camera() -> impl Bundle {
    (
        Camera2d,
        Camera {
            clear_color: bevy::camera::ClearColorConfig::None,
            ..default()
        },
    )
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
    menu_button_sized(index, label, font, 220.0)
}

/// [`menu_button`] with an explicit row width, for screens whose labels outgrow
/// the default 220 px (the pixel font runs ~15 px per glyph at the button size,
/// so the default fits ~13 characters). One line per row is the contract — a
/// label that wraps has outgrown its screen's width budget.
pub fn menu_button_sized(
    index: usize,
    label: impl Into<String>,
    font: Handle<Font>,
    width: f32,
) -> impl Bundle {
    (
        Button,
        Focusable::new(index),
        Node {
            width: px(width),
            height: px(40),
            margin: UiRect::all(px(4)),
            border: UiRect::all(px(1)),
            border_radius: BorderRadius::all(px(2)),
            justify_content: JustifyContent::Center,
            align_items: AlignItems::Center,
            ..default()
        },
        BackgroundColor(theme::BG),
        BorderColor::all(theme::FRAME),
        children![(
            FocusLabel,
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
