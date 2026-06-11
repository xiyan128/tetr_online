//! Help feature (A1.8).
//!
//! Renders the controls + rules reference on
//! [`GameState::Help`](crate::GameState::Help), attaching its content under the
//! [`HelpRoot`](crate::screens::HelpRoot) that the screen shell spawns. The
//! panel lists the *current* [`Keybinds`](crate::settings::Keybinds) from
//! [`GameSettings`](crate::settings::GameSettings) and a short, spec-accurate
//! how-to-play section (Matrix, Tetrimino, Mino, Lock Down, Hard/Soft Drop,
//! Hold, Ghost, T-Spin, Back-to-Back).
//!
//! The body is a fixed-height, vertically-scrolling viewport
//! ([`Overflow::scroll_y`]) so long content stays reachable without overflowing
//! the window. It is keyboard-navigable: Up/Down (and W/S) scroll by a line,
//! PageUp/PageDown by a page, Home/End jump to the extremes. Esc-to-back is
//! owned by the screen shell.

use bevy::prelude::*;

use crate::GameState;
use crate::assets::GameAssets;
use crate::screens::HelpRoot;
use crate::settings::{GameAction, GameSettings, Keybinds};
use crate::ui::theme;

/// Help-screen content: a scrollable controls/rules reference.
pub struct HelpPlugin;

impl Plugin for HelpPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(OnEnter(GameState::Help), spawn_help_content)
            .add_systems(Update, scroll_help.run_if(in_state(GameState::Help)));
    }
}

/// Marker for the scrolling viewport node (the one carrying [`ScrollPosition`]).
#[derive(Component)]
struct HelpScroll;

/// Logical pixels scrolled per line-step / per page-step. The page step is a
/// little under the viewport height so a line of context carries over.
const SCROLL_LINE_STEP: f32 = 28.0;
const SCROLL_PAGE_STEP: f32 = 320.0;
/// Height of the scrolling viewport, in logical pixels.
const VIEWPORT_HEIGHT: f32 = 420.0;
/// A large value used to clamp to the bottom; the layout system re-clamps the
/// `ScrollPosition` to the real content height each frame, so any value past the
/// end resolves to "fully scrolled down".
const SCROLL_MAX: f32 = 100_000.0;

/// Append the scrollable help body under the shell's [`HelpRoot`] column.
///
/// We attach as a child of the existing root (rather than spawning a second
/// screen root) so we reuse the shell's camera, "Help" title and "Esc to go
/// back" hint and avoid a duplicate centered overlay.
fn spawn_help_content(
    mut commands: Commands,
    assets: Res<GameAssets>,
    settings: Res<GameSettings>,
    // No shell root this frame ⇒ `Single` skips the system; nothing to attach to.
    root: Single<Entity, With<HelpRoot>>,
) {
    let root = *root;
    // Headings keep the display voice (Dogica); everything readable is the
    // working voice (Departure Mono).
    let heading_font = assets.font.clone();
    let font = assets.font_body.clone();

    // The scrolling viewport: a fixed-height, clipped column that scrolls on Y.
    let viewport = commands
        .spawn((
            HelpScroll,
            Node {
                width: px(560.0),
                height: px(VIEWPORT_HEIGHT),
                flex_direction: FlexDirection::Column,
                align_items: AlignItems::Stretch,
                overflow: Overflow::scroll_y(),
                padding: UiRect::axes(px(20.0), px(12.0)),
                row_gap: px(6.0),
                ..default()
            },
            BackgroundColor(theme::GRID),
            ScrollPosition::default(),
        ))
        .id();

    // Navigation hint sits inside the viewport so it scrolls with the rest.
    spawn_body_label(
        &mut commands,
        viewport,
        &font,
        "Scroll: Up / Down  -  Page: PgUp / PgDn  -  Home / End",
        theme::TEXT_DIM,
        theme::MICRO_FONT_SIZE,
    );

    spawn_section_heading(&mut commands, viewport, &heading_font, "Controls");
    for action in GameAction::ALL {
        spawn_keybind_row(&mut commands, viewport, &font, &settings.keybinds, action);
    }

    spawn_section_heading(&mut commands, viewport, &heading_font, "How To Play");
    for (term, blurb) in HOW_TO_PLAY {
        spawn_term_row(&mut commands, viewport, &font, term, blurb);
    }

    commands.entity(root).add_child(viewport);
}

/// Keyboard-driven vertical scrolling of the help viewport. The layout system
/// clamps the offset to valid bounds, so we only push it in the right direction.
fn scroll_help(
    keys: Res<ButtonInput<KeyCode>>,
    mut scroll: Single<&mut ScrollPosition, With<HelpScroll>>,
) {
    let mut delta = 0.0;
    if keys.just_pressed(KeyCode::ArrowDown) || keys.just_pressed(KeyCode::KeyS) {
        delta += SCROLL_LINE_STEP;
    }
    if keys.just_pressed(KeyCode::ArrowUp) || keys.just_pressed(KeyCode::KeyW) {
        delta -= SCROLL_LINE_STEP;
    }
    if keys.just_pressed(KeyCode::PageDown) {
        delta += SCROLL_PAGE_STEP;
    }
    if keys.just_pressed(KeyCode::PageUp) {
        delta -= SCROLL_PAGE_STEP;
    }

    if keys.just_pressed(KeyCode::Home) {
        scroll.0.y = 0.0;
    } else if keys.just_pressed(KeyCode::End) {
        scroll.0.y = SCROLL_MAX;
    } else if delta != 0.0 {
        // Never let our own bookkeeping go negative; the layout system handles
        // the upper bound against the real content height.
        scroll.0.y = (scroll.0.y + delta).max(0.0);
    }
}

/// A bold-ish section heading (uses the accent color at button size).
fn spawn_section_heading(commands: &mut Commands, parent: Entity, font: &Handle<Font>, text: &str) {
    let child = commands
        .spawn((
            Text::new(text),
            TextFont {
                font: font.clone(),
                font_size: theme::BUTTON_FONT_SIZE,
                ..default()
            },
            TextColor(theme::ACCENT),
            Node {
                margin: UiRect::top(px(10.0)),
                ..default()
            },
        ))
        .id();
    commands.entity(parent).add_child(child);
}

/// A plain body line.
fn spawn_body_label(
    commands: &mut Commands,
    parent: Entity,
    font: &Handle<Font>,
    text: &str,
    color: Color,
    size: f32,
) {
    let child = commands
        .spawn((
            Text::new(text),
            TextFont {
                font: font.clone(),
                font_size: size,
                ..default()
            },
            TextColor(color),
        ))
        .id();
    commands.entity(parent).add_child(child);
}

/// One "Action . . . Key(s)" row.
fn spawn_keybind_row(
    commands: &mut Commands,
    parent: Entity,
    font: &Handle<Font>,
    binds: &Keybinds,
    action: GameAction,
) {
    let (primary, secondary) = binds.get(action);
    let keys = match secondary {
        Some(second) => format!("{} / {}", key_label(primary), key_label(second)),
        None => key_label(primary),
    };
    spawn_body_label(
        commands,
        parent,
        font,
        &format!("{:<12} {}", action.label(), keys),
        theme::TEXT,
        14.0,
    );
}

/// One "Term: explanation" row, with the term accented.
fn spawn_term_row(
    commands: &mut Commands,
    parent: Entity,
    font: &Handle<Font>,
    term: &str,
    blurb: &str,
) {
    let child = commands
        .spawn((
            Node {
                flex_direction: FlexDirection::Row,
                align_items: AlignItems::FlexStart,
                column_gap: px(6.0),
                ..default()
            },
            children![
                (
                    Text::new(format!("{term}:")),
                    TextFont {
                        font: font.clone(),
                        font_size: 14.0,
                        ..default()
                    },
                    TextColor(theme::ACCENT),
                ),
                (
                    Text::new(blurb),
                    TextFont {
                        font: font.clone(),
                        font_size: 14.0,
                        ..default()
                    },
                    TextColor(theme::TEXT),
                ),
            ],
        ))
        .id();
    commands.entity(parent).add_child(child);
}

/// Spec-terminology rules reference. Kept verbatim with the official terms so the
/// help text matches the language used elsewhere in the game.
const HOW_TO_PLAY: &[(&str, &str)] = &[
    (
        "Matrix",
        "The play field. Stack Minos and clear full rows before the stack tops out.",
    ),
    (
        "Tetrimino",
        "A falling piece of four Minos (I, O, T, S, Z, J, L).",
    ),
    ("Mino", "A single cell; four Minos make one Tetrimino."),
    (
        "Soft Drop",
        "Hold to make the Tetrimino fall faster, still under your control.",
    ),
    (
        "Hard Drop",
        "Instantly drop the Tetrimino to the bottom and Lock Down.",
    ),
    (
        "Lock Down",
        "A landed Tetrimino fixes to the Matrix after a short delay (or on Hard Drop).",
    ),
    (
        "Hold",
        "Stash the current Tetrimino for later; swap it back once per piece.",
    ),
    (
        "Ghost",
        "A preview outline showing where the Tetrimino lands if Hard Dropped.",
    ),
    (
        "T-Spin",
        "Rotate a T-Tetrimino into a tight slot for bonus line-clear points.",
    ),
    (
        "Back-to-Back",
        "Chaining special clears (Tetris / T-Spin) without a normal clear adds a bonus.",
    ),
];

/// A short, human-readable name for a [`KeyCode`].
///
/// Covers the keys reachable through the default [`Keybinds`] and common rebind
/// targets; anything else falls back to its `Debug` form (Bevy `KeyCode`s print
/// readably, e.g. `KeyQ`, `F5`).
fn key_label(key: KeyCode) -> String {
    let name = match key {
        KeyCode::ArrowLeft => "Left",
        KeyCode::ArrowRight => "Right",
        KeyCode::ArrowUp => "Up",
        KeyCode::ArrowDown => "Down",
        KeyCode::Space => "Space",
        KeyCode::Enter => "Enter",
        KeyCode::NumpadEnter => "Numpad Enter",
        KeyCode::Escape => "Esc",
        KeyCode::Tab => "Tab",
        KeyCode::Backspace => "Backspace",
        KeyCode::ShiftLeft => "Left Shift",
        KeyCode::ShiftRight => "Right Shift",
        KeyCode::ControlLeft => "Left Ctrl",
        KeyCode::ControlRight => "Right Ctrl",
        KeyCode::AltLeft => "Left Alt",
        KeyCode::AltRight => "Right Alt",
        other => return strip_key_prefix(&format!("{other:?}")),
    };
    name.to_string()
}

/// Turn a `KeyCode` debug string into something friendlier: `KeyX` -> `X`,
/// `Digit5` -> `5`, `Numpad7` -> `Numpad 7`. Other forms pass through unchanged.
fn strip_key_prefix(debug: &str) -> String {
    if let Some(rest) = debug.strip_prefix("Key") {
        return rest.to_string();
    }
    if let Some(rest) = debug.strip_prefix("Digit") {
        return rest.to_string();
    }
    if let Some(rest) = debug.strip_prefix("Numpad") {
        return format!("Numpad {rest}");
    }
    debug.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn key_label_names_common_keys() {
        assert_eq!(key_label(KeyCode::ArrowLeft), "Left");
        assert_eq!(key_label(KeyCode::Space), "Space");
        assert_eq!(key_label(KeyCode::ShiftLeft), "Left Shift");
        assert_eq!(key_label(KeyCode::Escape), "Esc");
    }

    #[test]
    fn key_label_strips_debug_prefixes() {
        assert_eq!(key_label(KeyCode::KeyX), "X");
        assert_eq!(key_label(KeyCode::KeyZ), "Z");
        assert_eq!(key_label(KeyCode::Digit5), "5");
    }

    #[test]
    fn default_binds_render_for_every_action() {
        let binds = Keybinds::default();
        for action in GameAction::ALL {
            let (primary, secondary) = binds.get(action);
            let rendered = match secondary {
                Some(second) => format!("{} / {}", key_label(primary), key_label(second)),
                None => key_label(primary),
            };
            assert!(!rendered.is_empty(), "{} rendered empty", action.label());
        }
        // Rotate CW carries the WASD-side secondary (Up / W).
        let (p, s) = binds.get(GameAction::RotateCw);
        assert_eq!(key_label(p), "Up");
        assert_eq!(s.map(key_label).as_deref(), Some("W"));
    }

    #[test]
    fn how_to_play_covers_required_spec_terms() {
        let required = [
            "Matrix",
            "Tetrimino",
            "Mino",
            "Lock Down",
            "Hard Drop",
            "Soft Drop",
            "Hold",
            "Ghost",
            "T-Spin",
            "Back-to-Back",
        ];
        for term in required {
            assert!(
                HOW_TO_PLAY.iter().any(|(t, _)| *t == term),
                "missing spec term: {term}"
            );
        }
    }
}
