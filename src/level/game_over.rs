//! The game-over screen: its own camera, UI, and restart/quit navigation.
//!
//! Spawned on entering [`GameState::GameOver`](crate::GameState::GameOver).
//! Because the gameplay camera is torn down on leaving the session, this screen
//! brings its own [`Camera2d`] so its UI renders. Both mouse buttons and a
//! keyboard fallback (Enter/Space to restart, Esc to the main menu) drive the
//! state transition.
//!
//! Styling reuses [`crate::ui::theme`] so the screen sits in the same dark
//! palette as the menus: a centered card over a dimmed backdrop, a muted-red
//! "GAME OVER" heading, a key hint, and themed buttons — instead of a flat red
//! box.

use crate::assets::GameAssets;
use crate::ui::theme;
use crate::GameState;
use bevy::prelude::*;

pub(crate) struct GameOverPlugin;

impl Plugin for GameOverPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(OnEnter(GameState::GameOver), setup_game_over_ui)
            .add_systems(
                Update,
                (button_system, keyboard_nav).run_if(in_state(GameState::GameOver)),
            );
    }
}

#[derive(Component)]
enum MenuActions {
    Restart,
    Quit,
}

/// Muted red for the heading — signals "game over" without the old full-red fill.
const GAME_OVER_RED: Color = Color::srgb(0.88, 0.33, 0.34);
/// The card's dark panel and its subtle red-tinted border.
const CARD_BG: Color = Color::srgb(0.11, 0.11, 0.13);
const CARD_BORDER: Color = Color::srgb(0.45, 0.22, 0.24);

pub fn setup_game_over_ui(mut commands: Commands, game_assets: Res<GameAssets>) {
    // The gameplay camera is despawned on leaving `GameState::Playing`, so the
    // game-over screen needs its own camera or its UI renders to nothing.
    commands.spawn((Camera2d, DespawnOnExit(GameState::GameOver)));

    let title_font = TextFont {
        font: game_assets.font.clone(),
        font_size: 26.0,
        ..default()
    };
    let hint_font = TextFont {
        font: game_assets.font.clone(),
        font_size: 11.0,
        ..default()
    };
    let button_font = TextFont {
        font: game_assets.font.clone(),
        font_size: 14.0,
        ..default()
    };

    commands.spawn((
        // Full-window dimmed backdrop, centering the card.
        Node {
            width: percent(100),
            height: percent(100),
            align_items: AlignItems::Center,
            justify_content: JustifyContent::Center,
            ..default()
        },
        BackgroundColor(Color::srgba(0.0, 0.0, 0.0, 0.55)),
        DespawnOnExit(GameState::GameOver),
        children![(
            // The card itself: dark panel, soft border, rounded corners.
            Node {
                flex_direction: FlexDirection::Column,
                align_items: AlignItems::Center,
                padding: UiRect::axes(px(36), px(28)),
                row_gap: px(8),
                border: UiRect::all(px(2)),
                border_radius: BorderRadius::all(px(12)),
                ..default()
            },
            BackgroundColor(CARD_BG),
            BorderColor::all(CARD_BORDER),
            children![
                (Text::new("GAME OVER"), title_font, TextColor(GAME_OVER_RED)),
                (
                    Text::new("Enter to retry      Esc for menu"),
                    hint_font,
                    TextColor(theme::TEXT_DIM),
                    // A touch of breathing room above the buttons.
                    Node {
                        margin: UiRect::new(px(0), px(0), px(2), px(14)),
                        ..default()
                    },
                ),
                game_over_button("Retry", MenuActions::Restart, button_font.clone()),
                game_over_button("Menu", MenuActions::Quit, button_font),
            ]
        )],
    ));
}

type ButtonInteractionQuery<'w, 's> = Query<
    'w,
    's,
    (
        &'static Interaction,
        &'static mut BackgroundColor,
        &'static MenuActions,
    ),
    (Changed<Interaction>, With<Button>),
>;

fn game_over_button(label: &'static str, action: MenuActions, font: TextFont) -> impl Bundle {
    (
        Button,
        Node {
            width: px(200),
            height: px(38),
            margin: UiRect::all(px(4)),
            justify_content: JustifyContent::Center,
            align_items: AlignItems::Center,
            border_radius: BorderRadius::all(px(6)),
            ..default()
        },
        BackgroundColor(theme::BUTTON_NORMAL),
        action,
        children![(Text::new(label), font, TextColor(theme::TEXT))],
    )
}

fn button_system(
    mut interaction_query: ButtonInteractionQuery,
    mut next_game_state: ResMut<NextState<GameState>>,
) {
    for (interaction, mut color, menu_actions) in &mut interaction_query {
        match *interaction {
            Interaction::Pressed => {
                *color = theme::BUTTON_PRESSED.into();
                match menu_actions {
                    MenuActions::Restart => {
                        next_game_state.set(GameState::Playing);
                    }
                    MenuActions::Quit => {
                        next_game_state.set(GameState::MainMenu);
                    }
                }
            }
            Interaction::Hovered => {
                *color = theme::BUTTON_FOCUSED.into();
            }
            Interaction::None => {
                *color = theme::BUTTON_NORMAL.into();
            }
        }
    }
}

/// Keyboard fallback so the game-over screen behaves like every other screen:
/// Enter/Space restarts the run, Esc returns to the main menu. (The buttons
/// above are mouse-driven; this makes the screen keyboard-navigable.)
fn keyboard_nav(
    keys: Res<ButtonInput<KeyCode>>,
    mut next_game_state: ResMut<NextState<GameState>>,
) {
    if keys.just_pressed(KeyCode::Escape) {
        next_game_state.set(GameState::MainMenu);
    } else if keys.just_pressed(KeyCode::Enter)
        || keys.just_pressed(KeyCode::NumpadEnter)
        || keys.just_pressed(KeyCode::Space)
    {
        next_game_state.set(GameState::Playing);
    }
}
