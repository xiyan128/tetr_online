//! The game-over screen: its own camera, UI, and restart/quit navigation.
//!
//! Spawned on entering [`GameState::GameOver`](crate::GameState::GameOver).
//! Because the gameplay camera is torn down on leaving the session, this screen
//! brings its own [`Camera2d`] so its UI renders. Both mouse buttons and a
//! keyboard fallback (Enter/Space to restart, Esc to the main menu) drive the
//! state transition.

use crate::assets::GameAssets;
use crate::GameState;
use bevy::prelude::*;

pub(crate) struct GameOverPlugin;

impl Plugin for GameOverPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(OnEnter(GameState::GameOver), (setup_game_over_ui,).chain())
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

pub fn setup_game_over_ui(mut commands: Commands, game_assets: Res<GameAssets>) {
    // The gameplay camera is despawned on leaving `GameState::Playing`, so the
    // game-over screen needs its own camera or its UI renders to nothing.
    commands.spawn((Camera2d, DespawnOnExit(GameState::GameOver)));

    let font = TextFont {
        font: game_assets.font.clone(),
        font_size: 12.0,
        ..default()
    };
    let title_font = TextFont {
        font: game_assets.font.clone(),
        font_size: 16.0,
        ..default()
    };

    commands.spawn((
        Node {
            width: percent(100),
            height: percent(100),
            align_items: AlignItems::Center,
            justify_content: JustifyContent::Center,
            ..default()
        },
        DespawnOnExit(GameState::GameOver),
        children![(
            Node {
                flex_direction: FlexDirection::Column,
                align_items: AlignItems::Center,
                padding: UiRect::all(px(16)),
                row_gap: px(10),
                ..default()
            },
            BackgroundColor(Color::srgb(1.0, 0.27, 0.0)),
            children![
                (
                    Text::new("Game Over"),
                    title_font,
                    TextColor(Color::srgb(0.9, 0.9, 0.9)),
                    Node {
                        margin: UiRect::all(px(10)),
                        ..default()
                    },
                ),
                game_over_button("Menu", MenuActions::Quit, font.clone()),
                game_over_button("Restart", MenuActions::Restart, font),
            ]
        )],
    ));
}

const NORMAL_BUTTON: Color = Color::srgb(0.15, 0.15, 0.15);
const HOVERED_BUTTON: Color = Color::srgb(0.25, 0.25, 0.25);
const PRESSED_BUTTON: Color = Color::srgb(0.35, 0.75, 0.35);

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
            width: percent(70),
            height: px(30),
            margin: UiRect::all(px(10)),
            justify_content: JustifyContent::Center,
            align_items: AlignItems::Center,
            ..default()
        },
        BackgroundColor(NORMAL_BUTTON),
        action,
        children![(
            Text::new(label),
            font,
            TextColor(Color::srgb(0.9, 0.9, 0.9)),
        )],
    )
}

fn button_system(
    mut interaction_query: ButtonInteractionQuery,
    mut next_game_state: ResMut<NextState<GameState>>,
) {
    for (interaction, mut color, menu_actions) in &mut interaction_query {
        match *interaction {
            Interaction::Pressed => {
                *color = PRESSED_BUTTON.into();
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
                *color = HOVERED_BUTTON.into();
            }
            Interaction::None => {
                *color = NORMAL_BUTTON.into();
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
