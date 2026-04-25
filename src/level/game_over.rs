use crate::assets::GameAssets;
use crate::level::common::LevelState;
use crate::GameState;
use bevy::prelude::*;

pub(crate) struct GameOverPlugin;

impl Plugin for GameOverPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(OnEnter(GameState::GameOver), (setup_game_over_ui,).chain())
            .add_systems(Update, button_system)
            .add_systems(OnExit(GameState::GameOver), (cleanup_level,).chain());
    }
}

fn cleanup_level(mut commands: Commands, query: Query<Entity, With<GameOverCleanup>>) {
    for entity in query.iter() {
        commands.entity(entity).despawn();
    }
}

#[derive(Component)]
enum MenuActions {
    Restart,
    Quit,
}

#[derive(Component)]
struct GameOverCleanup;

pub fn setup_game_over_ui(mut commands: Commands, game_assets: Res<GameAssets>) {
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
        GameOverCleanup,
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
    mut interaction_query: Query<
        (&Interaction, &mut BackgroundColor, &MenuActions),
        (Changed<Interaction>, With<Button>),
    >,
    mut next_game_state: ResMut<NextState<GameState>>,
    mut next_level_state: ResMut<NextState<LevelState>>,
) {
    for (interaction, mut color, menu_actions) in &mut interaction_query {
        match *interaction {
            Interaction::Pressed => {
                *color = PRESSED_BUTTON.into();
                match menu_actions {
                    MenuActions::Restart => {
                        next_game_state.set(GameState::InGame);
                        next_level_state.set(LevelState::Setup);
                    }
                    MenuActions::Quit => {
                        next_game_state.set(GameState::MainMenu);
                    }
                };
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
