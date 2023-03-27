use bevy::prelude::*;
use crate::assets::GameAssets;
use crate::GameState;
use crate::level::LevelState;

pub(crate) struct GameOverPlugin;

impl Plugin for GameOverPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(
            (setup_game_over_ui, ).chain().in_schedule(OnEnter(GameState::GameOver))
        ).add_system(button_system)
            .add_systems(
                (cleanup_level, ).chain().in_schedule(OnExit(GameState::GameOver))
            );
        ;
    }
}

fn cleanup_level(mut commands: Commands, query: Query<Entity, With<GameOverCleanup>>) {
    for entity in query.iter() {
        commands.entity(entity).despawn_recursive();
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
    commands.spawn(Camera2dBundle::default()).insert(GameOverCleanup);
    commands
        .spawn((
            NodeBundle {
                style: Style {
                    size: Size::new(Val::Percent(100.0), Val::Percent(100.0)),
                    align_items: AlignItems::Center,
                    justify_content: JustifyContent::Center,
                    ..default()
                },
                ..default()
            }, GameOverCleanup
        ))
        .with_children(|parent| {
            parent
                .spawn(NodeBundle {
                    style: Style {
                        flex_direction: FlexDirection::Column,
                        align_items: AlignItems::Center,
                        ..default()
                    },
                    background_color: Color::ORANGE_RED.into(),
                    ..default()
                })
                .with_children(|parent| {
                    parent.spawn(
                        TextBundle::from_section(
                            "Game Over",
                            TextStyle {
                                font: game_assets.font.clone(),
                                font_size: 16.0,
                                color: Color::rgb(0.9, 0.9, 0.9),
                            },
                        )
                            .with_style(Style {
                                margin: UiRect::all(Val::Px(20.0)),
                                ..default()
                            }),
                    );

                    parent
                        .spawn((
                            ButtonBundle {
                                style: Style {
                                    size: Size::new(Val::Percent(70.), Val::Px(30.0)),
                                    margin: UiRect::all(Val::Px(10.0)),
                                    justify_content: JustifyContent::Center,
                                    align_items: AlignItems::Center,
                                    ..default()
                                },
                                background_color: Color::rgb(0.15, 0.15, 0.15).into(),
                                ..default()
                            },
                            MenuActions::Quit
                        ))
                        .with_children(|parent| {
                            parent.spawn(TextBundle::from_section(
                                "Quit to Menu",
                                TextStyle {
                                    font: game_assets.font.clone(),
                                    font_size: 12.0,
                                    color: Color::rgb(0.9, 0.9, 0.9),
                                },
                            ));
                        });

                    parent
                        .spawn((
                            ButtonBundle {
                                style: Style {
                                    size: Size::new(Val::Percent(70.), Val::Px(30.0)),
                                    margin: UiRect::all(Val::Px(10.0)),
                                    justify_content: JustifyContent::Center,
                                    align_items: AlignItems::Center,
                                    ..default()
                                },
                                background_color: Color::rgb(0.15, 0.15, 0.15).into(),
                                ..default()
                            },
                            MenuActions::Restart
                        ))
                        .with_children(|parent| {
                            parent.spawn(TextBundle::from_section(
                                "Restart",
                                TextStyle {
                                    font: game_assets.font.clone(),
                                    font_size: 12.0,
                                    color: Color::rgb(0.9, 0.9, 0.9),
                                },
                            ));
                        });
                });
        });
}


const NORMAL_BUTTON: Color = Color::rgb(0.15, 0.15, 0.15);
const HOVERED_BUTTON: Color = Color::rgb(0.25, 0.25, 0.25);
const PRESSED_BUTTON: Color = Color::rgb(0.35, 0.75, 0.35);

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
            Interaction::Clicked => {
                *color = PRESSED_BUTTON.into();
                match menu_actions {
                    MenuActions::Restart => {
                        next_game_state.set(GameState::InGame);
                        next_level_state.set(LevelState::Ready);
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

fn cleanup_game_over_menu(mut commands: Commands, query: Query<Entity, With<GameOverCleanup>>) {
    for entity in query.iter() {
        commands.entity(entity).despawn_recursive();
    }
}