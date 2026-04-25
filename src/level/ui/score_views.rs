use crate::assets::GameAssets;
use crate::level::common::{to_translation, LevelCleanup, LevelConfig};
use crate::level::score::{ScoreTypes, Scorer};
use crate::level::ui::calc_ui_offset;
use bevy::color::Alpha;
use bevy::prelude::*;
use bevy::sprite::Anchor;

#[derive(Component)]
pub struct ScoreText;

#[derive(Component)]
pub struct LineCountText;

#[derive(Component)]
pub struct ScoreTypeText;

fn make_line_count_text(scorer: &Scorer) -> String {
    format!("LINES: {}", scorer.lines)
}

fn make_score_text(scorer: &Scorer) -> String {
    format!("SCORE: {}", scorer.score)
}

pub fn spawn_score_text(
    mut commands: Commands,
    config: Res<LevelConfig>,
    game_assets: Res<GameAssets>,
) {
    let offset = Vec3::new(calc_ui_offset(&config), 0., 0.);

    commands
        .spawn((
            Text2d::new(make_score_text(&Scorer::default())),
            TextFont {
                font: game_assets.font.clone(),
                font_size: 14.0,
                ..default()
            },
            TextColor(Color::WHITE),
            Transform::from_translation(
                to_translation(config.board_width as isize, 1, config.block_size) + offset,
            ),
            Anchor::TOP_LEFT,
        ))
        .insert(ScoreText)
        .insert(LevelCleanup);
}

pub fn spawn_line_count_text(
    mut commands: Commands,
    config: Res<LevelConfig>,
    game_assets: Res<GameAssets>,
) {
    let offset = Vec3::new(calc_ui_offset(&config), 0., 0.);

    commands
        .spawn((
            Text2d::new(make_line_count_text(&Scorer::default())),
            TextFont {
                font: game_assets.font.clone(),
                font_size: 14.0,
                ..default()
            },
            TextColor(Color::WHITE),
            Transform::from_translation(
                to_translation(config.board_width as isize, 2, config.block_size) + offset,
            ),
            Anchor::TOP_LEFT,
        ))
        .insert(LineCountText)
        .insert(LevelCleanup);
}

pub fn spawn_score_type_text(
    mut commands: Commands,
    config: Res<LevelConfig>,
    game_assets: Res<GameAssets>,
) {
    let offset = -Vec3::new(calc_ui_offset(&config), 0., 0.);

    commands
        .spawn((
            Text2d::new(""),
            TextFont {
                font: game_assets.font.clone(),
                font_size: 16.0,
                ..default()
            },
            TextColor(Color::WHITE),
            TextLayout::new_with_justify(Justify::Center),
            Transform::from_translation(
                to_translation(
                    0,
                    (1 + config.board_height >> 1) as isize,
                    config.block_size,
                ) + offset,
            ),
            Anchor::TOP_RIGHT,
        ))
        .insert(ScoreTypeText)
        .insert(LevelCleanup);
}

pub fn update_score_text(mut text_query: Query<&mut Text2d, With<ScoreText>>, scorer: Res<Scorer>) {
    let Ok(mut text) = text_query.single_mut() else {
        return;
    };
    text.0 = make_score_text(&scorer);
}

pub fn update_line_count_text(
    mut text_query: Query<&mut Text2d, With<LineCountText>>,
    scorer: Res<Scorer>,
) {
    let Ok(mut text) = text_query.single_mut() else {
        return;
    };
    text.0 = make_line_count_text(&scorer);
}

pub fn update_score_type_text(
    mut text_query: Query<(&mut Text2d, &mut TextColor), With<ScoreTypeText>>,
    mut ev_score_type: MessageReader<ScoreTypes>,
) {
    let Ok((mut text, mut color)) = text_query.single_mut() else {
        return;
    };
    for ev in ev_score_type.read() {
        text.0 =
            ev.0.iter()
                .map(|score_type| format!("{score_type:?}"))
                .collect::<Vec<_>>()
                .join("\n\n");
        color.0 = Color::WHITE;
    }
}

pub fn fade_out_score_type_text(
    mut text_query: Query<&mut TextColor, With<ScoreTypeText>>,
    time: Res<Time>,
) {
    let Ok(mut color) = text_query.single_mut() else {
        return;
    };
    let alpha = color.0.alpha() - time.delta_secs();
    color.0 = color.0.with_alpha(alpha.max(0.0));
}
