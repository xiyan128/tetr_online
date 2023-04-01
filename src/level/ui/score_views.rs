use crate::assets::GameAssets;
use crate::level::common::{to_translation, LevelCleanup, LevelConfig};
use crate::level::score::{Scorer, ScoreType};
use crate::level::ui::calc_ui_offset;
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
        .spawn(Text2dBundle {
            text: Text::from_section(
                make_score_text(&Scorer::default()),
                TextStyle {
                    font: game_assets.font.clone(),
                    font_size: 14.0,
                    color: Color::WHITE,
                },
            ),
            transform: Transform::from_translation(
                to_translation(config.board_width as isize, 1, config.block_size) + offset,
            ),
            text_anchor: Anchor::TopRight,
            ..Default::default()
        })
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
        .spawn(Text2dBundle {
            text: Text::from_section(
                make_line_count_text(&Scorer::default()),
                TextStyle {
                    font: game_assets.font.clone(),
                    font_size: 14.0,
                    color: Color::WHITE,
                },
            ),
            transform: Transform::from_translation(
                to_translation(config.board_width as isize, 2, config.block_size) + offset,
            ),
            text_anchor: Anchor::TopRight,
            ..Default::default()
        })
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
        .spawn(Text2dBundle {
            text: Text {
                alignment: TextAlignment::Center,
                ..Default::default()
            },
            transform: Transform::from_translation(
                to_translation(0, (1 + config.board_height >> 1) as isize, config.block_size) + offset,
            ),
            text_anchor: Anchor::TopLeft,
            ..Default::default()
        })
        .insert(ScoreTypeText)
        .insert(LevelCleanup);
}

pub fn update_score_text(mut text_query: Query<&mut Text, With<ScoreText>>, scorer: Res<Scorer>) {
    let mut text = text_query.single_mut();
    text.sections[0].value = make_score_text(&scorer);
}

pub fn update_line_count_text(
    mut text_query: Query<&mut Text, With<LineCountText>>,
    scorer: Res<Scorer>,
) {
    let mut text = text_query.single_mut();
    text.sections[0].value = make_line_count_text(&scorer);
}

pub fn update_score_type_text(mut text_query: Query<&mut Text, With<ScoreTypeText>>,
                              mut ev_score_type: EventReader<Vec<ScoreType>>,
                              game_assets: Res<GameAssets>,
) {
    let mut text = text_query.single_mut();
    for ev in ev_score_type.iter() {
        text.sections = ev.iter().map(
            |score_type| TextSection {
                value: format!("{:?}\n\n", score_type),
                style: TextStyle {
                    font: game_assets.font.clone(),
                    font_size: 16.0,
                    color: Color::WHITE,
                },
            }
        ).collect();
    }
}

pub fn fade_out_score_type_text(
    mut text_query: Query<&mut Text, With<ScoreTypeText>>,
    time: Res<Time>,
) {
    let mut text = text_query.single_mut();
    text.sections.iter_mut().for_each(|section| {
        section.style.color.set_a(section.style.color.a() - time.delta_seconds());
    });
}