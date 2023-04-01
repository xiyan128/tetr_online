use crate::assets::GameAssets;
use crate::level::common::{ActionEvent, LevelState, PlacingEvent};
use crate::GameState;
use bevy::prelude::*;
use crate::level::score::ScoreType;

pub struct SoundEffectsPlugin;

impl Plugin for SoundEffectsPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems((placing_sounds, action_sounds).in_set(OnUpdate(LevelState::Playing)));
    }
}

fn placing_sounds(
    mut ev_placing: EventReader<PlacingEvent>,
    game_assets: Res<GameAssets>,
    audio: Res<Audio>,
) {
    for ev in ev_placing.iter() {
        match ev {
            PlacingEvent::Placed => {
                audio.play(game_assets.placed_sound.clone());
            }
            PlacingEvent::Locked(0) => {
                audio.play(game_assets.locked_sound.clone());
            }
            PlacingEvent::Locked(lines) => {
                match lines  {
                    1 => {
                        audio.play(game_assets.line_clear_1.clone());
                    }
                    2 => {
                        audio.play(game_assets.line_clear_2.clone());
                    }
                    3 => {
                        audio.play(game_assets.line_clear_3.clone());
                    }
                    4 => {
                        audio.play(game_assets.line_clear_4.clone());
                    }
                    _ => unreachable!()
                }
            }
        }
    }
}

fn action_sounds(
    mut ev_action: EventReader<ActionEvent>,
    game_assets: Res<GameAssets>,
    audio: Res<Audio>,
) {
    for ev in ev_action.iter() {
        match ev {
            ActionEvent::Hold => {
                audio.play(game_assets.hold_sound.clone());
            }
            ActionEvent::Rotation(_, _, _, _) => {
                audio.play(game_assets.rotation_sound.clone());
            }
            ActionEvent::HardDrop(_) => {
                audio.play(game_assets.hard_drop_sound.clone());
            }
            _ => {}
        }
    }
}