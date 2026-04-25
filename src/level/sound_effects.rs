use crate::assets::GameAssets;
use crate::level::common::{ActionEvent, LevelState, PlacingEvent};
use bevy::prelude::*;

pub struct SoundEffectsPlugin;

impl Plugin for SoundEffectsPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(
            Update,
            (placing_sounds, action_sounds).run_if(in_state(LevelState::Playing)),
        );
    }
}

fn placing_sounds(
    mut commands: Commands,
    mut ev_placing: MessageReader<PlacingEvent>,
    game_assets: Res<GameAssets>,
) {
    for ev in ev_placing.read() {
        match ev {
            PlacingEvent::Locked(0) => {
                commands.spawn((
                    AudioPlayer::new(game_assets.locked_sound.clone()),
                    PlaybackSettings::DESPAWN,
                ));
            }
            PlacingEvent::Locked(lines) => {
                let sound = match lines {
                    1 => game_assets.line_clear_1.clone(),
                    2 => game_assets.line_clear_2.clone(),
                    3 => game_assets.line_clear_3.clone(),
                    4 => game_assets.line_clear_4.clone(),
                    _ => unreachable!(),
                };
                commands.spawn((AudioPlayer::new(sound), PlaybackSettings::DESPAWN));
            }
            PlacingEvent::Placed => {
                commands.spawn((
                    AudioPlayer::new(game_assets.placed_sound.clone()),
                    PlaybackSettings::DESPAWN,
                ));
            }
        }
    }
}

fn action_sounds(
    mut commands: Commands,
    mut ev_action: MessageReader<ActionEvent>,
    game_assets: Res<GameAssets>,
) {
    for ev in ev_action.read() {
        match ev {
            ActionEvent::Hold => {
                commands.spawn((
                    AudioPlayer::new(game_assets.hold_sound.clone()),
                    PlaybackSettings::DESPAWN,
                ));
            }
            ActionEvent::Rotation(_, _, _, _) => {
                commands.spawn((
                    AudioPlayer::new(game_assets.rotation_sound.clone()),
                    PlaybackSettings::DESPAWN,
                ));
            }
            ActionEvent::HardDrop(_) => {
                commands.spawn((
                    AudioPlayer::new(game_assets.hard_drop_sound.clone()),
                    PlaybackSettings::DESPAWN,
                ));
            }
            _ => {}
        }
    }
}
