use crate::assets::GameAssets;
use crate::level::common::AudioCue;
use bevy::prelude::*;

pub struct SoundEffectsPlugin;

impl Plugin for SoundEffectsPlugin {
    fn build(&self, app: &mut App) {
        app.add_observer(play_audio_cue);
    }
}

fn play_audio_cue(cue: On<AudioCue>, mut commands: Commands, game_assets: Res<GameAssets>) {
    let sound = match cue.event() {
        AudioCue::Rotation => game_assets.rotation_sound.clone(),
        AudioCue::HardDrop => game_assets.hard_drop_sound.clone(),
        AudioCue::Hold => game_assets.hold_sound.clone(),
        AudioCue::Placed => game_assets.placed_sound.clone(),
        AudioCue::Locked(0) => game_assets.locked_sound.clone(),
        AudioCue::Locked(1) => game_assets.line_clear_1.clone(),
        AudioCue::Locked(2) => game_assets.line_clear_2.clone(),
        AudioCue::Locked(3) => game_assets.line_clear_3.clone(),
        AudioCue::Locked(4) => game_assets.line_clear_4.clone(),
        AudioCue::Locked(lines) => {
            warn!("ignoring invalid line clear audio cue: {lines}");
            return;
        }
    };

    commands.spawn((AudioPlayer::new(sound), PlaybackSettings::DESPAWN));
}
