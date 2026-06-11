//! SFX / music feature.
//!
//! Owns audio *volume* and background music. Two responsibilities:
//!
//! 1. **SFX volume** — the per-action sound effects are spawned by
//!    [`SoundEffectsPlugin`](crate::level)'s `AudioCue` observer with a hardcoded
//!    `PlaybackSettings::DESPAWN` (i.e. full volume). We do **not** edit that
//!    observer. Instead, [`apply_volume_to_new_sinks`] watches for freshly
//!    inserted [`AudioSink`]s and scales each one to the configured volume:
//!    [`GameSettings::sfx_volume`] for effects, [`GameSettings::music_volume`]
//!    for the looping music track. Bevy inserts the sink one frame after the
//!    `AudioPlayer` entity is spawned (via its `play_queued_audio_system`), so
//!    catching `Added<AudioSink>` reliably re-volumes every cue regardless of
//!    where it originated.
//!
//! 2. **Background music** — [`start_music`] spawns a single looping track on
//!    entering [`GameState::Playing`], tagged [`MusicTrack`] and despawned on
//!    exit. Because no music asset ships in `assets/` yet, the track only spawns
//!    when [`MusicAsset`] holds a handle; otherwise it warns once.

use bevy::audio::Volume;
use bevy::prelude::*;

use crate::settings::GameSettings;
use crate::{DespawnOnExit, GameState};

/// Marks the looping background-music entity so its sink is volumed against
/// [`GameSettings::music_volume`] instead of `sfx_volume`, and so live volume
/// changes can target it specifically.
#[derive(Component, Reflect)]
#[reflect(Component)]
struct MusicTrack;

/// Optional handle to a looping background-music asset.
///
/// No music file ships in `assets/` yet, so this defaults to `None` and music
/// playback stays dormant. Once an asset exists, loading it into
/// [`GameAssets`](crate::assets::GameAssets) and populating this resource lights up
/// the music path with zero further changes here.
#[derive(Resource, Default, Reflect)]
#[reflect(Resource)]
struct MusicAsset(Option<Handle<AudioSource>>);

/// Music playback + volume control.
pub struct SfxPlugin;

impl Plugin for SfxPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<MusicAsset>()
            // Inspector/scene registration for this feature's resource + marker.
            .register_type::<MusicAsset>()
            .register_type::<MusicTrack>()
            // Re-volume every newly spawned sink (SFX cues + music) to honor the
            // current settings. Runs every frame; the `Added` filter keeps it to
            // only the sinks Bevy inserted this frame.
            .add_systems(Update, apply_volume_to_new_sinks)
            // Keep the looping music track in sync when settings change live
            // (one-shot SFX are already gone by the time a slider moves, so they
            // only need the spawn-time pass above).
            .add_systems(
                Update,
                update_music_volume.run_if(resource_changed::<GameSettings>),
            )
            // Background music spans the whole play session.
            .add_systems(OnEnter(GameState::Session), start_music);
    }
}

/// Scale freshly inserted [`AudioSink`]s to the configured volume.
///
/// Bevy's `play_queued_audio_system` inserts the sink (already multiplied by the
/// global volume, which we leave at its 1.0 default) one frame after the
/// `AudioPlayer` entity appears. We override that here so the existing SFX
/// observer — which we must not edit — still respects `sfx_volume`, and so the
/// music track respects `music_volume`.
fn apply_volume_to_new_sinks(
    settings: Res<GameSettings>,
    mut sinks: Query<(&mut AudioSink, Has<MusicTrack>), Added<AudioSink>>,
) {
    for (mut sink, is_music) in &mut sinks {
        let volume = if is_music {
            settings.music_volume
        } else {
            settings.sfx_volume
        };
        sink.set_volume(Volume::Linear(volume));
    }
}

/// Re-apply [`GameSettings::music_volume`] to the live looping track when the
/// settings resource changes (e.g. the options slider moves mid-game).
fn update_music_volume(
    settings: Res<GameSettings>,
    mut music: Query<&mut AudioSink, With<MusicTrack>>,
) {
    for mut sink in &mut music {
        sink.set_volume(Volume::Linear(settings.music_volume));
    }
}

/// Spawn the looping background-music track for the play session.
///
/// No-ops with a one-time warning while no music asset is configured (none ships
/// yet). The spawned entity is tagged [`MusicTrack`] so
/// [`apply_volume_to_new_sinks`] volumes it against `music_volume`, and carries
/// `DespawnOnExit(Playing)` so it stops when the session ends.
fn start_music(mut commands: Commands, music_asset: Res<MusicAsset>, mut warned: Local<bool>) {
    let Some(handle) = music_asset.0.clone() else {
        if !*warned {
            warn!(
                "no background-music asset configured; skipping music playback \
                 (add a looping track to GameAssets + MusicAsset to enable)"
            );
            *warned = true;
        }
        return;
    };

    commands.spawn((
        AudioPlayer::new(handle),
        // Loop the track; the initial volume is overwritten by
        // `apply_volume_to_new_sinks` once the sink is inserted.
        PlaybackSettings::LOOP,
        MusicTrack,
        DespawnOnExit(GameState::Session),
    ));
}
