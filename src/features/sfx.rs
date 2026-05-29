//! SFX / music feature (STUB — fan-out fills this file only).
//!
//! Goal: own background music and volume control. Read `music_volume` /
//! `sfx_volume` from [`GameSettings`](crate::settings::GameSettings) and apply
//! them (e.g. via Bevy audio sink volumes). NOTE: per-action sound effects are
//! already handled by [`SoundEffectsPlugin`](crate::level) reacting to
//! `AudioCue`s; this feature adds music playback and makes the existing SFX honor
//! `sfx_volume`. Coordinate volume application here rather than editing the
//! existing observer.
//!
//! Touch only this file.

use bevy::prelude::*;

/// Music playback + volume control. Currently a no-op stub.
pub struct SfxPlugin;

impl Plugin for SfxPlugin {
    fn build(&self, _app: &mut App) {}
}
