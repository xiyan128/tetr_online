//! Options feature (STUB — fan-out fills this file only).
//!
//! Goal: build the interactive Options UI on
//! [`GameState::Options`](crate::GameState::Options), attaching widgets under the
//! [`OptionsRoot`](crate::screens::OptionsRoot) the screen shell spawns. Let the
//! player edit [`GameSettings`](crate::settings::GameSettings): `next_count`
//! (1..=6), `hold_enabled`, `ghost_enabled`, `lock_down_mode`, `music_volume`,
//! `sfx_volume`, and the [`Keybinds`](crate::settings::Keybinds). Call
//! `GameSettings::sanitize` after edits. Persist via
//! [`StorageResource`](crate::storage::StorageResource) under
//! [`storage::keys::SETTINGS`](crate::storage::keys::SETTINGS) (choose a string
//! encoding); load it back on startup. Reuse [`crate::ui::widgets`] +
//! [`crate::ui::focus`] for the look + navigation.
//!
//! Touch only this file.

use bevy::prelude::*;

/// Options-screen settings editor. Currently a no-op stub.
pub struct OptionsPlugin;

impl Plugin for OptionsPlugin {
    fn build(&self, _app: &mut App) {}
}
