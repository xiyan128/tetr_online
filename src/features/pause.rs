//! Pause feature (STUB — fan-out fills this file only).
//!
//! Goal: while [`GameState::Playing`](crate::GameState::Playing), pressing the
//! Pause action toggles to [`GameState::Paused`](crate::GameState::Paused) and
//! back. On `Paused`: freeze the engine driver (the level plugin already gates
//! its systems on `Playing`, so simply being in `Paused` halts the sim — keep it
//! that way), draw a "Paused" overlay (use [`crate::ui::widgets`]), and offer
//! Resume / Quit-to-menu. Read the Pause keybind from
//! [`GameSettings`](crate::settings::GameSettings).
//!
//! Touch only this file.

use bevy::prelude::*;

/// Pause overlay + state toggle. Currently a no-op stub.
pub struct PausePlugin;

impl Plugin for PausePlugin {
    fn build(&self, _app: &mut App) {}
}
