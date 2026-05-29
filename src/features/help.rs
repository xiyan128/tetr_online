//! Help feature (STUB — fan-out fills this file only).
//!
//! Goal: render the controls/about content on
//! [`GameState::Help`](crate::GameState::Help), attaching it under the
//! [`HelpRoot`](crate::screens::HelpRoot) the screen shell spawns. List the
//! current bindings from [`GameSettings`](crate::settings::GameSettings)'s
//! [`Keybinds`](crate::settings::Keybinds) and a short how-to-play blurb. Reuse
//! [`crate::ui::widgets`] for the look.
//!
//! Touch only this file.

use bevy::prelude::*;

/// Help-screen content. Currently a no-op stub.
pub struct HelpPlugin;

impl Plugin for HelpPlugin {
    fn build(&self, _app: &mut App) {}
}
