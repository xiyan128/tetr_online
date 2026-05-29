//! Notifications feature (STUB — fan-out fills this file only).
//!
//! Goal: a lightweight transient-message system. Define a message/event (e.g.
//! `Notification { text, ttl }`), a writer other systems can use, and a renderer
//! that shows toasts and fades them out. Reuse [`crate::ui::widgets`] theme for
//! consistent styling. Likely shown during
//! [`GameState::Playing`](crate::GameState::Playing) (e.g. "New high score!",
//! level-up) but keep the API general.
//!
//! Touch only this file.

use bevy::prelude::*;

/// Transient on-screen notifications. Currently a no-op stub.
pub struct NotificationsPlugin;

impl Plugin for NotificationsPlugin {
    fn build(&self, _app: &mut App) {}
}
