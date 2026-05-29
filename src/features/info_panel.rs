//! Info-panel feature (STUB — fan-out fills this file only).
//!
//! Goal: an in-game side panel shown while
//! [`GameState::Playing`](crate::GameState::Playing) that reflects the active
//! [`Variant`](crate::variant::Variant). Read [`ActiveVariant`] +
//! [`VariantProgress`] and the engine [`LatestSnapshot`] to display the relevant
//! figure(s): Marathon -> level/lines/score; Sprint -> lines remaining + elapsed
//! time; Ultra -> time remaining + score. Use the variant's
//! [`VariantDef`](crate::variant::VariantDef) (`line_target`,
//! `time_limit_seconds`, `score_kind`) to label things.
//!
//! [`ActiveVariant`]: crate::variant::ActiveVariant
//! [`VariantProgress`]: crate::variant::VariantProgress
//! [`LatestSnapshot`]: crate::level::engine_bridge::LatestSnapshot
//!
//! Touch only this file.

use bevy::prelude::*;

/// In-game variant info panel. Currently a no-op stub.
pub struct InfoPanelPlugin;

impl Plugin for InfoPanelPlugin {
    fn build(&self, _app: &mut App) {}
}
