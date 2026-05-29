//! High-scores feature (STUB — fan-out fills this file only).
//!
//! Goal: two halves, both in this file.
//! 1. **Record**: on entering [`GameState::GameOver`](crate::GameState::GameOver),
//!    build a [`HighScore`](crate::high_scores::HighScore) from the final
//!    [`LatestSnapshot`](crate::level::engine_bridge::LatestSnapshot) (`score`,
//!    `lines`, `level`) + [`VariantProgress::elapsed_seconds`] and try
//!    [`HighScores::insert`] for the [`ActiveVariant`]. Persist the table via
//!    [`StorageResource`](crate::storage::StorageResource) under
//!    [`storage::keys::HIGH_SCORES`](crate::storage::keys::HIGH_SCORES); load on
//!    startup.
//! 2. **Display**: on [`GameState::HighScores`](crate::GameState::HighScores),
//!    render the per-variant tables under
//!    [`HighScoresRoot`](crate::screens::HighScoresRoot), formatting the primary
//!    column per [`ScoreKind`](crate::variant::ScoreKind) (Sprint time asc,
//!    others score desc). Reuse [`crate::ui::widgets`].
//!
//! [`HighScores::insert`]: crate::high_scores::HighScores::insert
//! [`ActiveVariant`]: crate::variant::ActiveVariant
//! [`VariantProgress::elapsed_seconds`]: crate::variant::VariantProgress
//!
//! Touch only this file.

use bevy::prelude::*;

/// Records runs into [`HighScores`](crate::high_scores::HighScores) and renders
/// the leaderboard. Currently a no-op stub.
pub struct HighScoresFeaturePlugin;

impl Plugin for HighScoresFeaturePlugin {
    fn build(&self, _app: &mut App) {}
}
