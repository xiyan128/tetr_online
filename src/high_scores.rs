//! High-score model: the shared tables every screen and the session read.
//!
//! [`HighScores`] holds a top-10 leaderboard per [`Variant`]. The *primary* sort
//! key is variant-specific (Sprint ranks by elapsed time ascending — fastest
//! first; Marathon/Ultra rank by score descending — highest first), driven by
//! the variant's [`ScoreKind`]. The model is in-memory here; the high-scores
//! feature + storage layer handle persistence (serialize/deserialize through the
//! [`Storage`](crate::storage::Storage) trait) later.
//!
//! Both the high-scores screen (display) and the high-scores feature (insert on
//! game-over) code against these types.

use bevy::prelude::*;

use crate::variant::{ScoreKind, Variant};

/// Maximum entries retained per variant.
pub const MAX_ENTRIES_PER_VARIANT: usize = 10;

/// A single completed-run result.
///
/// Both `score` and `time_seconds` are always recorded; which one is the
/// *primary* ranking key is decided by the variant's [`ScoreKind`]. `lines` and
/// `level` are kept for display.
#[derive(Debug, Clone, Copy, PartialEq, Reflect)]
pub struct HighScore {
    pub score: usize,
    pub time_seconds: f32,
    pub lines: usize,
    pub level: u8,
}

impl HighScore {
    /// Compare `self` against `other` for `kind`, returning `true` when `self`
    /// ranks strictly *better* (should sort earlier). Sprint: lower time wins;
    /// Score modes: higher score wins.
    fn is_better_than(&self, other: &HighScore, kind: ScoreKind) -> bool {
        match kind {
            ScoreKind::Time => self.time_seconds < other.time_seconds,
            ScoreKind::Score => self.score > other.score,
        }
    }
}

/// Per-variant top-10 boards: one slot per [`Variant`], indexed by its position in
/// [`Variant::ALL`]. Keying by index — rather than a named field plus a `match`
/// arm per variant — means adding a mode is a single edit to `Variant`, not a
/// shotgun edit across this file.
#[derive(Resource, Debug, Clone, Reflect)]
#[reflect(Resource)]
pub struct HighScores {
    tables: Vec<Vec<HighScore>>,
}

impl Default for HighScores {
    fn default() -> Self {
        // One empty board per variant; slots line up with `Variant::ALL`.
        Self {
            tables: vec![Vec::new(); Variant::ALL.len()],
        }
    }
}

impl HighScores {
    /// `variant`'s slot index (its position in [`Variant::ALL`]).
    fn slot(variant: Variant) -> usize {
        Variant::ALL
            .iter()
            .position(|v| *v == variant)
            .expect("Variant::ALL contains every variant")
    }

    /// The (sorted, best-first) table for `variant`.
    pub fn table(&self, variant: Variant) -> &[HighScore] {
        &self.tables[Self::slot(variant)]
    }

    fn table_mut(&mut self, variant: Variant) -> &mut Vec<HighScore> {
        &mut self.tables[Self::slot(variant)]
    }

    /// Whether `candidate` would make `variant`'s top-10 (the board isn't full,
    /// or `candidate` ranks better than the current worst entry).
    pub fn qualifies(&self, variant: Variant, candidate: &HighScore) -> bool {
        let kind = variant.def().score_kind;
        let table = self.table(variant);
        if table.len() < MAX_ENTRIES_PER_VARIANT {
            return true;
        }
        // Table is full and kept sorted best-first, so the last entry is the
        // worst; qualify if the candidate beats it.
        table
            .last()
            .is_some_and(|worst| candidate.is_better_than(worst, kind))
    }

    /// Insert `candidate` into `variant`'s board, keeping it sorted best-first
    /// and truncated to the top 10. Returns the candidate's 0-based rank if it
    /// landed on the board, or `None` if it did not qualify.
    pub fn insert(&mut self, variant: Variant, candidate: HighScore) -> Option<usize> {
        if !self.qualifies(variant, &candidate) {
            return None;
        }
        let kind = variant.def().score_kind;
        let table = self.table_mut(variant);
        // Find the first existing entry the candidate beats; insert before it.
        let position = table
            .iter()
            .position(|existing| candidate.is_better_than(existing, kind))
            .unwrap_or(table.len());
        table.insert(position, candidate);
        table.truncate(MAX_ENTRIES_PER_VARIANT);
        Some(position)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(score: usize, time: f32) -> HighScore {
        HighScore {
            score,
            time_seconds: time,
            lines: 0,
            level: 1,
        }
    }

    #[test]
    fn empty_board_qualifies_anything() {
        let boards = HighScores::default();
        assert!(boards.qualifies(Variant::Marathon, &run(0, 0.0)));
    }

    #[test]
    fn marathon_ranks_by_score_descending() {
        let mut boards = HighScores::default();
        boards.insert(Variant::Marathon, run(100, 50.0));
        boards.insert(Variant::Marathon, run(300, 90.0));
        boards.insert(Variant::Marathon, run(200, 10.0));

        let scores: Vec<usize> = boards
            .table(Variant::Marathon)
            .iter()
            .map(|s| s.score)
            .collect();
        assert_eq!(scores, vec![300, 200, 100]);
    }

    #[test]
    fn sprint_ranks_by_time_ascending() {
        let mut boards = HighScores::default();
        boards.insert(Variant::Sprint, run(0, 60.0));
        boards.insert(Variant::Sprint, run(0, 30.0));
        boards.insert(Variant::Sprint, run(0, 45.0));

        let times: Vec<f32> = boards
            .table(Variant::Sprint)
            .iter()
            .map(|s| s.time_seconds)
            .collect();
        assert_eq!(times, vec![30.0, 45.0, 60.0]);
    }

    #[test]
    fn board_keeps_only_top_ten_and_reports_rank() {
        let mut boards = HighScores::default();
        for i in 0..MAX_ENTRIES_PER_VARIANT {
            // Scores 10, 20, ... 100.
            boards.insert(Variant::Ultra, run((i + 1) * 10, 0.0));
        }
        assert_eq!(boards.table(Variant::Ultra).len(), MAX_ENTRIES_PER_VARIANT);

        // A new best lands at rank 0 and evicts the worst.
        assert_eq!(boards.insert(Variant::Ultra, run(999, 0.0)), Some(0));
        assert_eq!(boards.table(Variant::Ultra).len(), MAX_ENTRIES_PER_VARIANT);
        assert_eq!(boards.table(Variant::Ultra)[0].score, 999);

        // A score below the (new) worst does not qualify.
        assert!(!boards.qualifies(Variant::Ultra, &run(5, 0.0)));
        assert_eq!(boards.insert(Variant::Ultra, run(5, 0.0)), None);
    }

    #[test]
    fn each_variant_addresses_an_independent_board() {
        // Guards the index mapping: every variant has its own slot (default has one
        // empty board per variant) and inserting into one never aliases another.
        let mut boards = HighScores::default();
        for variant in Variant::ALL {
            assert!(boards.table(variant).is_empty());
        }
        boards.insert(Variant::Marathon, run(100, 0.0));
        assert_eq!(boards.table(Variant::Marathon).len(), 1);
        assert!(boards.table(Variant::Sprint).is_empty());
        assert!(boards.table(Variant::Ultra).is_empty());
    }
}
