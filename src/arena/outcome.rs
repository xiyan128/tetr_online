//! What one game produced — the trustworthy measurement record.
//!
//! Headline totals (lines, score, level) are read from the engine's authoritative
//! snapshot; the per-clear *breakdown* ([`ClearCounts`]) is tallied from the event
//! stream. [`crate::arena::play`] reconciles the two, so these numbers cannot
//! silently disagree with the engine.

use crate::engine::{EngineScoreAction, TSpinKind};

/// Why a game ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Termination {
    /// The engine declared game over (block-out or lock-out).
    ToppedOut,
    /// The controller placed the setup's full piece budget without topping out.
    ReachedPieceBudget,
    /// The hard frame cap was reached first. A healthy run never ends this way;
    /// it means the controller stopped making progress, surfaced for debugging.
    HitFrameCap,
}

/// Line clears tallied by type over one game.
///
/// Counts *clearing placements*, one per clear, classified by the engine's own
/// scoring action. Non-clearing placements (soft/hard drop, a lock that cleared
/// nothing, a zero-line spin) are not counted here — they place a piece without
/// clearing a line.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ClearCounts {
    pub single: u32,
    pub double: u32,
    pub triple: u32,
    pub tetris: u32,
    /// Mini T-spin that cleared a line (the "mini" reward bucket; a zero-line mini
    /// is not a clear and is not counted).
    pub tspin_mini: u32,
    pub tspin_single: u32,
    pub tspin_double: u32,
    pub tspin_triple: u32,
}

impl ClearCounts {
    /// Fold one scored action into the tally. Only line-clearing actions count;
    /// everything else (drops, no-clear, zero-line spins) is ignored.
    pub(crate) fn record(&mut self, action: &EngineScoreAction) {
        match action {
            EngineScoreAction::Single => self.single += 1,
            EngineScoreAction::Double => self.double += 1,
            EngineScoreAction::Triple => self.triple += 1,
            EngineScoreAction::Tetris => self.tetris += 1,
            EngineScoreAction::TSpin {
                kind: TSpinKind::Mini,
                lines,
            } if *lines >= 1 => self.tspin_mini += 1,
            EngineScoreAction::TSpin {
                kind: TSpinKind::Full,
                lines: 1,
            } => self.tspin_single += 1,
            EngineScoreAction::TSpin {
                kind: TSpinKind::Full,
                lines: 2,
            } => self.tspin_double += 1,
            EngineScoreAction::TSpin {
                kind: TSpinKind::Full,
                lines: 3,
            } => self.tspin_triple += 1,
            // Zero-line spins, soft/hard drop, no-clear: placed a piece, cleared
            // no line.
            _ => {}
        }
    }

    /// Every bucket as an array, in declaration order. The single source of truth
    /// for "what counts as a clear" — [`total`](Self::total) folds over this, so a
    /// new bucket added here is automatically included.
    pub fn buckets(&self) -> [u32; 8] {
        [
            self.single,
            self.double,
            self.triple,
            self.tetris,
            self.tspin_mini,
            self.tspin_single,
            self.tspin_double,
            self.tspin_triple,
        ]
    }

    /// Total clearing placements (the sum of every bucket).
    pub fn total(&self) -> u32 {
        self.buckets().iter().sum()
    }
}

/// The complete, deterministic result of one game.
///
/// `lines_cleared`, `final_score`, and `final_level` come from the engine's final
/// snapshot (authoritative); `pieces_placed`, `clears`, and `back_to_back_awards`
/// are tallied from the event stream and reconciled against that snapshot in
/// [`crate::arena::play`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GameOutcome {
    /// The engine seed this game was played with (reproducibility handle).
    pub seed: u64,
    /// Pieces locked (placed) over the game.
    pub pieces_placed: u32,
    /// Total lines cleared (engine truth).
    pub lines_cleared: u32,
    /// Clears broken down by type.
    pub clears: ClearCounts,
    /// Number of Back-to-Back bonus awards (B2B chain sustains).
    pub back_to_back_awards: u32,
    /// Final score (engine truth).
    pub final_score: u32,
    /// Final level reached (engine truth).
    pub final_level: u8,
    /// Frames simulated.
    pub frames: u32,
    /// Why the game ended.
    pub termination: Termination,
}

impl GameOutcome {
    /// Whether the game ended by topping out (vs. reaching the piece budget).
    pub fn topped_out(&self) -> bool {
        self.termination == Termination::ToppedOut
    }

    /// Lines cleared per piece placed — stacking efficiency. `0.0` if no pieces
    /// were placed.
    pub fn lines_per_piece(&self) -> f64 {
        if self.pieces_placed == 0 {
            0.0
        } else {
            f64::from(self.lines_cleared) / f64::from(self.pieces_placed)
        }
    }

    /// Fraction of clears that were Tetrises — a coarse signal of downstacking
    /// vs. line-by-line play. `0.0` if there were no clears.
    pub fn tetris_rate(&self) -> f64 {
        let clears = self.clears.total();
        if clears == 0 {
            0.0
        } else {
            f64::from(self.clears.tetris) / f64::from(clears)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn record_maps_each_action_to_its_bucket() {
        let mut c = ClearCounts::default();
        c.record(&EngineScoreAction::Single);
        c.record(&EngineScoreAction::Double);
        c.record(&EngineScoreAction::Double);
        c.record(&EngineScoreAction::Triple);
        c.record(&EngineScoreAction::Tetris);
        c.record(&EngineScoreAction::TSpin {
            kind: TSpinKind::Mini,
            lines: 1,
        });
        c.record(&EngineScoreAction::TSpin {
            kind: TSpinKind::Full,
            lines: 2,
        });

        assert_eq!(c.single, 1);
        assert_eq!(c.double, 2);
        assert_eq!(c.triple, 1);
        assert_eq!(c.tetris, 1);
        assert_eq!(c.tspin_mini, 1);
        assert_eq!(c.tspin_double, 1);
        assert_eq!(c.total(), 7);
    }

    #[test]
    fn non_clearing_actions_are_ignored() {
        let mut c = ClearCounts::default();
        c.record(&EngineScoreAction::NoClear);
        c.record(&EngineScoreAction::SoftDrop);
        c.record(&EngineScoreAction::HardDrop { cells: 18 });
        // A zero-line spin places a piece but clears nothing.
        c.record(&EngineScoreAction::TSpin {
            kind: TSpinKind::Mini,
            lines: 0,
        });
        c.record(&EngineScoreAction::TSpin {
            kind: TSpinKind::Full,
            lines: 0,
        });

        assert_eq!(c, ClearCounts::default());
        assert_eq!(c.total(), 0);
    }

    #[test]
    fn derived_metrics() {
        let outcome = GameOutcome {
            seed: 1,
            pieces_placed: 100,
            lines_cleared: 40,
            clears: ClearCounts {
                tetris: 8,
                single: 2,
                ..Default::default()
            },
            back_to_back_awards: 7,
            final_score: 12_000,
            final_level: 5,
            frames: 3_000,
            termination: Termination::ReachedPieceBudget,
        };

        assert!((outcome.lines_per_piece() - 0.4).abs() < 1e-9);
        // 8 tetrises out of 10 total clears.
        assert!((outcome.tetris_rate() - 0.8).abs() < 1e-9);
        assert!(!outcome.topped_out());
    }
}
