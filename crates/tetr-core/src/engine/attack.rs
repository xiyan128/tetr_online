//! Versus attack table: garbage lines sent per clear.
//!
//! A **pure** function from a scored clear + chain state to the number of garbage
//! lines it sends — the primitive the versus / attack benchmark measures, and the
//! shared rule a future versus mode and an attack-rewarding evaluator will reuse.
//! No engine state is touched (ADR-7): the caller supplies the Back-to-Back flag,
//! the current combo count, and whether the lock perfect-cleared the board, all of
//! which it can read from the engine's events/snapshot.
//!
//! Values follow the modern guideline versus table (TETR.IO-compatible base lines):
//! Single 0, Double 1, Triple 2, Tetris 4; T-Spin Mini Single 0 / Double 1; T-Spin
//! Single 2 / Double 4 / Triple 6; +1 line for a Back-to-Back clear; a combo bonus
//! from [`COMBO_TABLE`]; and `+`[`PERFECT_CLEAR_ATTACK`] for an all-clear. These are
//! the numbers that discriminate offensive efficiency (APP / attack-per-piece),
//! the tractable proxy for "beats Cold Clear 2" before a full garbage-exchange
//! versus harness exists.

use super::scoring::EngineScoreAction;
use super::t_spin::TSpinKind;

/// Guideline combo bonus: extra garbage lines from an N-combo (consecutive
/// line-clearing placements), indexed by the combo count BEFORE this clear (0 on
/// the first clear of a chain). Saturates at the last entry.
pub const COMBO_TABLE: [u32; 13] = [0, 0, 1, 1, 1, 2, 2, 3, 3, 4, 4, 4, 5];

/// Garbage lines a perfect clear (all-clear) sends, on top of the clear's lines.
pub const PERFECT_CLEAR_ATTACK: u32 = 10;

/// Garbage lines sent by a single locked placement.
///
/// - `action`: the scored clear (already classifies T-spin kind + line count); the
///   value carried on [`EngineEvent::ScoreAwarded`](super::EngineEvent).
/// - `back_to_back`: whether this clear is awarded the Back-to-Back bonus — pass the
///   engine's `back_to_back_bonus` flag from the same `ScoreAwarded` event, so the
///   eligibility rule stays the engine's single source of truth.
/// - `combo`: the combo count BEFORE this clear (number of immediately preceding
///   consecutive line-clearing placements; `0` for the first clear of a chain).
/// - `perfect_clear`: whether the board is empty after this lock.
///
/// Non-clearing actions (no clear, soft/hard drop, a spin that cleared no lines)
/// send `0`.
pub fn attack_lines(
    action: EngineScoreAction,
    back_to_back: bool,
    combo: u32,
    perfect_clear: bool,
) -> u32 {
    let (t_spin, lines) = match action {
        EngineScoreAction::Single => (None, 1usize),
        EngineScoreAction::Double => (None, 2),
        EngineScoreAction::Triple => (None, 3),
        EngineScoreAction::Tetris => (None, 4),
        EngineScoreAction::TSpin { kind, lines } => (Some(kind), lines),
        EngineScoreAction::NoClear
        | EngineScoreAction::SoftDrop
        | EngineScoreAction::HardDrop { .. } => return 0,
    };
    // A spin that cleared no lines sends nothing (and does not extend a combo).
    if lines == 0 {
        return 0;
    }

    let base = match (t_spin, lines) {
        (None, 1) => 0, // Single
        (None, 2) => 1, // Double
        (None, 3) => 2, // Triple
        (None, 4) => 4, // Tetris
        (Some(TSpinKind::Mini), 1) => 0,
        (Some(TSpinKind::Mini), 2) => 1,
        (Some(TSpinKind::Full), 1) => 2, // T-Spin Single
        (Some(TSpinKind::Full), 2) => 4, // T-Spin Double
        (Some(TSpinKind::Full), 3) => 6, // T-Spin Triple
        _ => 0,
    };

    let b2b_bonus = u32::from(back_to_back);
    let combo_bonus = COMBO_TABLE[(combo as usize).min(COMBO_TABLE.len() - 1)];
    let pc_bonus = if perfect_clear {
        PERFECT_CLEAR_ATTACK
    } else {
        0
    };

    base + b2b_bonus + combo_bonus + pc_bonus
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tspin(kind: TSpinKind, lines: usize) -> EngineScoreAction {
        EngineScoreAction::TSpin { kind, lines }
    }

    #[test]
    fn base_table_matches_guideline() {
        assert_eq!(attack_lines(EngineScoreAction::Single, false, 0, false), 0);
        assert_eq!(attack_lines(EngineScoreAction::Double, false, 0, false), 1);
        assert_eq!(attack_lines(EngineScoreAction::Triple, false, 0, false), 2);
        assert_eq!(attack_lines(EngineScoreAction::Tetris, false, 0, false), 4);
        assert_eq!(attack_lines(tspin(TSpinKind::Full, 1), false, 0, false), 2);
        assert_eq!(attack_lines(tspin(TSpinKind::Full, 2), false, 0, false), 4);
        assert_eq!(attack_lines(tspin(TSpinKind::Full, 3), false, 0, false), 6);
        assert_eq!(attack_lines(tspin(TSpinKind::Mini, 1), false, 0, false), 0);
        assert_eq!(attack_lines(tspin(TSpinKind::Mini, 2), false, 0, false), 1);
    }

    #[test]
    fn back_to_back_adds_one() {
        assert_eq!(attack_lines(EngineScoreAction::Tetris, true, 0, false), 5);
        assert_eq!(attack_lines(tspin(TSpinKind::Full, 2), true, 0, false), 5);
    }

    #[test]
    fn combo_bonus_from_table_and_saturates() {
        // combo index 5 -> +2 on a Tetris.
        assert_eq!(attack_lines(EngineScoreAction::Tetris, false, 5, false), 6);
        // saturates past the table length.
        assert_eq!(
            attack_lines(EngineScoreAction::Tetris, false, 999, false),
            4 + COMBO_TABLE[COMBO_TABLE.len() - 1]
        );
    }

    #[test]
    fn perfect_clear_stacks_on_top() {
        // Tetris + B2B + perfect clear = 4 + 1 + 10.
        assert_eq!(attack_lines(EngineScoreAction::Tetris, true, 0, true), 15);
    }

    #[test]
    fn non_clears_send_nothing() {
        assert_eq!(attack_lines(EngineScoreAction::NoClear, false, 0, false), 0);
        assert_eq!(
            attack_lines(EngineScoreAction::SoftDrop, false, 0, false),
            0
        );
        assert_eq!(
            attack_lines(EngineScoreAction::HardDrop { cells: 5 }, false, 0, false),
            0
        );
        // A spin that cleared no lines.
        assert_eq!(attack_lines(tspin(TSpinKind::Full, 0), false, 0, false), 0);
    }
}
