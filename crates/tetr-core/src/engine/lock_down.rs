//! Lock-down timer reset policy.
//!
//! When a grounded piece is moved or rotated, whether its lock timer resets
//! depends on [`LockDownMode`]: Extended Placement allows a bounded number of
//! resets ([`EXTENDED_LOCK_RESET_BUDGET`]) per new lowest row, Infinite always
//! resets, and Classic never does. [`apply_grounded_move_or_rotation`] applies
//! the policy and reports whether the timer was reset.

use crate::engine::active_piece::ActivePiece;

pub const LOCK_DOWN_SECONDS: f32 = 0.5;
pub const EXTENDED_LOCK_RESET_BUDGET: u8 = 15;

#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub enum LockDownMode {
    #[default]
    Extended,
    Infinite,
    Classic,
}

pub fn apply_grounded_move_or_rotation(
    active_piece: &mut ActivePiece,
    mode: LockDownMode,
    lock_seconds: f32,
) -> bool {
    match mode {
        LockDownMode::Extended => {
            if active_piece.grounded_move_rotate_count_since_lowest() >= EXTENDED_LOCK_RESET_BUDGET
            {
                return false;
            }

            active_piece.record_grounded_move_or_rotate();
            active_piece.reset_lock_timer(lock_seconds);
            true
        }
        LockDownMode::Infinite => {
            active_piece.reset_lock_timer(lock_seconds);
            true
        }
        LockDownMode::Classic => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::PieceType;

    #[test]
    fn extended_resets_until_budget_is_exhausted() {
        let mut active_piece = ActivePiece::new(PieceType::T, (3, 19));

        for expected_count in 1..=EXTENDED_LOCK_RESET_BUDGET {
            assert!(apply_grounded_move_or_rotation(
                &mut active_piece,
                LockDownMode::Extended,
                LOCK_DOWN_SECONDS,
            ));
            assert_eq!(
                active_piece.grounded_move_rotate_count_since_lowest(),
                expected_count
            );
            assert_eq!(active_piece.lock_timer_seconds(), LOCK_DOWN_SECONDS);
        }

        assert!(!apply_grounded_move_or_rotation(
            &mut active_piece,
            LockDownMode::Extended,
            LOCK_DOWN_SECONDS,
        ));
        assert_eq!(
            active_piece.grounded_move_rotate_count_since_lowest(),
            EXTENDED_LOCK_RESET_BUDGET
        );
    }

    #[test]
    fn falling_below_previous_lowest_reopens_extended_budget() {
        let mut active_piece = ActivePiece::new(PieceType::T, (3, 19));

        for _ in 0..EXTENDED_LOCK_RESET_BUDGET {
            assert!(apply_grounded_move_or_rotation(
                &mut active_piece,
                LockDownMode::Extended,
                LOCK_DOWN_SECONDS,
            ));
        }
        active_piece.move_to((3, 18), crate::engine::PieceAction::Fall);

        assert_eq!(active_piece.grounded_move_rotate_count_since_lowest(), 0);
        assert!(apply_grounded_move_or_rotation(
            &mut active_piece,
            LockDownMode::Extended,
            LOCK_DOWN_SECONDS,
        ));
    }

    #[test]
    fn infinite_resets_without_budget() {
        let mut active_piece = ActivePiece::new(PieceType::T, (3, 19));

        for _ in 0..=EXTENDED_LOCK_RESET_BUDGET {
            assert!(apply_grounded_move_or_rotation(
                &mut active_piece,
                LockDownMode::Infinite,
                LOCK_DOWN_SECONDS,
            ));
        }

        assert_eq!(active_piece.grounded_move_rotate_count_since_lowest(), 0);
        assert_eq!(active_piece.lock_timer_seconds(), LOCK_DOWN_SECONDS);
    }

    #[test]
    fn classic_does_not_reset_on_grounded_move_or_rotation() {
        let mut active_piece = ActivePiece::new(PieceType::T, (3, 19));

        assert!(!apply_grounded_move_or_rotation(
            &mut active_piece,
            LockDownMode::Classic,
            LOCK_DOWN_SECONDS,
        ));
        assert_eq!(active_piece.lock_timer_seconds(), 0.0);
    }
}
