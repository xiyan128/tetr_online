//! T-spin detection and Mini/Full classification.
//!
//! A T-spin requires a T piece whose last successful action was a rotation (or
//! the kick-5-into-slot exception) and at least three of its four diagonal
//! corners blocked. [`t_spin_corners`] reports the corners relative to the
//! piece's facing (front pair `a`/`b`, back pair `c`/`d`); [`classify_t_spin`]
//! turns that into [`TSpinKind`] per the guideline, including the SRS kick-5
//! override that promotes a Mini to Full.

use crate::engine::active_piece::{ActivePiece, PieceAction};
use crate::engine::board::{Board, CellKind};
use crate::engine::pieces::{PieceRotation, PieceType};

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum TSpinKind {
    Mini,
    Full,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct TSpinCorners {
    pub a: bool,
    pub b: bool,
    pub c: bool,
    pub d: bool,
}

impl TSpinCorners {
    fn blocked_count(&self) -> usize {
        [self.a, self.b, self.c, self.d]
            .into_iter()
            .filter(|blocked| *blocked)
            .count()
    }
}

pub fn t_spin_corners(active_piece: &ActivePiece, board: &Board) -> Option<TSpinCorners> {
    if active_piece.piece_type() != PieceType::T {
        return None;
    }

    let center = t_center(active_piece.origin());
    let corner = |offset: (isize, isize)| {
        let (x, y) = (center.0 + offset.0, center.1 + offset.1);
        is_corner_blocked(board, x, y)
    };

    let nw = corner((-1, 1));
    let ne = corner((1, 1));
    let sw = corner((-1, -1));
    let se = corner((1, -1));

    Some(match active_piece.rotation() {
        PieceRotation::R0 => TSpinCorners {
            a: nw,
            b: ne,
            c: sw,
            d: se,
        },
        PieceRotation::R90 => TSpinCorners {
            a: ne,
            b: se,
            c: nw,
            d: sw,
        },
        PieceRotation::R180 => TSpinCorners {
            a: se,
            b: sw,
            c: ne,
            d: nw,
        },
        PieceRotation::R270 => TSpinCorners {
            a: sw,
            b: nw,
            c: se,
            d: ne,
        },
    })
}

pub fn classify_t_spin(active_piece: &ActivePiece, board: &Board) -> Option<TSpinKind> {
    if active_piece.piece_type() != PieceType::T {
        return None;
    }

    if active_piece.last_successful_action() != PieceAction::Rotate
        && !active_piece.used_kick_5_into_t_slot()
    {
        return None;
    }

    let corners = t_spin_corners(active_piece, board).expect("T piece corners should exist");
    if corners.blocked_count() < 3 {
        return None;
    }

    let full_by_corners = corners.a && corners.b && (corners.c || corners.d);
    let full_by_kick_5 = active_piece.used_kick_5_into_t_slot()
        || active_piece.last_rotation_kick_number() == Some(5);
    if full_by_corners || full_by_kick_5 {
        return Some(TSpinKind::Full);
    }

    if corners.c && corners.d && (corners.a || corners.b) {
        Some(TSpinKind::Mini)
    } else {
        None
    }
}

fn t_center(origin: (isize, isize)) -> (isize, isize) {
    (origin.0 + 1, origin.1 + 1)
}

fn is_corner_blocked(board: &Board, x: isize, y: isize) -> bool {
    !matches!(board.get_cell_kind(x, y), CellKind::None)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::active_piece::RotationDirection;

    const ORIGIN: (isize, isize) = (4, 4);

    fn rotated_t(
        rotation: PieceRotation,
        kick_number: u8,
        entered_t_slot_with_kick_5: bool,
    ) -> ActivePiece {
        let mut active_piece = ActivePiece::new(PieceType::T, ORIGIN);
        active_piece.rotate_to(
            rotation,
            ORIGIN,
            RotationDirection::Clockwise,
            kick_number,
            entered_t_slot_with_kick_5,
        );
        active_piece
    }

    fn board_with_blocked_corners(corners: &[(isize, isize)]) -> Board {
        let mut board = Board::new(10, 20);
        let center = t_center(ORIGIN);
        for (x_offset, y_offset) in corners {
            assert!(board.set(
                center.0 + x_offset,
                center.1 + y_offset,
                CellKind::Some(PieceType::O)
            ));
        }
        board
    }

    #[test]
    fn non_t_piece_is_not_t_spin() {
        let mut active_piece = ActivePiece::new(PieceType::L, ORIGIN);
        active_piece.rotate_to(
            PieceRotation::R90,
            ORIGIN,
            RotationDirection::Clockwise,
            1,
            false,
        );
        let board = board_with_blocked_corners(&[(-1, 1), (1, 1), (-1, -1)]);

        assert_eq!(classify_t_spin(&active_piece, &board), None);
        assert_eq!(t_spin_corners(&active_piece, &board), None);
    }

    #[test]
    fn t_piece_without_prior_rotation_is_not_t_spin() {
        let active_piece = ActivePiece::new(PieceType::T, ORIGIN);
        let board = board_with_blocked_corners(&[(-1, 1), (1, 1), (-1, -1)]);

        assert_eq!(classify_t_spin(&active_piece, &board), None);
    }

    #[test]
    fn less_than_three_blocked_corners_is_not_t_slot() {
        let active_piece = rotated_t(PieceRotation::R0, 1, false);
        let board = board_with_blocked_corners(&[(-1, 1), (1, 1)]);

        assert_eq!(classify_t_spin(&active_piece, &board), None);
    }

    #[test]
    fn front_corners_blocked_classifies_full_t_spin() {
        let active_piece = rotated_t(PieceRotation::R0, 1, false);
        let board = board_with_blocked_corners(&[(-1, 1), (1, 1), (-1, -1)]);

        assert_eq!(
            classify_t_spin(&active_piece, &board),
            Some(TSpinKind::Full)
        );
    }

    #[test]
    fn facing_controls_which_corners_are_front() {
        let active_piece = rotated_t(PieceRotation::R90, 1, false);
        let board = board_with_blocked_corners(&[(1, 1), (1, -1), (-1, 1)]);

        assert_eq!(
            t_spin_corners(&active_piece, &board),
            Some(TSpinCorners {
                a: true,
                b: true,
                c: true,
                d: false,
            })
        );
        assert_eq!(
            classify_t_spin(&active_piece, &board),
            Some(TSpinKind::Full)
        );
    }

    #[test]
    fn back_corner_pattern_classifies_mini_t_spin() {
        let active_piece = rotated_t(PieceRotation::R0, 1, false);
        let board = board_with_blocked_corners(&[(-1, -1), (1, -1), (-1, 1)]);

        assert_eq!(
            classify_t_spin(&active_piece, &board),
            Some(TSpinKind::Mini)
        );
    }

    #[test]
    fn kick_five_overrides_mini_pattern_to_full() {
        let active_piece = rotated_t(PieceRotation::R0, 5, true);
        let board = board_with_blocked_corners(&[(-1, -1), (1, -1), (-1, 1)]);

        assert_eq!(
            classify_t_spin(&active_piece, &board),
            Some(TSpinKind::Full)
        );
    }

    #[test]
    fn kick_five_exception_survives_later_non_rotation_action() {
        let mut active_piece = rotated_t(PieceRotation::R0, 5, true);
        active_piece.move_to(ORIGIN, PieceAction::Move);
        let board = board_with_blocked_corners(&[(-1, -1), (1, -1), (-1, 1)]);

        assert_eq!(active_piece.last_rotation_kick_number(), None);
        assert_eq!(
            classify_t_spin(&active_piece, &board),
            Some(TSpinKind::Full)
        );
    }
}
