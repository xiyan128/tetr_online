//! Top-out detection: the three guideline lose conditions.
//!
//! - **Block out** — a freshly spawned piece overlaps an existing block.
//! - **Lock out** — a piece locks entirely above the visible skyline.
//! - **Top out** — a piece is forced above the total (including hidden) board.
//!
//! Each is a pure predicate so the engine can check them at the relevant moment
//! without owning the game-over policy.

use crate::engine::board::Board;
use crate::engine::pieces::Piece;

pub fn is_block_out(piece: &Piece, board: &Board, spawn_origin: (isize, isize)) -> bool {
    piece.collide_with(board, spawn_origin)
}

pub fn is_lock_out(piece: &Piece, origin: (isize, isize), visible_height: usize) -> bool {
    piece
        .board()
        .cell_coords()
        .into_iter()
        .all(|(_, y)| y + origin.1 >= visible_height as isize)
}

pub fn is_top_out(
    cells: impl IntoIterator<Item = (isize, isize)>,
    total_board_height: usize,
) -> bool {
    cells
        .into_iter()
        .any(|(_, y)| y >= total_board_height as isize)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::board::CellKind;
    use crate::engine::pieces::PieceType;

    #[test]
    fn block_out_detects_spawn_overlap_before_immediate_drop() {
        let mut board = Board::with_top_margin(10, 20, 20);
        let piece = Piece::from(PieceType::T);
        let spawn_origin = piece.spawn_coords(10, 20);
        assert!(board.set(4, 20, CellKind::Some(PieceType::O)));

        assert!(is_block_out(&piece, &board, spawn_origin));
    }

    #[test]
    fn lock_out_requires_every_mino_above_skyline() {
        let piece = Piece::from(PieceType::T);

        assert!(is_lock_out(&piece, (3, 20), 20));
        assert!(!is_lock_out(&piece, (3, 18), 20));
    }

    #[test]
    fn locking_partly_above_skyline_is_allowed() {
        let piece = Piece::from(PieceType::L);

        assert!(!is_lock_out(&piece, (3, 18), 20));
    }

    #[test]
    fn top_out_detects_cells_forced_above_total_board_height() {
        assert!(!is_top_out([(0, 39)], 40));
        assert!(is_top_out([(0, 40)], 40));
    }
}
