//! Locking a piece and clearing completed rows.
//!
//! [`lock_and_clear`] writes the active piece's cells onto the board, clears any
//! now-full rows, and reports what happened via [`LockOutcome`]. It is a pure,
//! board-shaped operation (no engine/tick state) so search bots, replay
//! validators, and garbage solvers can reuse it (ADR-7, roadmap §10).

use crate::engine::active_piece::ActivePiece;
use crate::engine::bit_board::{full_rows, highest_occupied_y};
use crate::engine::board::{Board, CellKind};

/// Result of locking a piece onto the board and clearing any resulting full
/// rows.
///
/// The data captured here is intentionally board-shaped — no engine state, no
/// per-tick state — so that callers from future placement APIs (search bots,
/// replay validators, garbage solvers) can reuse the same primitive without
/// dragging the entire `Engine` along.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LockOutcome {
    /// Cells written to the board by the lock, in piece-cell order
    /// (pre-clear coordinates).
    pub cells_locked: Vec<(isize, isize, CellKind)>,
    /// Row indices (in pre-clear coordinates) that were full and cleared.
    /// Always sorted ascending.
    pub cleared_rows: Vec<isize>,
    /// Highest y still occupied after the lock+clear, or `None` if the board
    /// is empty.
    pub top_y_after_lock: Option<isize>,
}

/// Lock the given piece into the board and clear any resulting full rows.
///
/// Free function rather than a method on `Engine` so the same code path can be
/// reused by a future placement / search API (ADR-7, roadmap §10). The
/// function never reads "current tick" or input state — it only consults the
/// piece and the board.
pub fn lock_and_clear(active: &ActivePiece, board: &mut Board) -> LockOutcome {
    let piece = active.piece();
    let origin = active.origin();
    let piece_type = active.piece_type();

    let mut cells_locked = Vec::with_capacity(4);
    for (cx, cy) in piece.cells() {
        let x = cx + origin.0;
        let y = cy + origin.1;
        let cell_kind = CellKind::Some(piece_type);
        board.set(x, y, cell_kind);
        cells_locked.push((x, y, cell_kind));
    }

    // Full-row detection and the post-lock skyline both come from the column
    // bitboard: one cheap scan plus O(width) bit ops, rather than repeatedly
    // materialising `board.cells()` (a whole-board scan that allocates a `Vec` of
    // every occupied cell). The lock path is the search's hottest mutation — it runs
    // once per candidate placement — so this is a large constant-factor win, and it
    // is exactly equivalent: a row is full iff every column's bit is set there, and
    // the skyline is the highest set bit across columns.
    let cols = board.column_bits();
    let cleared_rows = full_rows(&cols);
    let top_y_after_lock = if cleared_rows.is_empty() {
        highest_occupied_y(&cols)
    } else {
        board.clear_lines();
        highest_occupied_y(&board.column_bits())
    };

    LockOutcome {
        cells_locked,
        cleared_rows,
        top_y_after_lock,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::pieces::PieceType;

    #[test]
    fn lock_and_clear_writes_piece_cells_to_board() {
        let mut board = Board::new(10, 20);
        let active = ActivePiece::new(PieceType::T, (3, 0));

        let outcome = lock_and_clear(&active, &mut board);

        assert_eq!(outcome.cells_locked.len(), 4);
        for (x, y, kind) in &outcome.cells_locked {
            assert_eq!(board.get_cell_kind(*x, *y), *kind);
        }
        assert!(outcome.cleared_rows.is_empty());
        // T at R0 has cells [(0,1),(1,1),(1,2),(2,1)]; at origin (3,0) the
        // top-most occupied row is y = 2.
        assert_eq!(outcome.top_y_after_lock, Some(2));
    }

    #[test]
    fn lock_and_clear_returns_full_row_indices_and_clears_them() {
        let mut board = Board::new(4, 4);
        // Fill row 0 except column 0; the I piece (horizontal at (0,-2))
        // sits on row 0 across columns 0..4 — completing the row.
        for x in 1..4 {
            assert!(board.set(x, 0, CellKind::Some(PieceType::O)));
        }
        let active = ActivePiece::new(PieceType::I, (0, -2));

        let outcome = lock_and_clear(&active, &mut board);

        assert_eq!(outcome.cleared_rows, vec![0]);
        assert!(outcome.top_y_after_lock.is_none());
        assert_eq!(board.cells().len(), 0);
    }

    #[test]
    fn lock_and_clear_reports_top_y_after_partial_clear() {
        let mut board = Board::new(4, 4);
        // Place a stack at row 1, column 2 — survives any line clear at row 0.
        assert!(board.set(2, 1, CellKind::Some(PieceType::O)));
        // Set up row 0 to clear when the I piece lands across it.
        for x in 1..4 {
            assert!(board.set(x, 0, CellKind::Some(PieceType::O)));
        }
        let active = ActivePiece::new(PieceType::I, (0, -2));

        let outcome = lock_and_clear(&active, &mut board);

        assert_eq!(outcome.cleared_rows, vec![0]);
        // The cell that was at (2, 1) drops to (2, 0) after the clear.
        assert_eq!(outcome.top_y_after_lock, Some(0));
        assert_eq!(board.get_cell_kind(2, 0), CellKind::Some(PieceType::O));
    }
}
