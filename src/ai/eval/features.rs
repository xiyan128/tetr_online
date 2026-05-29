//! Static + per-move board features for the linear evaluator (Dellacherie / BCTS).
//!
//! This is the *measurement* layer: it turns a [`Board`] (and the
//! [`LockOutcome`] of the move that produced it) into a small struct of integer
//! counts. The *policy* layer ([`super::weights`]) then dots those counts with a
//! tunable weight vector. Keeping the two apart means the feature definitions can
//! be pinned by exact unit tests independent of any particular weight set.
//!
//! # Feature catalog (finding [1], [2])
//!
//! [`BoardFeatures`] ships Dellacherie's canonical **six** plus the BCTS **two**:
//!
//! | feature              | meaning                                                        |
//! |----------------------|----------------------------------------------------------------|
//! | `landing_height`     | 1-indexed row the placed piece's bottom came to rest on        |
//! | `eroded_piece_cells` | `lines_cleared × (placed-piece cells inside cleared rows)`      |
//! | `row_transitions`    | horizontal filled⇄empty alternations (walls count as filled)   |
//! | `column_transitions` | vertical filled⇄empty alternations (floor counts as filled)    |
//! | `holes`              | empty cells with a filled cell somewhere above in their column |
//! | `board_wells`        | cumulative well depth: `Σ 1+2+…+depth` over each well run      |
//! | `hole_depth`         | BCTS: filled cells stacked directly above each hole            |
//! | `rows_with_holes`    | BCTS: distinct rows containing ≥1 hole                          |
//!
//! Extending to the full BCTS-8 / DT-9 set (a `diversity` feature) is additive:
//! give [`BoardFeatures`] another field, extract it here, weight it in
//! [`super::weights::BoardWeights`].
//!
//! # Coordinate conventions
//!
//! [`Board`] has its origin at the **bottom-left** with `y` increasing upward;
//! off-grid `x` reads as [`CellKind::Wall`] and off-grid `y` reads as empty
//! ([`CellKind::None`]). So "above" means larger `y`, the floor is `y = -1`, and
//! the side walls are `x = -1` / `x = width`. Feature scans run over rows
//! `0..stack_height`, where `stack_height` is the tallest column — everything
//! above that is empty and (for Dellacherie's definitions) contributes nothing, so
//! the engine's hidden spawn buffer never needs special-casing.

use crate::engine::{Board, LockOutcome};

/// Static board-quality features (the Dellacherie-6 + BCTS-2 set).
///
/// Built by [`BoardFeatures::extract`]. All counts are non-negative integers; the
/// sign/scale of their contribution lives entirely in the weights.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BoardFeatures {
    /// 1-indexed height of the row the last piece's lowest cell rested on
    /// (`0` when no piece was placed, e.g. on a freshly built board).
    pub landing_height: i32,
    /// `lines_cleared × (cells of the placed piece that were in cleared rows)`.
    pub eroded_piece_cells: i32,
    /// Horizontal filled⇄empty alternations summed over the stacked rows, with
    /// both side walls treated as filled.
    pub row_transitions: i32,
    /// Vertical filled⇄empty alternations summed over the columns, with the floor
    /// treated as filled and the open top treated as empty.
    pub column_transitions: i32,
    /// Empty cells that have at least one filled cell above them in the same
    /// column.
    pub holes: i32,
    /// Cumulative well depth: each maximal vertical run of well cells of depth `d`
    /// contributes the triangular number `1 + 2 + … + d`.
    pub board_wells: i32,
    /// BCTS: for every hole, the number of filled cells stacked directly above it
    /// in its column, summed over all holes.
    pub hole_depth: i32,
    /// BCTS: the number of distinct rows that contain at least one hole.
    pub rows_with_holes: i32,
}

impl BoardFeatures {
    /// Extract the full feature set from `board` (the board *after* the move's
    /// line clears) and `lock` (the outcome of the move that produced it).
    ///
    /// The per-move features — `landing_height` and `eroded_piece_cells` — read
    /// `lock` because they describe the placement, not the resting board; the rest
    /// are pure functions of `board`. Pass a [`LockOutcome::default`]-style empty
    /// outcome to score a board with no associated move (those two features come
    /// out `0`).
    pub fn extract(board: &Board, lock: &LockOutcome) -> Self {
        let heights = column_heights(board);
        let stack_height = heights.iter().copied().max().unwrap_or(0);

        let (holes, hole_depth, rows_with_holes) = hole_features(board, &heights);

        Self {
            landing_height: landing_height(lock),
            eroded_piece_cells: eroded_piece_cells(lock),
            row_transitions: row_transitions(board, stack_height),
            column_transitions: column_transitions(board, &heights, stack_height),
            holes,
            board_wells: board_wells(board, stack_height),
            hole_depth,
            rows_with_holes,
        }
    }
}

/// Per-column stack height: `(highest filled y) + 1`, or `0` for an empty column.
fn column_heights(board: &Board) -> Vec<i32> {
    let width = board.width();
    let scan_top = board_scan_top(board);
    (0..width as isize)
        .map(|x| {
            (0..scan_top)
                .rev()
                .find(|&y| board.get_cell_kind(x, y).is_some())
                .map_or(0, |y| (y + 1) as i32)
        })
        .collect()
}

/// An exclusive upper `y` bound that covers every filled cell, including any in
/// the hidden spawn buffer. `Board` does not expose its total row count, so we
/// bound by the highest filled cell it reports and add one (everything above is
/// guaranteed empty). Returns `0` for an empty board.
fn board_scan_top(board: &Board) -> isize {
    board
        .cells()
        .iter()
        .map(|cell| cell.coords().1)
        .max()
        .map_or(0, |max_y| max_y + 1)
}

/// 1-indexed height of the lowest cell of the just-placed piece.
///
/// Uses the pre-clear `cells_locked` (so the height reflects where the piece
/// actually came to rest, before any rows it completed were removed). `0` when
/// the move locked nothing.
fn landing_height(lock: &LockOutcome) -> i32 {
    lock.cells_locked
        .iter()
        .map(|(_, y, _)| *y)
        .min()
        .map_or(0, |min_y| (min_y + 1) as i32)
}

/// `lines_cleared × (placed-piece cells that sat in cleared rows)` — Dellacherie's
/// "eroded piece cells", rewarding placements that immediately pay off in clears.
fn eroded_piece_cells(lock: &LockOutcome) -> i32 {
    let lines_cleared = lock.cleared_rows.len() as i32;
    if lines_cleared == 0 {
        return 0;
    }
    let piece_cells_cleared = lock
        .cells_locked
        .iter()
        .filter(|(_, y, _)| lock.cleared_rows.contains(y))
        .count() as i32;
    lines_cleared * piece_cells_cleared
}

/// Horizontal filled⇄empty alternations over rows `0..stack_height`, with both
/// side walls counted as filled (so a gap at the edge of a row still registers).
fn row_transitions(board: &Board, stack_height: i32) -> i32 {
    let width = board.width() as isize;
    let mut transitions = 0;
    for y in 0..stack_height as isize {
        // Walk x = -1 (wall) .. width (wall); a transition is an occupancy flip
        // between adjacent cells.
        let mut prev_filled = true; // left wall
        for x in 0..width {
            let filled = board.get_cell_kind(x, y).is_some();
            if filled != prev_filled {
                transitions += 1;
            }
            prev_filled = filled;
        }
        if !prev_filled {
            // last interior cell empty -> right wall is filled: one more flip.
            transitions += 1;
        }
    }
    transitions
}

/// Vertical filled⇄empty alternations over each column's rows `0..stack_height`,
/// with the floor (`y = -1`) counted as filled and the open top counted as empty.
fn column_transitions(board: &Board, heights: &[i32], stack_height: i32) -> i32 {
    let mut transitions = 0;
    for (x, &_height) in heights.iter().enumerate() {
        let mut prev_filled = true; // floor below y = 0
        for y in 0..stack_height as isize {
            let filled = board.get_cell_kind(x as isize, y).is_some();
            if filled != prev_filled {
                transitions += 1;
            }
            prev_filled = filled;
        }
        // Above the scanned region is empty; if the top scanned cell was filled
        // that is one final flip. (Only possible for the tallest column, where
        // stack_height == its height, so the cell at stack_height-1 is filled.)
        if prev_filled && stack_height > 0 {
            transitions += 1;
        }
    }
    transitions
}

/// Hole-derived features in one column-pass: `(holes, hole_depth, rows_with_holes)`.
///
/// A *hole* is an empty cell that lies below the column's top (i.e. has a filled
/// cell somewhere above it). `hole_depth` sums, per hole, the filled cells stacked
/// directly above it; `rows_with_holes` counts the distinct rows any hole sits in.
fn hole_features(board: &Board, heights: &[i32]) -> (i32, i32, i32) {
    let mut holes = 0;
    let mut hole_depth = 0;
    let mut rows_with_holes_mask: Vec<bool> = Vec::new();

    for (x, &height) in heights.iter().enumerate() {
        // Track how many filled cells we have seen above as we descend, so each
        // hole knows its burial depth in a single pass.
        let mut filled_above = 0;
        for y in (0..height as isize).rev() {
            if board.get_cell_kind(x as isize, y).is_some() {
                filled_above += 1;
            } else {
                holes += 1;
                hole_depth += filled_above;
                let row = y as usize;
                if row >= rows_with_holes_mask.len() {
                    rows_with_holes_mask.resize(row + 1, false);
                }
                rows_with_holes_mask[row] = true;
            }
        }
    }

    let rows_with_holes = rows_with_holes_mask.iter().filter(|&&b| b).count() as i32;
    (holes, hole_depth, rows_with_holes)
}

/// Cumulative well depth (Dellacherie's "wells"): an empty cell whose left and
/// right neighbours are both filled (walls count as filled) is a *well cell*; each
/// maximal vertical run of well cells of depth `d` contributes `1 + 2 + … + d`.
///
/// Scanning each column top-down, a counter tracks the current run depth: a well
/// cell increments it and adds the running depth, a non-well cell resets it.
fn board_wells(board: &Board, stack_height: i32) -> i32 {
    let width = board.width() as isize;
    let mut total = 0;
    for x in 0..width {
        let mut depth = 0;
        for y in (0..stack_height as isize).rev() {
            let empty = board.get_cell_kind(x, y).is_none();
            // Walls bound a well too, so "blocked on this side" means filled *or*
            // off the playfield. `get_cell_kind` returns `Wall` past the edges and
            // `None` for an in-bounds empty cell, so `!is_none()` captures both
            // "filled cell" and "wall".
            let left_blocked = !board.get_cell_kind(x - 1, y).is_none();
            let right_blocked = !board.get_cell_kind(x + 1, y).is_none();
            if empty && left_blocked && right_blocked {
                depth += 1;
                total += depth;
            } else {
                depth = 0;
            }
        }
    }
    total
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{CellKind, PieceType};

    /// Build a `Board` from an ASCII sketch, top row first (so the picture reads
    /// like the screen). `'X'`/`'#'` = filled, anything else = empty. Width is the
    /// length of the first row; height is the number of rows.
    fn board_from_ascii(rows: &[&str]) -> Board {
        let height = rows.len();
        let width = rows[0].len();
        let mut board = Board::new(width, height);
        for (i, line) in rows.iter().enumerate() {
            // rows[0] is the TOP row -> highest y.
            let y = (height - 1 - i) as isize;
            for (x, ch) in line.chars().enumerate() {
                if ch == 'X' || ch == '#' {
                    board.set(x as isize, y, CellKind::Some(PieceType::O));
                }
            }
        }
        board
    }

    /// An empty lock outcome — used when scoring a board with no associated move.
    fn no_move() -> LockOutcome {
        LockOutcome {
            cells_locked: Vec::new(),
            cleared_rows: Vec::new(),
            top_y_after_lock: None,
        }
    }

    #[test]
    fn empty_board_has_all_zero_features() {
        let board = Board::new(10, 20);
        let f = BoardFeatures::extract(&board, &no_move());
        assert_eq!(f, BoardFeatures::default());
    }

    #[test]
    fn column_heights_track_topmost_filled_cell() {
        // Intended shape: col0=3, col1=1, col2 empty, col3=2.
        //   y2: X . . .
        //   y1: X . . X
        //   y0: X X . X
        let board = board_from_ascii(&[
            "X...", // y=2
            "X..X", // y=1
            "XX.X", // y=0
        ]);
        assert_eq!(column_heights(&board), vec![3, 1, 0, 2]);
    }

    #[test]
    fn holes_counts_empty_cells_under_the_surface() {
        // Column 0: filled at y1, empty at y0 -> 1 hole, buried by 1 cell.
        // Column 1: filled at y2, empty y1 and y0 -> 2 holes, depths 1 each.
        let board = board_from_ascii(&[
            ".X", // y=2
            "X.", // y=1
            "..", // y=0
        ]);
        let (holes, hole_depth, rows_with_holes) = hole_features(&board, &column_heights(&board));
        assert_eq!(holes, 3, "col0 y0 + col1 y1,y0");
        // col0: hole at y0 has 1 filled above (y1). col1: holes at y1 (1 above:
        // y2) and y0 (1 above: y2) -> depth 1 + 1. Total 1 + 2 = 3.
        assert_eq!(hole_depth, 3);
        // Holes sit in rows y0 (cols 0,1) and y1 (col1) -> 2 distinct rows.
        assert_eq!(rows_with_holes, 2);
    }

    #[test]
    fn row_transitions_counts_horizontal_flips_with_walls_filled() {
        // Single row "X.X." width 4. Sequence with walls:
        // wall(F) X(F) .(E) X(F) .(E) wall(F)
        //   F->F = 0, F->E = 1, E->F = 1, F->E = 1, E->F = 1  => 4
        let board = board_from_ascii(&["X.X."]);
        assert_eq!(row_transitions(&board, 1), 4);
    }

    #[test]
    fn row_transitions_full_row_has_zero() {
        let board = board_from_ascii(&["XXXX"]);
        // wall(F) F F F F wall(F) -> no flips.
        assert_eq!(row_transitions(&board, 1), 0);
    }

    #[test]
    fn column_transitions_counts_vertical_flips_with_floor_filled() {
        // One column, cells (bottom..top): filled, empty, filled (stack height 3).
        // floor(F) y0(F) y1(E) y2(F) top(E)
        //   F->F=0, F->E=1, E->F=1, F->E(top)=1 => 3
        let mut b = Board::new(1, 4);
        b.set(0, 0, CellKind::Some(PieceType::O));
        b.set(0, 2, CellKind::Some(PieceType::O));
        let heights = column_heights(&b);
        assert_eq!(heights, vec![3]);
        assert_eq!(column_transitions(&b, &heights, 3), 3);
    }

    #[test]
    fn board_wells_uses_triangular_depth() {
        // Width 3. Columns 0 and 2 filled to height 3; column 1 empty -> a well of
        // depth 3 => 1 + 2 + 3 = 6.
        let board = board_from_ascii(&[
            "X.X", // y=2
            "X.X", // y=1
            "X.X", // y=0
        ]);
        assert_eq!(board_wells(&board, 3), 6);
    }

    #[test]
    fn board_wells_edge_column_uses_wall_as_filled() {
        // Width 2. Column 1 filled to height 2; column 0 empty. Column 0's left
        // neighbour is the wall (filled), right neighbour (col1) filled => a well
        // of depth 2 => 1 + 2 = 3.
        let board = board_from_ascii(&[
            ".X", // y=1
            ".X", // y=0
        ]);
        assert_eq!(board_wells(&board, 2), 3);
    }

    #[test]
    fn landing_height_is_lowest_locked_cell_one_indexed() {
        let lock = LockOutcome {
            cells_locked: vec![
                (3, 4, CellKind::Some(PieceType::T)),
                (4, 4, CellKind::Some(PieceType::T)),
                (4, 5, CellKind::Some(PieceType::T)),
                (5, 4, CellKind::Some(PieceType::T)),
            ],
            cleared_rows: Vec::new(),
            top_y_after_lock: Some(5),
        };
        // Lowest locked y is 4 -> 1-indexed height 5.
        assert_eq!(landing_height(&lock), 5);
    }

    #[test]
    fn eroded_piece_cells_is_lines_times_piece_cells_cleared() {
        // A piece whose 4 cells locked into rows 0 and 1; rows 0 and 1 both clear,
        // so all 4 piece cells sit in cleared rows -> 2 lines × 4 cells = 8.
        let lock = LockOutcome {
            cells_locked: vec![
                (0, 0, CellKind::Some(PieceType::I)),
                (1, 0, CellKind::Some(PieceType::I)),
                (2, 0, CellKind::Some(PieceType::I)),
                (0, 1, CellKind::Some(PieceType::I)),
            ],
            cleared_rows: vec![0, 1],
            top_y_after_lock: None,
        };
        assert_eq!(eroded_piece_cells(&lock), 2 * 4);
    }

    #[test]
    fn eroded_piece_cells_counts_only_cells_in_cleared_rows() {
        // 3 piece cells in row 0 (which clears) and 1 in row 1 (which does not):
        // 1 line × 3 cleared piece cells = 3.
        let lock = LockOutcome {
            cells_locked: vec![
                (0, 0, CellKind::Some(PieceType::J)),
                (1, 0, CellKind::Some(PieceType::J)),
                (2, 0, CellKind::Some(PieceType::J)),
                (2, 1, CellKind::Some(PieceType::J)),
            ],
            cleared_rows: vec![0],
            top_y_after_lock: Some(1),
        };
        assert_eq!(eroded_piece_cells(&lock), 3); // 1 line × 3 cells
    }

    #[test]
    fn eroded_piece_cells_zero_without_clears() {
        let lock = LockOutcome {
            cells_locked: vec![(0, 0, CellKind::Some(PieceType::I))],
            cleared_rows: Vec::new(),
            top_y_after_lock: Some(0),
        };
        assert_eq!(eroded_piece_cells(&lock), 0);
    }

    #[test]
    fn extract_pins_a_known_composite_board() {
        // A hand-built board with simultaneously known values for every feature.
        //
        //   y2:  X . .
        //   y1:  X . .
        //   y0:  X . X
        //
        // Heights: col0=3, col1=0, col2=1.
        // holes: none (every empty cell has nothing filled above it). col1 fully
        //   empty; col2 y>=1 empty but nothing above. => 0.
        // row_transitions per row (width 3), walls (F) on both ends:
        //   y0 "X.X": F | X F | . E | X F | F => F-F,F-E,E-F,F-F = 2
        //   y1 "X..": F | X F | . E | . E | F => F-F,F-E,E-E,E-F = 2
        //   y2 "X..": same as y1 => 2.  total = 6.
        // column_transitions (stack_height 3):
        //   col0 filled y0..y2: floor F, F,F,F, top E => only F->E at top = 1
        //   col1 empty: floor F, E,E,E => F->E once = 1
        //   col2 filled y0 only: floor F, F(y0),E(y1),E(y2), top E => F-F,F-E,E-E => 1
        //   total = 3.
        // board_wells (stack_height 3):
        //   col1 is empty with col0 (filled) left and col2... col2 only filled at
        //   y0. At y0: left col0 F, right col2 F -> well cell (depth 1). At y1:
        //   right col2 empty -> not a well. At y2: right empty -> not a well.
        //   col0/col2 are filled (not empty) so no wells. => total 1.
        let mut b = Board::new(3, 4);
        b.set(0, 0, CellKind::Some(PieceType::O));
        b.set(0, 1, CellKind::Some(PieceType::O));
        b.set(0, 2, CellKind::Some(PieceType::O));
        b.set(2, 0, CellKind::Some(PieceType::O));

        let f = BoardFeatures::extract(&b, &no_move());
        assert_eq!(f.holes, 0);
        assert_eq!(f.hole_depth, 0);
        assert_eq!(f.rows_with_holes, 0);
        assert_eq!(f.row_transitions, 6);
        assert_eq!(f.column_transitions, 3);
        assert_eq!(f.board_wells, 1);
        assert_eq!(f.landing_height, 0, "no move => 0");
        assert_eq!(f.eroded_piece_cells, 0);
    }
}
