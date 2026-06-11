//! Static + per-move board features for the linear evaluator (Dellacherie / BCTS).
//!
//! This is the *measurement* layer: it turns a [`Board`] (and the
//! [`LockOutcome`] of the move that produced it) into a small struct of integer
//! counts. The *policy* layer ([`super::weights`]) then dots those counts with a
//! tunable weight vector. Keeping the two apart means the feature definitions can
//! be pinned by exact unit tests independent of any particular weight set.
//!
//! # Performance: one column bitboard, then bit ops
//!
//! [`BoardFeatures::extract`] builds a `[u64; width]` column bitboard **once** (bit
//! `y` of column `x` set ⇔ `(x, y)` filled) and every feature reads it with cheap bit
//! tests / `leading_zeros`, instead of re-scanning the board through bounds-checked
//! [`Board::get_cell_kind`] per cell. The evaluator runs millions of times inside the
//! beam, so this is the difference between a research climb taking minutes vs hours.
//! The algorithms are unchanged — only the cell-access primitive — so the pinned
//! feature tests below still hold exactly.
//!
//! # Feature catalog
//!
//! [`BoardFeatures`] ships Dellacherie's canonical **six** plus the BCTS **two**, a
//! Tetris-well offense term, and a combo-readiness term:
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
//! | `tetris_well`        | rows ready to clear via an I-piece in the single lowest column |
//! | `near_full_rows`     | rows filled to exactly width−1 (combo-readiness)               |
//!
//! # Coordinate conventions
//!
//! [`Board`] has its origin at the **bottom-left** with `y` increasing upward;
//! off-grid `x` reads as [`CellKind::Wall`](crate::engine::CellKind::Wall) and off-grid `y` reads as empty
//! ([`CellKind::None`](crate::engine::CellKind::None)). So "above" means larger `y`, the floor is `y = -1`, and
//! the side walls are `x = -1` / `x = width`. Feature scans run over rows
//! `0..stack_height`, where `stack_height` is the tallest column — everything
//! above that is empty and (for Dellacherie's definitions) contributes nothing, so
//! the engine's hidden spawn buffer never needs special-casing.

use crate::engine::{Board, LockOutcome};

/// Static board-quality features (the Dellacherie-6 + BCTS-2 set, plus `tetris_well`).
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
    /// Tetris-well readiness: rows that would clear if the single lowest column were
    /// filled — i.e. lines an I-piece in the well clears. Rewards a Tetris setup,
    /// the offense the general [`board_wells`](Self::board_wells) penalty discourages.
    pub tetris_well: i32,
    /// Combo-readiness: rows filled to exactly `width − 1` (missing a single cell), so
    /// one well-placed cell clears each. A stack of these is a **combo machine** —
    /// clearing one per piece runs up the guideline combo bonus. Rewarding it (with
    /// combo-aware search) is the offense path to high *clean* attack-per-piece, which
    /// concentrated Tetris/T-spin play alone caps out below.
    pub near_full_rows: i32,
}

impl BoardFeatures {
    /// Extract the full feature set from `board` (the board *after* the move's
    /// line clears) and `lock` (the outcome of the move that produced it).
    ///
    /// The per-move features — `landing_height` and `eroded_piece_cells` — read
    /// `lock` because they describe the placement, not the resting board; the rest
    /// are pure functions of `board`. Pass a `LockOutcome::default()`-style empty
    /// outcome to score a board with no associated move (those two features come
    /// out `0`).
    pub fn extract(board: &Board, lock: &LockOutcome) -> Self {
        Self::extract_cols(&board.column_bits(), lock)
    }

    /// Extract the full feature set from a column bitboard (bit `y` of `cols[x]` set
    /// ⇔ `(x, y)` filled) — the zero-copy core behind [`extract`](Self::extract), and
    /// the path an evaluator already holding the search's columns takes directly.
    pub fn extract_cols(cols: &[u64], lock: &LockOutcome) -> Self {
        let heights = column_heights(cols);
        let stack_height = heights.iter().copied().max().unwrap_or(0);

        let (holes, hole_depth, rows_with_holes) = hole_features(cols, &heights);

        Self {
            landing_height: landing_height(lock),
            eroded_piece_cells: eroded_piece_cells(lock),
            row_transitions: row_transitions(cols, stack_height),
            column_transitions: column_transitions(cols, stack_height),
            holes,
            board_wells: board_wells(cols, stack_height),
            hole_depth,
            rows_with_holes,
            tetris_well: tetris_well(cols, &heights, stack_height),
            near_full_rows: near_full_rows(cols, stack_height),
        }
    }
}

/// Combo-readiness: count of rows (below the skyline) filled to exactly `width − 1`
/// cells — one placed cell away from clearing. See
/// [`BoardFeatures::near_full_rows`](BoardFeatures::near_full_rows).
fn near_full_rows(cols: &[u64], stack_height: i32) -> i32 {
    let width = cols.len();
    if width == 0 {
        return 0;
    }
    (0..stack_height)
        .filter(|&y| {
            let filled = cols.iter().filter(|&&c| (c >> y) & 1 == 1).count();
            filled == width - 1
        })
        .count() as i32
}

/// Whether `(x, y)` is a filled in-bounds cell. Off the playfield (any `x` outside
/// `0..width`, or `y < 0` / `y ≥ 64`) reads as **empty** — matching
/// `get_cell_kind(..).is_some()` (a [`CellKind::Wall`] is not `Some`).
fn filled(cols: &[u64], x: isize, y: isize) -> bool {
    x >= 0 && (x as usize) < cols.len() && (0..64).contains(&y) && (cols[x as usize] >> y) & 1 == 1
}

/// Whether `(x, y)` is "blocked" for a well: a filled cell **or** off the playfield
/// (a side wall / the floor). Matches `!get_cell_kind(..).is_none()` (which is true
/// for both `Some` and `Wall`).
fn blocked(cols: &[u64], x: isize, y: isize) -> bool {
    if x < 0 || x as usize >= cols.len() || y < 0 {
        return true; // wall or floor
    }
    filled(cols, x, y)
}

/// Per-column stack height: `(highest filled y) + 1`, or `0` for an empty column.
fn column_heights(cols: &[u64]) -> Vec<i32> {
    cols.iter()
        .map(|&c| {
            if c == 0 {
                0
            } else {
                64 - c.leading_zeros() as i32
            }
        })
        .collect()
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
fn row_transitions(cols: &[u64], stack_height: i32) -> i32 {
    let width = cols.len() as isize;
    let mut transitions = 0;
    for y in 0..stack_height as isize {
        let mut prev_filled = true; // left wall
        for x in 0..width {
            let f = filled(cols, x, y);
            if f != prev_filled {
                transitions += 1;
            }
            prev_filled = f;
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
fn column_transitions(cols: &[u64], stack_height: i32) -> i32 {
    let mut transitions = 0;
    for x in 0..cols.len() as isize {
        let mut prev_filled = true; // floor below y = 0
        for y in 0..stack_height as isize {
            let f = filled(cols, x, y);
            if f != prev_filled {
                transitions += 1;
            }
            prev_filled = f;
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
fn hole_features(cols: &[u64], heights: &[i32]) -> (i32, i32, i32) {
    let mut holes = 0;
    let mut hole_depth = 0;
    let mut rows_with_holes_mask: Vec<bool> = Vec::new();

    for (x, &height) in heights.iter().enumerate() {
        // Track how many filled cells we have seen above as we descend, so each
        // hole knows its burial depth in a single pass.
        let mut filled_above = 0;
        for y in (0..height as isize).rev() {
            if filled(cols, x as isize, y) {
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
fn board_wells(cols: &[u64], stack_height: i32) -> i32 {
    let width = cols.len() as isize;
    let mut total = 0;
    for x in 0..width {
        let mut depth = 0;
        for y in (0..stack_height as isize).rev() {
            let empty = !filled(cols, x, y);
            let left_blocked = blocked(cols, x - 1, y);
            let right_blocked = blocked(cols, x + 1, y);
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

/// Tetris-well readiness (CC2-style): take the single lowest column as the well, then
/// count the consecutive rows at/above its height where *every other* column is filled
/// — i.e. how many lines an I-piece dropped into the well would clear at once.
///
/// This is the offense counterpart to [`board_wells`]: that penalizes wells in general
/// (good for survival), but a Tetris *needs* one deep clean well with the rest filled.
/// A dedicated term lets the policy reward building toward a Tetris without also
/// rewarding general bumpiness. Returns `0` unless exactly one column is the lowest and
/// the rows above its height are complete-except-well.
fn tetris_well(cols: &[u64], heights: &[i32], stack_height: i32) -> i32 {
    let width = cols.len();
    if width == 0 || stack_height == 0 {
        return 0;
    }
    // The well is the lowest column (ties resolve to the lowest index, which then
    // fails the "every other column filled" test below — so ties score 0, correct:
    // two equally-low columns are not a single-well Tetris setup).
    let well = (0..width).min_by_key(|&x| heights[x]).unwrap();
    let mut depth = 0;
    let mut y = heights[well] as isize;
    while y < stack_height as isize {
        let complete_except_well = (0..width).all(|x| x == well || filled(cols, x as isize, y));
        if !complete_except_well {
            break;
        }
        depth += 1;
        y += 1;
    }
    depth
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

    /// The column bitboard of a board — the input every feature helper now takes.
    fn cols_of(board: &Board) -> Vec<u64> {
        board.column_bits().to_vec()
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
    fn near_full_rows_counts_rows_one_cell_from_clearing() {
        // 4-wide: two rows missing exactly one cell (combo-ready), one missing three.
        //   y2: X . X X   (missing col 1)  -> near-full
        //   y1: X X . X   (missing col 2)  -> near-full
        //   y0: . X . .   (missing 3)      -> not
        let board = board_from_ascii(&["X.XX", "XX.X", ".X.."]);
        assert_eq!(BoardFeatures::extract(&board, &no_move()).near_full_rows, 2);
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
        assert_eq!(column_heights(&cols_of(&board)), vec![3, 1, 0, 2]);
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
        let cols = cols_of(&board);
        let (holes, hole_depth, rows_with_holes) = hole_features(&cols, &column_heights(&cols));
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
        assert_eq!(row_transitions(&cols_of(&board), 1), 4);
    }

    #[test]
    fn row_transitions_full_row_has_zero() {
        let board = board_from_ascii(&["XXXX"]);
        // wall(F) F F F F wall(F) -> no flips.
        assert_eq!(row_transitions(&cols_of(&board), 1), 0);
    }

    #[test]
    fn column_transitions_counts_vertical_flips_with_floor_filled() {
        // One column, cells (bottom..top): filled, empty, filled (stack height 3).
        // floor(F) y0(F) y1(E) y2(F) top(E)
        //   F->F=0, F->E=1, E->F=1, F->E(top)=1 => 3
        let mut b = Board::new(1, 4);
        b.set(0, 0, CellKind::Some(PieceType::O));
        b.set(0, 2, CellKind::Some(PieceType::O));
        let cols = cols_of(&b);
        assert_eq!(column_heights(&cols), vec![3]);
        assert_eq!(column_transitions(&cols, 3), 3);
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
        assert_eq!(board_wells(&cols_of(&board), 3), 6);
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
        assert_eq!(board_wells(&cols_of(&board), 2), 3);
    }

    #[test]
    fn tetris_well_counts_clear_ready_rows_over_the_lowest_column() {
        // Columns 0,1,2 filled to height 3; column 3 (the well) empty. An I-piece in
        // col 3 would clear all 3 rows ⇒ tetris_well = 3.
        let board = board_from_ascii(&[
            "XXX.", // y=2
            "XXX.", // y=1
            "XXX.", // y=0
        ]);
        let cols = cols_of(&board);
        assert_eq!(tetris_well(&cols, &column_heights(&cols), 3), 3);
    }

    #[test]
    fn tetris_well_zero_when_a_second_column_is_also_low() {
        // Two empty columns ⇒ not a single-well Tetris setup ⇒ 0.
        let board = board_from_ascii(&[
            "XX..", // y=1
            "XX..", // y=0
        ]);
        let cols = cols_of(&board);
        assert_eq!(tetris_well(&cols, &column_heights(&cols), 2), 0);
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
