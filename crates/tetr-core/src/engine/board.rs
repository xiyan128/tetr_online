//! The playfield grid and its cells.
//!
//! [`Board`] is a row-major matrix of [`Cell`]s addressed by signed `(x, y)`
//! coordinates with the origin at the bottom-left. An optional top margin holds
//! the hidden spawn rows above the visible field. Off-grid reads resolve to
//! [`CellKind::Wall`] (sides/floor) so collision checks need no bounds special-
//! casing.

use std::cmp::Ordering;
use std::fmt::{Display, Write};

use crate::engine::pieces::PieceType;
use array2d::Array2D;
use smallvec::{smallvec, SmallVec};

#[derive(Clone)]
pub struct Board {
    width: usize,
    height: usize,
    cells: Array2D<Cell>,
}

impl Board {
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            cells: Array2D::filled_with(Cell::default(), height, width),
        }
    }

    pub fn with_top_margin(width: usize, height: usize, margin: usize) -> Self {
        Self {
            width,
            height,
            cells: Array2D::filled_with(Cell::default(), height + margin, width),
        }
    }

    pub fn set(&mut self, x: isize, y: isize, cell_kind: CellKind) -> bool {
        if x < 0 || y < 0 || x >= self.width as isize || y >= self.cells.column_len() as isize {
            return false;
        }
        self.cells[(y as usize, x as usize)] = Cell::new(x, y, cell_kind);
        true
    }

    /// Total backing rows (visible field + hidden buffer): the grid height a cell can
    /// occupy. A `set`/lock at or above this is dropped — a cell cannot exist off the
    /// top of the grid.
    pub fn backing_rows(&self) -> usize {
        self.cells.column_len()
    }

    pub(crate) fn rows(&self) -> Vec<Vec<Cell>> {
        self.cells.as_rows()[..self.height].to_vec()
    }

    pub(crate) fn cells(&self) -> Vec<&Cell> {
        self.cells
            .elements_row_major_iter()
            .filter(|cell| cell.cell_kind.is_some())
            .collect()
    }

    /// True iff no playfield cell (visible **or** buffer) is filled — i.e. a perfect
    /// clear. Short-circuits on the first occupied cell and allocates nothing, unlike
    /// [`cells`](Self::cells) or a full snapshot; called on the line-clear hot path.
    pub fn is_empty(&self) -> bool {
        !self
            .cells
            .elements_row_major_iter()
            .any(|cell| cell.cell_kind.is_some())
    }

    pub fn get_cell_kind(&self, x: isize, y: isize) -> CellKind {
        if x < 0 || y < 0 || x >= self.width as isize {
            return CellKind::Wall;
        }
        // `x` is already in bounds, so index directly once `y` is on the backing grid —
        // this skips `Array2D::get`'s redundant `x` bound-check and `Option` wrapping.
        // Hot path: movegen collision + T-spin corners query this per cell per pose.
        if (y as usize) < self.cells.column_len() {
            self.cells[(y as usize, x as usize)].cell_kind
        } else {
            CellKind::None
        }
    }

    pub fn cell_coords(&self) -> Vec<(isize, isize)> {
        self.cells().iter().map(|cell| (cell.x, cell.y)).collect()
    }

    /// The column bitboard: `result[x]` has bit `y` set iff `(x, y)` is occupied
    /// (buffer rows included). Built in one pass over the backing grid — no
    /// intermediate cell `Vec` — so it is cheaper than `cells()`. Shared by the
    /// linear and CC2 evaluators so both pack the playfield identically; this runs
    /// once per board evaluation (the search hot path), so it returns a stack
    /// [`SmallVec`] — no heap allocation for the standard ≤16-wide board.
    pub fn column_bits(&self) -> SmallVec<[u64; 16]> {
        let mut cols: SmallVec<[u64; 16]> = smallvec![0u64; self.width];
        for cell in self.cells.elements_row_major_iter() {
            if cell.cell_kind.is_some() {
                let (x, y) = (cell.x, cell.y);
                if x >= 0 && (x as usize) < self.width && (0..64).contains(&y) {
                    cols[x as usize] |= 1u64 << y;
                }
            }
        }
        cols
    }

    pub(crate) fn row_cells(&self, row: usize) -> impl Iterator<Item = &Cell> {
        self.cells()
            .into_iter()
            .filter(move |cell| cell.y == row as isize)
    }

    pub(crate) fn clear_line(&mut self, y: usize) -> Vec<Cell> {
        let mut cleared = Vec::new();

        for x in 0..self.width {
            let cell = self.cells[(y, x)];
            self.set(x as isize, y as isize, CellKind::None);
            cleared.push(cell);
        }

        // Move every cell above the cleared row down one. Bound by the full
        // backing array (visible + buffer), not `self.height`: a piece can lock
        // partly in the buffer zone above the skyline (§16.4), and those cells
        // must fall too — otherwise they are left floating, unsupported (§11.3).
        for y in (y + 1)..self.cells.column_len() {
            for x in 0..self.width {
                let cell = self.cells[(y, x)];
                self.set(x as isize, y as isize, CellKind::None);
                self.set(x as isize, y as isize - 1, cell.cell_kind);
            }
        }

        cleared
    }

    /// Remove every completely-filled row across the **full backing matrix** (visible
    /// field + hidden buffer) and compact the stack downward, returning the count
    /// removed. Scanning the whole matrix — not just the visible field — is the
    /// guideline rule: a row that fills entirely in the buffer zone clears like any
    /// other. The scan re-examines the same index after each clear, so a row that drops
    /// into a just-cleared slot is itself checked.
    pub fn clear_lines(&mut self) -> usize {
        let mut cleared = 0;
        let mut y = 0;
        let backing = self.backing_rows();

        while y < backing {
            if self.row_cells(y).count() == self.width {
                self.clear_line(y);
                cleared += 1;
            } else {
                y += 1;
            }
        }

        cleared
    }

    /// Insert `count` garbage rows at the bottom, shifting the whole stack up —
    /// the inverse of [`clear_line`](Self::clear_line). Each new row is full except
    /// `hole_col`, painted [`CellKind::Garbage`] so a renderer can tell attack
    /// from the player's own stack. Returns `true` if any filled cell was forced
    /// past the backing top (a garbage-induced top-out for the caller to act on).
    pub fn insert_garbage_lines(&mut self, count: usize, hole_col: usize) -> bool {
        if count == 0 {
            return false;
        }
        // A hole column past the right wall would fill the whole row (no gap); clamp
        // so out-of-range garbage always leaves a diggable hole rather than a free clear.
        let hole_col = hole_col.min(self.width.saturating_sub(1));
        let backing = self.cells.column_len();
        let mut overflow = false;

        // Walk top row first so a cell is never overwritten before it has moved:
        // destination `y + count` is always a row we have already vacated.
        for y in (0..backing).rev() {
            for x in 0..self.width {
                let kind = self.cells[(y, x)].cell_kind;
                self.set(x as isize, y as isize, CellKind::None);
                // `is_some`, not a `Some(_)` pattern: rows risen by an earlier
                // call are `Garbage` cells and must shift up like any others.
                if kind.is_some() {
                    let ny = y + count;
                    if ny < backing {
                        self.set(x as isize, ny as isize, kind);
                    } else {
                        overflow = true;
                    }
                }
            }
        }

        for y in 0..count {
            for x in 0..self.width {
                if x != hole_col {
                    self.set(x as isize, y as isize, CellKind::Garbage);
                }
            }
        }

        overflow
    }

    pub fn width(&self) -> usize {
        self.width
    }

    pub fn height(&self) -> usize {
        self.height
    }
}

impl Display for Board {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Render top row first so the output reads like the on-screen board.
        for row in self.rows().iter().rev() {
            for cell in row {
                f.write_str(match cell.cell_kind {
                    CellKind::Some(_) => "X",
                    CellKind::Garbage => "G",
                    CellKind::None => "#",
                    CellKind::Wall => " ",
                })?;
            }
            f.write_char('\n')?;
        }

        Ok(())
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum CellKind {
    Some(PieceType),
    None,
    Wall,
    /// A garbage-row cell (versus). Occupied exactly like `Some` — it collides,
    /// fills rows, and clears — but carries no piece identity, so a renderer
    /// can paint it neutral instead of a piece colour. Occupancy predicates go
    /// through [`CellKind::is_some`] / [`CellKind::is_none`], which treat it as
    /// filled.
    Garbage,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fill_row(board: &mut Board, y: isize, piece_type: PieceType) {
        for x in 0..board.width {
            assert!(board.set(x as isize, y, CellKind::Some(piece_type)));
        }
    }

    #[test]
    fn set_and_get_round_trip_inside_visible_board() {
        let mut board = Board::new(10, 20);

        assert!(board.set(3, 4, CellKind::Some(PieceType::T)));
        assert_eq!(board.get_cell_kind(3, 4), CellKind::Some(PieceType::T));
    }

    #[test]
    fn is_empty_tracks_occupancy_including_the_buffer() {
        let mut board = Board::with_top_margin(10, 20, 20);
        assert!(board.is_empty(), "a fresh board is empty");

        // A filled cell up in the hidden buffer still counts as non-empty.
        assert!(board.set(4, 25, CellKind::Some(PieceType::I)));
        assert!(!board.is_empty(), "a buffer-zone cell makes it non-empty");

        // Clearing it back to None restores emptiness (a perfect clear).
        assert!(board.set(4, 25, CellKind::None));
        assert!(
            board.is_empty(),
            "clearing the only cell restores emptiness"
        );
    }

    #[test]
    fn horizontal_bounds_are_walls() {
        let board = Board::new(10, 20);

        assert_eq!(board.get_cell_kind(-1, 0), CellKind::Wall);
        assert_eq!(board.get_cell_kind(10, 0), CellKind::Wall);
    }

    #[test]
    fn negative_y_is_floor_collision() {
        let board = Board::new(10, 20);

        assert_eq!(board.get_cell_kind(0, -1), CellKind::Wall);
    }

    #[test]
    fn top_margin_accepts_hidden_spawn_cells() {
        let mut board = Board::with_top_margin(10, 20, 20);

        assert!(board.set(4, 25, CellKind::Some(PieceType::I)));
        assert_eq!(board.get_cell_kind(4, 25), CellKind::Some(PieceType::I));
    }

    #[test]
    fn clear_line_removes_row_and_drops_above_cells() {
        let mut board = Board::new(4, 4);
        fill_row(&mut board, 0, PieceType::I);
        assert!(board.set(1, 1, CellKind::Some(PieceType::T)));

        let cleared = board.clear_line(0);

        assert_eq!(cleared.len(), 4);
        assert_eq!(board.get_cell_kind(1, 0), CellKind::Some(PieceType::T));
        assert_eq!(board.get_cell_kind(1, 1), CellKind::None);
    }

    #[test]
    fn insert_garbage_shifts_stack_up_and_opens_a_hole() {
        let mut board = Board::new(4, 4);
        board.set(1, 0, CellKind::Some(PieceType::T)); // a cell on the floor

        let overflow = board.insert_garbage_lines(1, 2); // one row, hole at col 2

        assert!(!overflow);
        // The pre-existing cell rose by one row.
        assert_eq!(board.get_cell_kind(1, 1), CellKind::Some(PieceType::T));
        // New bottom row: full except the hole column.
        for x in 0..4 {
            let expected = if x == 2 {
                CellKind::None
            } else {
                CellKind::Garbage
            };
            assert_eq!(board.get_cell_kind(x, 0), expected, "col {x}");
        }
    }

    #[test]
    fn insert_garbage_reports_overflow_when_pushed_past_the_ceiling() {
        let mut board = Board::new(4, 4); // 4 rows, no buffer
        board.set(0, 3, CellKind::Some(PieceType::T)); // top visible row

        // Pushing up by 2 forces the top cell off the backing array.
        let overflow = board.insert_garbage_lines(2, 0);

        assert!(overflow);
        // No T cell survived anywhere; the bottom two rows are garbage, hole at col 0.
        assert!(board
            .cells()
            .iter()
            .all(|c| c.cell_kind != CellKind::Some(PieceType::T)));
        assert_eq!(board.get_cell_kind(0, 0), CellKind::None);
        assert_eq!(board.get_cell_kind(1, 0), CellKind::Garbage);
        assert_eq!(board.get_cell_kind(1, 1), CellKind::Garbage);
    }

    #[test]
    fn clear_line_drops_cells_in_the_buffer_zone_above_visible_height() {
        // Regression: the shift-down must cover the full backing array, not just
        // the visible height. A cell that locked in the buffer zone (y >= visible
        // height, legal per §16.4) above a cleared visible row has to fall like
        // any other — otherwise it is left floating above the skyline (§11.3).
        // The plain `Board::new` clear test above hides this: with no buffer, the
        // visible-height bound *is* the array bound.
        let mut board = Board::with_top_margin(4, 4, 4); // visible 4, buffer 4 => 8 rows
        fill_row(&mut board, 0, PieceType::I); // full visible row 0 -> clears
        assert!(board.set(0, 2, CellKind::Some(PieceType::T))); // visible cell above
        assert!(board.set(0, 4, CellKind::Some(PieceType::S))); // buffer-zone cell

        assert_eq!(board.clear_lines(), 1);

        // Both cells dropped one row; nothing left floating in the buffer.
        assert_eq!(board.get_cell_kind(0, 1), CellKind::Some(PieceType::T));
        assert_eq!(board.get_cell_kind(0, 3), CellKind::Some(PieceType::S));
        assert_eq!(board.get_cell_kind(0, 4), CellKind::None);
    }

    #[test]
    fn clear_lines_handles_multiple_adjacent_full_rows() {
        let mut board = Board::new(4, 4);
        fill_row(&mut board, 0, PieceType::I);
        fill_row(&mut board, 1, PieceType::O);
        assert!(board.set(2, 2, CellKind::Some(PieceType::T)));

        assert_eq!(board.clear_lines(), 2);
        assert_eq!(board.get_cell_kind(2, 0), CellKind::Some(PieceType::T));
        assert_eq!(board.cells().len(), 1);
    }

    #[test]
    fn clear_lines_clears_a_full_buffer_row() {
        // Guideline whole-matrix rule: a row that fills entirely in the buffer zone
        // (y >= visible height) clears and counts like any visible row. Regression guard
        // for the clear/score conflation — the physical clear must remove exactly what
        // the scored count (full_rows) reports.
        let mut board = Board::with_top_margin(4, 4, 4); // visible 4, buffer 4 => 8 backing
        fill_row(&mut board, 5, PieceType::I); // a full row, entirely in the buffer
        assert!(board.set(0, 6, CellKind::Some(PieceType::T))); // a lone sentinel above it

        assert_eq!(
            board.clear_lines(),
            1,
            "the full buffer row clears and is counted"
        );

        assert_eq!(
            board.get_cell_kind(0, 5),
            CellKind::Some(PieceType::T),
            "sentinel fell one row"
        );
        assert_eq!(board.get_cell_kind(0, 6), CellKind::None);
        assert_eq!(board.cells().len(), 1, "only the sentinel remains");
    }
}

impl CellKind {
    /// A filled mino cell — a locked piece or a garbage cell. This is the
    /// "counts toward a full row / collides / tops out" predicate; `Wall` is
    /// not `some`.
    pub fn is_some(&self) -> bool {
        matches!(self, CellKind::Some(_) | CellKind::Garbage)
    }

    pub fn is_none(&self) -> bool {
        matches!(self, CellKind::None)
    }

    pub fn unwrap(self) -> PieceType {
        match self {
            CellKind::Some(piece_type) => piece_type,
            _ => panic!("CellKind carries no piece type (None, Wall, or Garbage)"),
        }
    }
}

// `Copy`: all fields are `Copy` and there is no `Drop`, so a board (`Array2D<Cell>`,
// i.e. a `Vec<Cell>`) clones via memcpy rather than per-element — this is on the
// search's hot path, which clones a board per candidate placement.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Cell {
    // unique index
    pub(crate) x: isize,
    y: isize,
    pub(crate) cell_kind: CellKind,
}

impl Eq for Cell {}

impl PartialEq<Self> for Cell {
    fn eq(&self, other: &Self) -> bool {
        self.x == other.x && self.y == other.y
    }
}

impl PartialOrd<Self> for Cell {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Cell {
    fn cmp(&self, other: &Self) -> Ordering {
        // row and then column
        self.y.cmp(&other.y).then(self.x.cmp(&other.x))
    }
}

impl Default for Cell {
    fn default() -> Self {
        Self {
            x: 0,
            y: 0,
            cell_kind: CellKind::None,
        }
    }
}

impl Cell {
    pub fn new(x: isize, y: isize, cell_kind: CellKind) -> Self {
        Self { x, y, cell_kind }
    }

    pub fn coords(&self) -> (isize, isize) {
        (self.x, self.y)
    }

    pub fn cell_kind(&self) -> CellKind {
        self.cell_kind
    }
}
