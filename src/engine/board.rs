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
use itertools::{iproduct, Product};
use std::ops::Range;

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
            cells: Array2D::filled_with(Cell::dummy(0, 0), height, width),
        }
    }

    pub fn with_top_margin(width: usize, height: usize, margin: usize) -> Self {
        Self {
            width,
            height,
            cells: Array2D::filled_with(Cell::dummy(0, 0), height + margin, width),
        }
    }

    pub fn set(&mut self, x: isize, y: isize, cell_kind: CellKind) -> bool {
        if x < 0 || y < 0 || x >= self.width as isize || y >= self.cells.column_len() as isize {
            return false;
        }
        self.cells[(y as usize, x as usize)] = Cell::new(x, y, cell_kind);
        true
    }

    pub fn rows(&self) -> Vec<Vec<Cell>> {
        self.cells.as_rows()[..self.height].to_vec()
    }

    pub fn cells(&self) -> Vec<&Cell> {
        self.cells
            .elements_row_major_iter()
            .filter(|cell| cell.cell_kind.is_some())
            .collect()
    }

    pub fn get(&self, x: isize, y: isize) -> Option<&Cell> {
        if x < 0 || y < 0 || x >= self.width as isize || y >= self.height as isize {
            return None;
        }

        self.cells[(y as usize, x as usize)]
            .cell_kind
            .is_some()
            .then(|| &self.cells[(y as usize, x as usize)])
    }

    pub fn get_cell_kind(&self, x: isize, y: isize) -> CellKind {
        if x < 0 || y < 0 || x >= self.width as isize {
            return CellKind::Wall;
        }

        if let Some(cell) = self.cells.get(y as usize, x as usize) {
            return cell.cell_kind;
        }

        CellKind::None
    }

    pub fn coords(&self) -> Product<Range<isize>, Range<isize>> {
        iproduct!(0..self.width as isize, 0..self.height as isize)
    }

    pub fn cell_coords(&self) -> Vec<(isize, isize)> {
        self.cells().iter().map(|cell| (cell.x, cell.y)).collect()
    }

    pub fn row_cells(&self, row: usize) -> impl Iterator<Item = &Cell> {
        self.cells()
            .into_iter()
            .filter(move |cell| cell.y == row as isize)
    }

    pub fn clear_line(&mut self, y: usize) -> Vec<Cell> {
        let mut cleared = Vec::new();

        for x in 0..self.width {
            let cell = self.cells[(y, x)].clone();
            self.set(x as isize, y as isize, CellKind::None);
            cleared.push(cell);
        }

        // move all cells above down
        for y in (y + 1)..self.height {
            for x in 0..self.width {
                let cell = self.cells[(y, x)].clone();
                self.set(x as isize, y as isize, CellKind::None);
                self.set(x as isize, y as isize - 1, cell.cell_kind);
            }
        }

        cleared
    }

    pub fn clear_lines(&mut self) -> usize {
        let mut cleared = 0;
        let mut y = 0;

        while y < self.height {
            if self.row_cells(y).count() == self.width {
                self.clear_line(y);
                cleared += 1;
            } else {
                y += 1;
            }
        }

        cleared
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
        assert_eq!(
            board.get(3, 4).map(Cell::cell_kind),
            Some(CellKind::Some(PieceType::T))
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
        assert!(board.get(4, 25).is_none());
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
    fn clear_lines_handles_multiple_adjacent_full_rows() {
        let mut board = Board::new(4, 4);
        fill_row(&mut board, 0, PieceType::I);
        fill_row(&mut board, 1, PieceType::O);
        assert!(board.set(2, 2, CellKind::Some(PieceType::T)));

        assert_eq!(board.clear_lines(), 2);
        assert_eq!(board.get_cell_kind(2, 0), CellKind::Some(PieceType::T));
        assert_eq!(board.cells().len(), 1);
    }
}

impl CellKind {
    pub fn is_some(&self) -> bool {
        matches!(self, CellKind::Some(_))
    }

    pub fn is_none(&self) -> bool {
        matches!(self, CellKind::None)
    }

    pub fn unwrap(self) -> PieceType {
        match self {
            CellKind::Some(piece_type) => piece_type,
            _ => panic!("CellKind is None or Wall"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Cell {
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

    pub fn dummy(x: isize, y: isize) -> Self {
        Self {
            x,
            y,
            cell_kind: CellKind::None,
        }
    }

    #[cfg(test)]
    pub fn x(&self) -> isize {
        self.x
    }

    #[cfg(test)]
    pub fn y(&self) -> isize {
        self.y
    }

    pub fn coords(&self) -> (isize, isize) {
        (self.x, self.y)
    }

    pub fn cell_kind(&self) -> CellKind {
        self.cell_kind
    }
}
