use std::cmp::Ordering;

use std::ops::Range;
use array2d::Array2D;
use bevy::prelude::Component;
use crate::core::pieces::PieceType;
use itertools::{iproduct, Product};

#[derive(Component)]
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
            cells: Array2D::filled_with(Cell::dummy(0, 0), height + 10, width),
        }
    }

    pub fn set(&mut self, x: isize, y: isize, cell_kind: CellKind) {
        if x < 0 || y < 0 || x >= self.width as isize || y >= self.height as isize {
            return;
        }
        self.cells[(y as usize, x as usize)] = Cell::new(x, y, cell_kind);
    }

    pub fn rows(&self) -> Vec<Vec<Cell>> {
        return self.cells.as_rows();
    }

    pub fn cells(&self) -> Vec<&Cell> {
        self.cells.elements_row_major_iter().filter(|cell| cell.cell_kind.is_some()).collect()
    }

    pub fn get(&self, x: isize, y: isize) -> Option<&Cell> {
        if x < 0 || y < 0 || x >= self.width as isize || y >= self.height as isize {
            return None;
        }

        self.cells[(y as usize, x as usize)].cell_kind.is_some()
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

    pub fn row_cells(&self, row: usize) -> impl Iterator<Item=&Cell> {
        self.cells().into_iter().filter(move |cell| cell.y == row as isize)
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

    pub fn clear_lines(&mut self) -> isize {
        for y in 0..self.height {
            if self.row_cells(y).count() == self.width {
                self.clear_line(y);
                return self.clear_lines() + 1;
            }
        }
        0
    }
}

#[derive(Debug, Copy, Clone)]
pub enum CellKind {
    Some(PieceType),
    None,
    Wall,
}

impl CellKind {
    pub fn is_some(&self) -> bool {
        matches!(self, CellKind::Some(_))
    }

    pub fn is_none(&self) -> bool {
        matches!(self, CellKind::None)
    }

    pub fn is_wall(&self) -> bool {
        matches!(self, CellKind::Wall)
    }

    pub fn unwrap_or(self, default: PieceType) -> PieceType {
        match self {
            CellKind::Some(piece_type) => piece_type,
            _ => default,
        }
    }

    pub fn unwrap(self) -> PieceType {
        match self {
            CellKind::Some(piece_type) => piece_type,
            _ => panic!("CellKind is None or Wall"),
        }
    }

    pub fn as_some(&self) -> Option<PieceType> {
        match self {
            CellKind::Some(piece_type) => Some(*piece_type),
            _ => None,
        }
    }
}

#[derive(Component, Debug, Clone)]
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
        Self {
            x,
            y,
            cell_kind,
        }
    }

    pub fn dummy(x: isize, y: isize) -> Self {
        Self {
            x,
            y,
            cell_kind: CellKind::None,
        }
    }

    pub fn x(&self) -> isize {
        self.x
    }

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