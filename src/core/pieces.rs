use std::fmt::{Debug, Display};
use bevy::prelude::Component;
use crate::core::board::{Board, CellKind};

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub enum PieceType {
    I,
    J,
    L,
    O,
    S,
    T,
    Z,
}

impl PieceType {
    pub fn all() -> Vec<PieceType> {
        vec![
            PieceType::I,
            PieceType::J,
            PieceType::L,
            PieceType::O,
            PieceType::S,
            PieceType::T,
            PieceType::Z,
        ]
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub enum PieceRotation {
    R0 = 0,
    R90 = 1,
    R180 = 2,
    R270 = 3,
}

impl PieceRotation {
    pub fn all() -> Vec<PieceRotation> {
        vec![
            PieceRotation::R0,
            PieceRotation::R90,
            PieceRotation::R180,
            PieceRotation::R270,
        ]
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub enum MoveDirection {
    Left,
    Right,
    Down,
}

#[derive(Component, Clone, Debug)]
pub struct Piece {
    piece_type: PieceType,
    rotation: PieceRotation,
}

impl Piece {
    fn new(piece_type: PieceType) -> Self {
        Self {
            piece_type,
            rotation: PieceRotation::R0,
        }
    }

    pub fn rotate(&mut self) {
        self.rotation = match self.rotation {
            PieceRotation::R0 => PieceRotation::R90,
            PieceRotation::R90 => PieceRotation::R180,
            PieceRotation::R180 => PieceRotation::R270,
            PieceRotation::R270 => PieceRotation::R0,
        }
    }

    pub fn rotate_n(&mut self, n: u8) {
        let n = n.rem_euclid(4);
        for _ in 0..n {
            self.rotate();
        }
    }

    pub fn rotate_to(&mut self, rotation: PieceRotation) {
        self.rotation = rotation;
    }

    fn board_size(&self) -> (usize, usize) {
        match self.piece_type {
            PieceType::I => (4, 4),
            PieceType::O => (4, 3),
            _ => (3, 3),
        }
    }

    fn get_shape(piece_type: PieceType) -> [(i32, i32); 4] {
        match piece_type {
            PieceType::I => [(1, 0), (1, 1), (1, 2), (1, 3)],
            PieceType::J => [(0, 2), (1, 0), (1, 1), (1, 2)],
            PieceType::L => [(0, 0), (1, 0), (1, 1), (1, 2)],
            PieceType::O => [(1, 1), (1, 2), (2, 1), (2, 2)],
            PieceType::S => [(0, 1), (0, 2), (1, 0), (1, 1)],
            PieceType::T => [(0, 1), (1, 0), (1, 1), (1, 2)],
            PieceType::Z => [(0, 0), (0, 1), (1, 1), (1, 2)],
        }
    }

    pub(crate) fn piece_type(&self) -> PieceType {
        self.piece_type
    }


    pub fn board(&self) -> Board {
        let (width, height) = self.board_size();
        let mut board = Board::new(width, height);

        let mut shape = Self::get_shape(self.piece_type).to_vec();

        let n = self.rotation as u8;

        if self.piece_type != PieceType::O {
            // if piece is not O, rotate shape
            for _ in 0..n {
                shape = shape.iter().map(|(x, y)| (*y, (height as i32) - x - 1)).collect();
            }
        }

        for (id, &(x, y)) in shape.iter().enumerate() {
            board.set(x, y, CellKind::Some(self.piece_type));
        }

        board
    }

    pub fn collide_with(&self, board: &Board, offset: (i32, i32)) -> bool {
        let piece_board = self.board();
        let (x_offset, y_offset) = offset;

        for (x, y) in self.board().cell_coords() {
            let piece_cell = piece_board.get_cell_kind(x, y);
            let board_cell = board.get_cell_kind(x + x_offset, y + y_offset);

            if piece_cell.is_some() && !board_cell.is_none() {
                return true;
            }
        }
        false
    }

    pub fn try_move(&self, board : &Board, offset: (i32, i32), direction: MoveDirection) -> Result<(i32, i32), ()> {
        let (x_offset, y_offset) = offset;
        let (x_offset, y_offset) = match direction {
            MoveDirection::Left => (x_offset - 1, y_offset),
            MoveDirection::Right => (x_offset + 1, y_offset),
            MoveDirection::Down => (x_offset, y_offset - 1),
        };

        if self.collide_with(board, (x_offset, y_offset)) {
            Err(())
        } else {
            Ok((x_offset, y_offset)) // return new offset
        }
    }

    pub fn try_rotate(&mut self, board : &Board, offset: (i32, i32), rotation_n: u8) -> Result<PieceRotation, ()> {
        let (x_offset, y_offset) = offset;
        let mut new_piece = self.clone();
        new_piece.rotate_n(rotation_n);

        if new_piece.collide_with(board, (x_offset, y_offset)) {
            Err(())
        } else {
            Ok(new_piece.rotation) // return new rotation
        }
    }
}

impl From<PieceType> for Piece {
    fn from(piece_type: PieceType) -> Self {
        Self::new(piece_type)
    }
}

impl Display for Piece {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // show board representation
        let board = self.board();
        let mut s = String::new();

        for row in board.rows() {
            for cell in row {

                s.push_str(match cell.cell_kind {
                    CellKind::Some(_) => "X",
                    CellKind::None => "#",
                     _ => " ",
                });
            }
            s.push_str("\n");
        }

        write!(f, "{}", s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_piece_rotation() {
        let mut piece = Piece::new(PieceType::I);
        assert_eq!(piece.board_size(), (4, 4));
        assert_eq!(piece.board().rows().len(), 4);
        assert_eq!(piece.board().rows()[0].len(), 4);

        piece.rotate();
        assert_eq!(piece.board_size(), (4, 4));
        assert_eq!(piece.board().rows().len(), 4);
        assert_eq!(piece.board().rows()[0].len(), 4);
    }

    #[test]
    fn test_piece_display() {

        // print all pieces and rotations
        for piece_type in PieceType::all() {
            for rotation in PieceRotation::all() {
                let mut piece = Piece::new(piece_type);
                piece.rotation = rotation;
                println!("{}", piece);
            }
        }

        let mut piece = Piece::new(PieceType::L);

        println!("{}", piece);
    }
}