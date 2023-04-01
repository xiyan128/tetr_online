use crate::core::board::{Board, CellKind};
use bevy::prelude::{info, Component};
use itertools::Itertools;
use std::fmt::{Debug, Display};

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

    pub const LEN: usize = 7;
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

// add two PieceRotation
impl std::ops::Add for PieceRotation {
    type Output = PieceRotation;

    fn add(self, other: PieceRotation) -> PieceRotation {
        let sum = (self as u8 + other as u8).rem_euclid(4);
        match sum {
            0 => PieceRotation::R0,
            1 => PieceRotation::R90,
            2 => PieceRotation::R180,
            3 => PieceRotation::R270,
            _ => PieceRotation::R0,
        }
    }
}

impl std::ops::AddAssign for PieceRotation {
    fn add_assign(&mut self, other: PieceRotation) {
        *self = *self + other;
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

    pub(crate) fn rotation(&self) -> PieceRotation {
        self.rotation
    }

    pub fn rotate(&mut self) {
        self.rotation += PieceRotation::R90;
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

    pub fn board_size(&self) -> (usize, usize) {
        match self.piece_type {
            PieceType::I => (4, 4),
            PieceType::O => (4, 3),
            _ => (3, 3),
        }
    }

    fn get_shape(piece_type: PieceType) -> [(isize, isize); 4] {
        use crate::core::constants::shapes::*;
        match piece_type {
            PieceType::I => I,
            PieceType::J => J,
            PieceType::L => L,
            PieceType::O => O,
            PieceType::S => S,
            PieceType::T => T,
            PieceType::Z => Z,
        }
    }

    // margin-less board with fixed rotation
    fn get_avatar_shape(piece_type: PieceType) -> [(isize, isize); 4] {
        use crate::core::constants::avatar_shapes::*;
        match piece_type {
            PieceType::I => I,
            PieceType::J => J,
            PieceType::L => L,
            PieceType::O => O,
            PieceType::S => S,
            PieceType::T => T,
            PieceType::Z => Z,
        }
    }

    pub(crate) fn avatar_board(&self) -> Board {
        let mut shape = Self::get_avatar_shape(self.piece_type).to_vec();
        // get width and height of the avatar board
        // those are the max x and y values of the shape
        let (width, height) = shape.iter().fold((0, 0), |(max_x, max_y), (x, y)| {
            (max_x.max(*x), max_y.max(*y))
        });
        let mut board = Board::new(width as usize + 1, height as usize + 1);

        for (x, y) in shape {
            board.set(x, y, CellKind::Some(self.piece_type));
        }

        board
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
            Self::rotate_shape(height, &mut shape, n);
        }

        for (x, y) in shape {
            board.set(x, y, CellKind::Some(self.piece_type));
        }

        board
    }

    fn rotate_shape(height: usize, shape: &mut Vec<(isize, isize)>, n: u8) {
        for _ in 0..n {
            for (x, y) in shape.iter_mut() {
                let new_x = *y;
                let new_y = height as isize - 1 - *x;
                *x = new_x;
                *y = new_y;
            }
        }
    }

    pub fn collide_with(&self, board: &Board, offset: (isize, isize)) -> bool {
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

    pub fn try_move(
        &self,
        board: &Board,
        offset: (isize, isize),
        direction: MoveDirection,
    ) -> Result<(isize, isize), ()> {
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

    pub fn try_rotate_with_kicks(
        &self,
        board: &Board,
        offset: (isize, isize),
        rotation: PieceRotation,
    ) -> Result<(PieceRotation, (isize, isize), usize), ()> {
        use crate::core::constants::{DEFAULT_KICKS, I_KICKS};

        if self.piece_type == PieceType::O {
            return Ok((PieceRotation::R0, offset, 0)); // O piece doesn't rotate
        }

        let kicks_table = match self.piece_type {
            PieceType::I => &I_KICKS,
            _ => &DEFAULT_KICKS,
        };

        let kicks_idx = match (self.rotation, rotation) {
            //0->R
            // R->0
            // R->2
            // 2->R
            // 2->L
            // L->2
            // L->0
            // 0->L
            (PieceRotation::R0, PieceRotation::R90) => 0,
            (PieceRotation::R90, PieceRotation::R0) => 1,
            (PieceRotation::R90, PieceRotation::R180) => 2,
            (PieceRotation::R180, PieceRotation::R90) => 3,
            (PieceRotation::R180, PieceRotation::R270) => 4,
            (PieceRotation::R270, PieceRotation::R180) => 5,
            (PieceRotation::R270, PieceRotation::R0) => 6,
            (PieceRotation::R0, PieceRotation::R270) => 7,
            _ => unreachable!("Invalid rotation: {:?} -> {:?}", self.rotation, rotation),
        };

        let kicks = kicks_table[kicks_idx];

        for (set_idx, (x_offset, y_offset)) in kicks.iter().enumerate() {
            let new_offset = (offset.0 + x_offset, offset.1 + y_offset);
            if let Ok(new_rotation) = self.try_rotate(board, new_offset, rotation) {
                info!("Kicked to {:?} (set {:?})", (x_offset, y_offset), set_idx);
                return Ok((new_rotation, new_offset, set_idx));
            }
        }

        Err(())
    }

    pub fn try_rotate(
        &self,
        board: &Board,
        offset: (isize, isize),
        rotation: PieceRotation,
    ) -> Result<PieceRotation, ()> {
        let (x_offset, y_offset) = offset;
        let mut new_piece = self.clone();
        new_piece.rotate_to(rotation);

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
                println!("{}", piece.board());
            }
        }

        let mut piece = Piece::new(PieceType::L);

        println!("{}", piece.board());
    }

    #[test]
    fn test_avatars() {
        for piece_type in PieceType::all() {
            let piece = Piece::new(piece_type);
            let avatar = piece.avatar_board();
            println!("{:?}: \n{}", piece_type, piece.avatar_board());
        }
    }
}
