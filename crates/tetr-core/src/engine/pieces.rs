//! Tetromino geometry: shapes, rotation, and SRS wall kicks.
//!
//! A [`Piece`] is a [`PieceType`] plus a [`PieceRotation`]. Cell layouts come
//! from the shape tables in [`constants`](crate::engine::constants); rotation is
//! applied by rotating those cells within the piece's bounding box. Movement and
//! rotation queries ([`Piece::try_move`], [`Piece::try_rotate_with_kicks`])
//! return the resolved offset/rotation when unobstructed, implementing the SRS
//! kick tables, and `None` otherwise.

use crate::engine::bit_board::Occupancy;
use crate::engine::board::{Board, CellKind};
use std::fmt::Debug;

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
    pub const ALL: [PieceType; Self::LEN] = [
        PieceType::I,
        PieceType::J,
        PieceType::L,
        PieceType::O,
        PieceType::S,
        PieceType::T,
        PieceType::Z,
    ];

    pub fn all() -> [PieceType; Self::LEN] {
        Self::ALL
    }

    pub const LEN: usize = 7;

    /// A stable `0..7` index in **guideline colour order** (`I, O, T, S, Z, J, L`),
    /// the canonical mapping every renderer/binding should use to index a 7-entry
    /// palette or sprite sheet. Deliberately independent of this enum's *declaration*
    /// order (`I, J, L, O, S, T, Z`), so a binding never has to hard-code an order
    /// that could silently drift if the enum is reordered. The `match` is exhaustive,
    /// so adding a piece type is a compile error here rather than a wrong colour
    /// downstream.
    pub const fn render_index(self) -> u8 {
        match self {
            PieceType::I => 0,
            PieceType::O => 1,
            PieceType::T => 2,
            PieceType::S => 3,
            PieceType::Z => 4,
            PieceType::J => 5,
            PieceType::L => 6,
        }
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
    #[cfg(test)]
    pub const ALL: [PieceRotation; 4] = [
        PieceRotation::R0,
        PieceRotation::R90,
        PieceRotation::R180,
        PieceRotation::R270,
    ];

    #[cfg(test)]
    pub fn all() -> [PieceRotation; 4] {
        Self::ALL
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
            _ => unreachable!("rotation sum is normalized with rem_euclid(4)"),
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

#[derive(Clone, Debug, PartialEq, Eq)]
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
        use crate::engine::constants::shapes::*;
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
        use crate::engine::constants::avatar_shapes::*;
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

    /// The four mino coordinates of this piece in its canonical preview (spawn)
    /// rotation, tightly bounded to the origin. The host renders hold / next previews
    /// from these plus the piece type, without touching the engine's board types.
    pub fn avatar_cells(&self) -> [(isize, isize); 4] {
        Self::get_avatar_shape(self.piece_type)
    }

    /// The `(width, height)` bounding box of [`avatar_cells`](Self::avatar_cells) — the
    /// preview's layout size.
    pub fn avatar_dims(&self) -> (usize, usize) {
        let (max_x, max_y) = Self::get_avatar_shape(self.piece_type)
            .iter()
            .fold((0, 0), |(mx, my), &(x, y)| (mx.max(x), my.max(y)));
        (max_x as usize + 1, max_y as usize + 1)
    }

    pub(crate) fn piece_type(&self) -> PieceType {
        self.piece_type
    }

    pub fn spawn_coords(&self, board_width: usize, visible_height: usize) -> (isize, isize) {
        let x = board_width as isize / 2 - 2;
        let y = match self.piece_type {
            PieceType::I => visible_height as isize - 2,
            _ => visible_height as isize - 1,
        };
        (x, y)
    }

    pub fn cells(&self) -> [(isize, isize); 4] {
        let mut shape = Self::get_shape(self.piece_type);
        if self.piece_type != PieceType::O {
            let (_, height) = self.board_size();
            Self::rotate_shape(height, &mut shape, self.rotation as u8);
        }
        shape
    }

    fn rotate_shape(height: usize, shape: &mut [(isize, isize)], n: u8) {
        for _ in 0..n {
            for (x, y) in shape.iter_mut() {
                let new_x = *y;
                let new_y = height as isize - 1 - *x;
                *x = new_x;
                *y = new_y;
            }
        }
    }

    pub fn collide_with<B: Occupancy>(&self, board: &B, offset: (isize, isize)) -> bool {
        let (x_offset, y_offset) = offset;

        for (x, y) in self.cells() {
            // `Occupancy::blocked` is exactly the old `get_cell_kind(..) != None`:
            // out-of-bounds (wall/floor) or a filled cell. Generic over the board so the
            // search can collision-check on the fast `BitBoard`, not just the `Array2D`.
            if board.blocked(x + x_offset, y + y_offset) {
                return true;
            }
        }
        false
    }

    pub fn try_move<B: Occupancy>(
        &self,
        board: &B,
        offset: (isize, isize),
        direction: MoveDirection,
    ) -> Option<(isize, isize)> {
        let (x_offset, y_offset) = offset;
        let (x_offset, y_offset) = match direction {
            MoveDirection::Left => (x_offset - 1, y_offset),
            MoveDirection::Right => (x_offset + 1, y_offset),
            MoveDirection::Down => (x_offset, y_offset - 1),
        };

        if self.collide_with(board, (x_offset, y_offset)) {
            None
        } else {
            Some((x_offset, y_offset))
        }
    }

    pub fn try_rotate_with_kicks<B: Occupancy>(
        &self,
        board: &B,
        offset: (isize, isize),
        rotation: PieceRotation,
    ) -> Option<(PieceRotation, (isize, isize), u8)> {
        use crate::engine::constants::{DEFAULT_KICKS, I_KICKS};

        if self.piece_type == PieceType::O {
            return Some((PieceRotation::R0, offset, 0)); // O piece doesn't rotate
        }

        let kicks_table = match self.piece_type {
            PieceType::I => &I_KICKS,
            _ => &DEFAULT_KICKS,
        };

        // Row = current orientation, column = target, both keyed by the
        // `PieceRotation` discriminant (R0..R270 = 0..3). The value is the row to
        // use in the kick table. SRS only kicks between adjacent orientations, so
        // every non-adjacent (e.g. 0->180) cell is `NONE` — an unreachable state.
        const NONE: u8 = u8::MAX;
        #[rustfmt::skip]
        const KICK_INDEX: [[u8; 4]; 4] = [
            //         to:  R0    R90   R180  R270
            /* R0   */     [NONE, 0,    NONE, 7   ],
            /* R90  */     [1,    NONE, 2,    NONE],
            /* R180 */     [NONE, 3,    NONE, 4   ],
            /* R270 */     [6,    NONE, 5,    NONE],
        ];

        let kicks_idx = KICK_INDEX[self.rotation as usize][rotation as usize];
        if kicks_idx == NONE {
            unreachable!("Invalid rotation: {:?} -> {:?}", self.rotation, rotation);
        }

        let kicks = kicks_table[kicks_idx as usize];

        for (set_idx, (x_offset, y_offset)) in kicks.iter().enumerate() {
            let new_offset = (offset.0 + x_offset, offset.1 + y_offset);
            if let Some(new_rotation) = self.try_rotate(board, new_offset, rotation) {
                return Some((new_rotation, new_offset, (set_idx + 1) as u8));
            }
        }

        None
    }

    pub fn try_rotate<B: Occupancy>(
        &self,
        board: &B,
        offset: (isize, isize),
        rotation: PieceRotation,
    ) -> Option<PieceRotation> {
        let (x_offset, y_offset) = offset;
        let mut new_piece = self.clone();
        new_piece.rotate_to(rotation);

        if new_piece.collide_with(board, (x_offset, y_offset)) {
            None
        } else {
            Some(new_piece.rotation)
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

    fn global_cells(piece: &Piece, offset: (isize, isize)) -> Vec<(isize, isize)> {
        let mut cells = piece
            .cells()
            .into_iter()
            .map(|(x, y)| (x + offset.0, y + offset.1))
            .collect::<Vec<_>>();
        cells.sort();
        cells
    }

    #[test]
    fn render_index_is_the_canonical_guideline_colour_order() {
        // Pin the I,O,T,S,Z,J,L mapping every binding/renderer depends on. This is
        // deliberately decoupled from the enum's declaration order, so reordering the
        // enum must not change these — if it does, a downstream palette silently
        // mis-colours, and this test is the guard against that.
        assert_eq!(PieceType::I.render_index(), 0);
        assert_eq!(PieceType::O.render_index(), 1);
        assert_eq!(PieceType::T.render_index(), 2);
        assert_eq!(PieceType::S.render_index(), 3);
        assert_eq!(PieceType::Z.render_index(), 4);
        assert_eq!(PieceType::J.render_index(), 5);
        assert_eq!(PieceType::L.render_index(), 6);
        // Bijective over 0..7: every piece has a distinct index in range.
        let mut seen = [false; PieceType::LEN];
        for p in PieceType::all() {
            let i = p.render_index() as usize;
            assert!(i < PieceType::LEN, "render_index out of range");
            assert!(!seen[i], "render_index collision at {i}");
            seen[i] = true;
        }
    }

    #[test]
    fn every_piece_has_four_blocks_in_every_rotation() {
        for piece_type in PieceType::all() {
            let mut piece = Piece::new(piece_type);

            for rotation in PieceRotation::all() {
                piece.rotate_to(rotation);
                // Four DISTINCT cells: the array is always length 4, so the real
                // property is that no two cells of the rotated shape coincide.
                let unique: std::collections::HashSet<(isize, isize)> =
                    piece.cells().into_iter().collect();
                assert_eq!(
                    unique.len(),
                    4,
                    "{piece_type:?} at {rotation:?} should occupy four distinct cells"
                );
            }
        }
    }

    #[test]
    fn spawn_coords_match_translated_guideline_cells() {
        let cases = [
            (PieceType::I, vec![(3, 20), (4, 20), (5, 20), (6, 20)]),
            (PieceType::O, vec![(4, 20), (4, 21), (5, 20), (5, 21)]),
            (PieceType::T, vec![(3, 20), (4, 20), (4, 21), (5, 20)]),
            (PieceType::J, vec![(3, 20), (3, 21), (4, 20), (5, 20)]),
        ];

        for (piece_type, expected) in cases {
            let piece = Piece::new(piece_type);

            assert_eq!(global_cells(&piece, piece.spawn_coords(10, 20)), expected);
        }
    }

    #[test]
    fn o_piece_rotation_is_a_noop() {
        let board = Board::new(10, 20);
        let piece = Piece::new(PieceType::O);

        assert_eq!(
            piece.try_rotate_with_kicks(&board, (4, 18), PieceRotation::R90),
            Some((PieceRotation::R0, (4, 18), 0))
        );
    }

    #[test]
    fn avatar_cells_are_tightly_bounded() {
        for piece_type in PieceType::all() {
            let cells = Piece::new(piece_type).avatar_cells();

            assert_eq!(cells.len(), 4);
            assert!(cells.iter().all(|&(x, y)| x >= 0 && y >= 0));
            assert!(cells.iter().any(|&(x, _)| x == 0));
            assert!(cells.iter().any(|&(_, y)| y == 0));
        }
    }

    #[test]
    fn movement_returns_new_position_when_unblocked() {
        let board = Board::new(10, 20);
        let piece = Piece::new(PieceType::T);

        assert_eq!(
            piece.try_move(&board, (4, 18), MoveDirection::Left),
            Some((3, 18))
        );
        assert_eq!(
            piece.try_move(&board, (4, 18), MoveDirection::Right),
            Some((5, 18))
        );
        assert_eq!(
            piece.try_move(&board, (4, 18), MoveDirection::Down),
            Some((4, 17))
        );
    }

    #[test]
    fn movement_returns_none_on_collision() {
        let board = Board::new(10, 20);
        let piece = Piece::new(PieceType::T);

        assert_eq!(piece.try_move(&board, (-1, 0), MoveDirection::Left), None);
    }

    #[test]
    fn wall_kick_reports_one_based_srs_kick_number() {
        let board = Board::new(10, 20);
        let piece = Piece::new(PieceType::T);

        assert_eq!(
            piece.try_rotate_with_kicks(&board, (8, 5), PieceRotation::R90),
            Some((PieceRotation::R90, (7, 5), 2))
        );
    }
}
