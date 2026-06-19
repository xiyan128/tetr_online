//! SRS wall-kick tables and tetromino cell layouts.
//!
//! [`I_KICKS`] / [`DEFAULT_KICKS`] hold the five `(dx, dy)` offsets tried for
//! each of the eight rotation transitions (the I piece uses its own table). The
//! [`shapes`] module gives each piece's four cells inside its spawn bounding box;
//! [`avatar_shapes`] gives tight, margin-less layouts for preview/hold rendering.

// Some constant data are from https://github.com/DavideCanton/rust-tetris/blob/master/rust_tetris_core/src/constants.rs#L9
pub type Kick = (isize, isize);

pub(crate) static I_KICKS: [[Kick; 5]; 8] = [
    [(0, 0), (-2, 0), (1, 0), (-2, -1), (1, 2)],
    [(0, 0), (2, 0), (-1, 0), (2, 1), (-1, -2)],
    [(0, 0), (-1, 0), (2, 0), (-1, 2), (2, -1)],
    [(0, 0), (1, 0), (-2, 0), (1, -2), (-2, 1)],
    [(0, 0), (2, 0), (-1, 0), (2, 1), (-1, -2)],
    [(0, 0), (-2, 0), (1, 0), (-2, -1), (1, 2)],
    [(0, 0), (1, 0), (-2, 0), (1, -2), (-2, 1)],
    [(0, 0), (-1, 0), (2, 0), (-1, 2), (2, -1)],
];
pub(crate) static DEFAULT_KICKS: [[Kick; 5]; 8] = [
    [(0, 0), (-1, 0), (-1, 1), (0, -2), (-1, -2)],
    [(0, 0), (1, 0), (1, -1), (0, 2), (1, 2)],
    [(0, 0), (1, 0), (1, -1), (0, 2), (1, 2)],
    [(0, 0), (-1, 0), (-1, 1), (0, -2), (-1, -2)],
    [(0, 0), (1, 0), (1, 1), (0, -2), (1, -2)],
    [(0, 0), (-1, 0), (-1, -1), (0, 2), (-1, 2)],
    [(0, 0), (-1, 0), (-1, -1), (0, 2), (-1, 2)],
    [(0, 0), (1, 0), (1, 1), (0, -2), (1, -2)],
];

type Shape = [(isize, isize); 4];
pub mod shapes {
    use super::Shape;
    // `const` (not `static`) so `pieces.rs` can fold these into a compile-time
    // `cells()` rotation table; every caller copies them out by value anyway.
    pub const I: Shape = [(0, 2), (1, 2), (2, 2), (3, 2)];
    pub const J: Shape = [(0, 1), (1, 1), (2, 1), (0, 2)];
    pub const L: Shape = [(0, 1), (1, 1), (2, 1), (2, 2)];
    pub const O: Shape = [(1, 1), (1, 2), (2, 1), (2, 2)];
    pub const S: Shape = [(0, 1), (1, 1), (1, 2), (2, 2)];
    pub const T: Shape = [(0, 1), (1, 1), (1, 2), (2, 1)];
    pub const Z: Shape = [(0, 2), (1, 2), (1, 1), (2, 1)];
}

pub mod avatar_shapes {
    use super::Shape;
    pub static I: Shape = [(0, 0), (1, 0), (2, 0), (3, 0)];
    pub static J: Shape = [(0, 0), (1, 0), (2, 0), (0, 1)];
    pub static L: Shape = [(0, 0), (1, 0), (2, 0), (2, 1)];
    pub static O: Shape = [(0, 0), (1, 0), (0, 1), (1, 1)];
    pub static S: Shape = [(0, 0), (1, 0), (1, 1), (2, 1)];
    pub static T: Shape = [(0, 0), (1, 0), (2, 0), (1, 1)];
    pub static Z: Shape = [(0, 1), (1, 1), (1, 0), (2, 0)];
}
