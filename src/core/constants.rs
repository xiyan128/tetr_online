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
    //
    pub static I: Shape = [(0, 2), (1, 2), (2, 2), (3, 2)];
    pub static J: Shape = [(0, 1), (1, 1), (2, 1), (0, 2)];
    pub static L: Shape = [(0, 1), (1, 1), (2, 1), (2, 2)];
    pub static O: Shape = [(1, 1), (1, 2), (2, 1), (2, 2)];
    pub static S: Shape = [(0, 1), (1, 1), (1, 2), (2, 2)];
    pub static T: Shape = [(0, 1), (1, 1), (1, 2), (2, 1)];
    pub static Z: Shape = [(0, 2), (1, 2), (1, 1), (2, 1)];
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