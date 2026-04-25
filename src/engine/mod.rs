mod active_piece;
mod api;
mod board;
mod constants;
mod game_over;
mod generator;
mod goals;
mod gravity;
mod lock_down;
mod pieces;
mod scoring;
mod t_spin;

pub use active_piece::{ActivePiece, PieceAction, RotationDirection};
pub use api::{
    ActivePieceSnapshot, Engine, EngineConfig, EngineEvent, EngineSnapshot, GameOverStatus,
    InputFrame, SnapshotCell,
};
pub use board::{Board, Cell, CellKind};
pub use game_over::{is_block_out, is_lock_out, is_top_out};
pub use generator::PieceGenerator;
pub use goals::{
    breaks_back_to_back, fixed_goal_for_level, goal_for_level, qualifies_for_back_to_back,
    variable_goal_for_level, variable_goal_units, GoalProgress, GoalSystem,
};
pub use gravity::{
    fall_duration, fall_speed_seconds, soft_drop_duration, soft_drop_speed_seconds, MAX_LEVEL,
    MIN_LEVEL,
};
pub use lock_down::{
    apply_grounded_move_or_rotation, LockDownMode, EXTENDED_LOCK_RESET_BUDGET, LOCK_DOWN_SECONDS,
};
pub use pieces::{MoveDirection, Piece, PieceRotation, PieceType};
pub use scoring::EngineScoreAction;
pub use t_spin::{classify_t_spin, t_spin_corners, TSpinCorners, TSpinKind};
