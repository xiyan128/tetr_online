//! Engine-agnostic Tetris core.
//!
//! Everything under this module implements the game rules as plain Rust with no
//! Bevy (or other engine) dependency — the host drives it through the
//! [`Engine`] facade using plain data ([`InputFrame`] in, [`EngineEvent`]s and
//! [`EngineSnapshot`] out). Submodules are split by concern: board/piece
//! geometry, the seven-bag generator, gravity and lock-down timing, line
//! clearing, scoring, level goals, T-spin detection, and game-over conditions.
//! Most of those concerns are exposed as pure free functions so they can be
//! reused outside the per-frame loop (search bots, replay validators).

mod active_piece;
mod api;
mod attack;
mod bit_board;
mod board;
mod constants;
mod game_over;
pub(crate) mod garbage; // crate-visible: the search mirrors its rules (one home)
mod generator;
mod goals;
mod gravity;
mod lock_clear;
mod lock_down;
mod pieces;
mod scoring;
mod t_spin;
mod types;

pub use active_piece::{ActivePiece, PieceAction, RotationDirection};
pub use api::Engine;
pub use attack::{COMBO_TABLE, PERFECT_CLEAR_ATTACK, attack_lines};
pub use bit_board::{BitBoard, ColumnView, Occupancy};
pub use board::{Board, CellKind};
pub use game_over::{is_block_out, is_lock_out, is_top_out};
pub use garbage::GarbageBatch;
pub use generator::PieceGenerator;
pub use goals::{
    GoalProgress, GoalSystem, breaks_back_to_back, fixed_goal_for_level, goal_for_level,
    qualifies_for_back_to_back, variable_goal_for_level, variable_goal_units,
};
pub use gravity::{MAX_LEVEL, MIN_LEVEL, fall_speed_seconds, soft_drop_speed_seconds};
pub use lock_clear::{LockOutcome, lock_and_clear};
pub use lock_down::{
    EXTENDED_LOCK_RESET_BUDGET, LOCK_DOWN_SECONDS, LockDownMode, apply_grounded_move_or_rotation,
};
pub use pieces::{MoveDirection, Piece, PieceRotation, PieceType};
pub use scoring::EngineScoreAction;
pub use t_spin::{TSpinCorners, TSpinKind, classify_t_spin, is_t_slot, t_spin_corners};
pub use types::{
    ActivePieceSnapshot, BUFFER_HEIGHT, EngineConfig, EngineEvent, EngineSnapshot, GameOverStatus,
    InputFrame, SnapshotCell,
};
