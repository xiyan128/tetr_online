//! Engine-agnostic Tetris core.
//!
//! Everything under this module implements the game rules as plain Rust with no
//! Bevy (or other engine) dependency, per ADR-7 — the host drives it through the
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

pub use active_piece::{ActivePiece, PieceAction, RotationDirection};
pub use api::{
    ActivePieceSnapshot, Engine, EngineConfig, EngineEvent, EngineSnapshot, GameOverStatus,
    InputFrame, SnapshotCell,
};
pub use attack::{attack_lines, COMBO_TABLE, PERFECT_CLEAR_ATTACK};
pub use bit_board::{BitBoard, ColumnView, Occupancy};
pub use board::{Board, CellKind};
pub use game_over::{is_block_out, is_lock_out, is_top_out};
pub use garbage::GarbageBatch;
pub use generator::PieceGenerator;
pub use goals::{
    breaks_back_to_back, fixed_goal_for_level, goal_for_level, qualifies_for_back_to_back,
    variable_goal_for_level, variable_goal_units, GoalProgress, GoalSystem,
};
pub use gravity::{fall_speed_seconds, soft_drop_speed_seconds, MAX_LEVEL, MIN_LEVEL};
pub use lock_clear::{lock_and_clear, LockOutcome};
pub use lock_down::{
    apply_grounded_move_or_rotation, LockDownMode, EXTENDED_LOCK_RESET_BUDGET, LOCK_DOWN_SECONDS,
};
pub use pieces::{MoveDirection, Piece, PieceRotation, PieceType};
pub use scoring::EngineScoreAction;
pub use t_spin::{classify_t_spin, is_t_slot, t_spin_corners, TSpinCorners, TSpinKind};
