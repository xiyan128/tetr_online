//! The AI player (roadmap Milestone M2 / ADR-4).
//!
//! This module is the bot that drives an `Engine` through the same
//! [`PlayerController`](crate::player::PlayerController) seam the keyboard uses:
//! it reads an [`EngineSnapshot`](crate::engine::EngineSnapshot), searches for a
//! placement, and emits [`InputFrame`](crate::engine::InputFrame)s.
//!
//! # Determinism & the engine boundary
//!
//! The search *core* (this module's [`state`], and the evaluator / movegen /
//! search added in later AI3.x tasks) is **pure Rust with no Bevy imports** so it
//! stays deterministic and unit-testable exactly like [`crate::engine`]. It never
//! touches the engine's piece generator RNG; any randomness the AI needs (tie
//! breaking, error injection) lives behind an AI-owned seeded RNG in the
//! controller layer, never here. The core simulates placements by reusing the
//! engine's own board primitives ([`lock_and_clear`](crate::engine::lock_and_clear)),
//! so it can never disagree with the real rules.

pub mod eval;
pub mod movegen;
pub mod state;

pub use eval::{Evaluator, LinearEvaluator, Reward, Value, Weights};
pub use movegen::{generate, generate_with_hold, Move, Placement};
pub use state::{BagState, SearchState};
