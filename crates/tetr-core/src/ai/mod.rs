//! The AI player (roadmap Milestone M2 / ADR-4).
//!
//! This module is the bot that drives an `Engine` through the same
//! [`PlayerController`](crate::player::PlayerController) seam the keyboard uses:
//! it reads an [`EngineSnapshot`](crate::engine::EngineSnapshot), searches for a
//! placement, and emits [`InputFrame`](crate::engine::InputFrame)s.
//!
//! # Determinism & the engine boundary
//!
//! The search *core* — [`state`], [`eval`], [`movegen`], [`search`], [`plan`],
//! and even the [`controller`] and [`runner`] — is **pure Rust with no Bevy
//! imports**, so it stays deterministic and unit-testable exactly like
//! [`crate::engine`]. It never touches the engine's piece generator RNG; the only
//! randomness the AI needs (tie-breaking, error injection) lives behind an
//! AI-owned seeded [`StdRng`](rand::rngs::StdRng) in the [`controller`], never the
//! engine's. The core simulates placements by reusing the engine's own board
//! logic (`BitBoard` mirrors the engine's collision + lock/clear bit-for-bit), so it can never
//! disagree with the real rules.
//!
//! # Layers
//!
//! - [`state`] (AI3.1) — cheap cloneable search state from a snapshot.
//! - [`eval`] (AI3.2) — the `(Value, Reward)` evaluator seam.
//! - [`movegen`] + [`search`] (AI3.3) — reachable placements + the anytime
//!   search session ([`Mind`]).
//! - [`plan`] (AI3.4) — placement → [`InputFrame`](crate::engine::InputFrame)s.
//! - [`runner`] + [`controller`] (AI3.5) — the compute seam and the
//!   [`AiController`] (a [`PlayerController`](crate::player::PlayerController)).
//!
//! The Bevy-aware "Watch AI" sandbox (AI3.6) that drove a gameplay session with the
//! bot lives in the game crate now (`tetr_online::ai::sandbox`), outside this
//! engine-agnostic core — keeping `tetr-core` free of Bevy.

pub mod controller;
pub mod eval;
pub mod handicap;
pub mod movegen;
pub mod plan;
pub mod policy;
pub mod runner;
pub mod search;
pub mod state;

pub use controller::{AiController, DEFAULT_AI_SEED};
pub use eval::{
    Cc2Evaluator, Cc2Weights, EvalContext, Evaluator, LinearEvaluator, Reward, Value, Weights,
};
pub use handicap::Handicap;
pub use movegen::{generate, generate_with_hold, Move, Placement};
pub use plan::placement_to_inputs;
pub use policy::{Decision, Observation, Policy, SearchPolicy};
pub use runner::{DecisionRunner, SlicedRunner, SyncRunner};
pub use search::{
    think_to_completion, BeamPlanner, BestFirstPlanner, GreedyPlanner, Mind, PlacementPlan,
    SearchBudget, ThinkProgress,
};
pub use state::{BagState, SearchState};
