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
//! primitives ([`lock_and_clear`](crate::engine::lock_and_clear)), so it can never
//! disagree with the real rules.
//!
//! # Layers
//!
//! - [`state`] (AI3.1) — cheap cloneable search state from a snapshot.
//! - [`eval`] (AI3.2) — the `(Value, Reward)` evaluator seam.
//! - [`movegen`] + [`search`] (AI3.3) — reachable placements + the planner.
//! - [`plan`] (AI3.4) — placement → [`InputFrame`](crate::engine::InputFrame)s.
//! - [`runner`] + [`controller`] + [`difficulty`] (AI3.5) — the compute seam, the
//!   [`AiController`] (a [`PlayerController`](crate::player::PlayerController)),
//!   and the difficulty knobs.
//! - [`sandbox`] (AI3.6) — the **only** Bevy-aware AI module: a "Watch AI"
//!   gameplay session that drives the engine with the [`AiController`] through the
//!   existing renderer, for watching/tuning the bot.

pub mod controller;
pub mod handicap;
pub mod eval;
pub mod movegen;
pub mod plan;
pub mod policy;
pub mod runner;
pub mod sandbox;
pub mod search;
pub mod state;

pub use controller::{AiController, DEFAULT_AI_SEED};
pub use handicap::Handicap;
pub use eval::{Evaluator, LinearEvaluator, Reward, Value, Weights};
pub use movegen::{generate, generate_with_hold, Move, Placement};
pub use plan::placement_to_inputs;
pub use policy::{Decision, Observation, Policy, SearchPolicy};
pub use runner::{DecisionRunner, SyncRunner};
pub use sandbox::{AiPlayer, AiSandbox, AiSandboxPlugin};
pub use search::{GreedyPlanner, PlacementPlan, Planner, PlannerStep, SearchBudget};
pub use state::{BagState, SearchState};
