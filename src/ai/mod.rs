//! Game-side AI module.
//!
//! The engine-agnostic AI (the bot, its search, evaluator, and the model-agnostic
//! [`AiController`]) lives in the `tetr-core` crate now; it is re-exported here so
//! the host's existing `crate::ai::…` paths keep resolving unchanged. The only
//! Bevy-aware piece — the [`sandbox`] "Watch AI" integration — stays in this crate.

pub use tetr_core::ai::*;

/// The Watch-AI model registry (which "brain" the sandbox runs). Engine-agnostic
/// data; the picker screen and the sandbox both read it.
pub mod registry;
pub use registry::ModelRegistry;

// `pub` so the level scheduler can name `crate::ai::sandbox::sandbox_active` /
// `step_engine_ai` directly (it gates AI-vs-keyboard systems on them); the common
// types are also re-exported flat for the menus.
pub mod sandbox;
pub use sandbox::{AiPlayer, AiSandbox, AiSandboxPlugin};
