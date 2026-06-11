//! Game-side AI module.
//!
//! The engine-agnostic AI (the bot, its search, evaluator, and the model-agnostic
//! [`AiController`]) lives in the `tetr-core` crate; it is re-exported here so the
//! host addresses it as `crate::ai::…`. The Bevy-side piece is the model
//! [`registry`]: the catalog of bots a session seat can run.

pub use tetr_core::ai::*;

/// The Watch-AI model registry (which "brain" a bot seat runs). Engine-agnostic
/// data; the setup screens and the session's seat spawner both read it.
pub mod registry;
pub use registry::ModelRegistry;
