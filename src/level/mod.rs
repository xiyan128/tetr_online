//! The session's configuration seam and shared render vocabulary.
//!
//! Historically this module owned the whole single-player pipeline (its own
//! engine driver, reconcilers, HUD, and game-over flow). That pipeline was
//! re-homed onto the seat architecture in `src/session/` — single-player is a
//! one-seat session now — and what remains here is what every session reads:
//! [`common`] (colors, the camera marker, audio cues, `LevelConfig`) and
//! [`engine_bridge`] (the `LevelConfig`/`GameSettings`/`Variant` →
//! `EngineConfig` seam plus the input-latch types), with [`sound_effects`] as
//! the audio sink for the session's `AudioCue` triggers.

pub(crate) mod common;
pub(crate) mod engine_bridge;
pub(crate) mod sound_effects;

#[allow(unused_imports)]
pub use engine_bridge::SIM_DT_SECONDS;
