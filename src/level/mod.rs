//! The session's configuration seam and shared render vocabulary.
//!
//! Everything a session reads but does not own: [`common`] (colors, the
//! camera marker, audio cues, `LevelConfig`), [`engine_bridge`] (the
//! `LevelConfig`/`GameSettings`/`Variant` → `EngineConfig` seam plus the
//! input-latch types), and [`sound_effects`] as the audio sink for the
//! session's `AudioCue` triggers. The session itself (engines, seats,
//! rendering, overlays) lives in `src/session/`.

pub(crate) mod common;
pub(crate) mod engine_bridge;
pub(crate) mod sound_effects;

#[allow(unused_imports)]
pub use engine_bridge::SIM_DT_SECONDS;
