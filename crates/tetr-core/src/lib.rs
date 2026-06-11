//! `tetr-core` — the engine-agnostic core of tetr_online.
//!
//! This crate is the engine boundary made into a crate boundary: a pure,
//! deterministic Tetris rule [`engine`], a [`player`] controller abstraction, and
//! an [`ai`] bot — all with **no Bevy types**. The rule "the engine carries no
//! rendering or Bevy types" is enforced by the compiler, not by convention:
//! this crate does not depend on Bevy (except an optional, off-by-
//! default `bevy` feature that only adds a keyboard-input *adapter*).
//!
//! Two hosts drive it through the same plain-data contract — `Engine::step` /
//! `Engine::snapshot`, `PlayerController::poll`, and the pure `drive_engine` helper:
//!
//! - the **Bevy game** (`tetr_online`), which renders, plays audio, and owns menus;
//! - the **`tetr-embed`** wasm component, which renders snapshots to a canvas in the
//!   browser and weighs a few hundred KB instead of the game's ~14 MB.
//!
//! Because the AI is just another `PlayerController`, the embed gets autoplay for
//! free: drive the engine with an [`ai::AiController`] instead of the keyboard.

pub mod ai;
pub mod engine;
pub mod player;
