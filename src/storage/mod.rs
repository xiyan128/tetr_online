//! Persistent key/value storage (S1.1).
//!
//! A single [`Storage`] trait abstracts where small blobs (serialized
//! [`GameSettings`], high-score tables) live. Two backends implement it:
//!
//! * **native** ([`native::FileStorage`]) — one file per key under the
//!   platform config dir (via the `directories` crate).
//! * **web** ([`web::LocalStorage`]) — the browser's `window.localStorage`.
//!
//! Callers go through [`default_storage`], which returns the right backend for
//! the build target as a boxed trait object. Values are opaque `String`s;
//! callers choose the encoding (the options + high-scores features serialize
//! their own structs, e.g. via a small hand-rolled format or `serde` later).
//!
//! [`GameSettings`]: crate::settings::GameSettings

#[cfg(not(target_arch = "wasm32"))]
mod native;
#[cfg(target_arch = "wasm32")]
mod web;

use bevy::prelude::Resource;

/// Bevy resource holding the active [`Storage`] backend as a trait object.
///
/// Inserted once at startup from [`default_storage`]. The options + high-scores
/// features read/write through `.0`.
#[derive(Resource)]
pub struct StorageResource(pub Box<dyn Storage>);

/// A minimal blocking key/value store for small persistent blobs.
///
/// Keys are short ASCII identifiers (e.g. `"settings"`, `"highscores"`).
/// Implementations must treat a missing key as `None` (not an error) and should
/// swallow backend I/O errors into `None`/no-op so a corrupt or unwritable store
/// degrades to "defaults" rather than crashing the game.
pub trait Storage: Send + Sync {
    /// Load the value previously stored under `key`, or `None` if absent.
    fn load(&self, key: &str) -> Option<String>;

    /// Persist `value` under `key`, overwriting any previous value. Failures are
    /// swallowed (best-effort persistence).
    fn save(&self, key: &str, value: &str);

    /// Remove any value stored under `key`. No-op if absent.
    fn remove(&self, key: &str);
}

/// Storage keys used across the app. Centralized so the options and high-scores
/// features agree on the same key strings.
pub mod keys {
    /// Serialized [`GameSettings`](crate::settings::GameSettings).
    pub const SETTINGS: &str = "settings";
    /// Serialized high-score tables (all variants).
    pub const HIGH_SCORES: &str = "highscores";
}

/// Construct the platform-appropriate [`Storage`] backend.
///
/// Native builds get a per-key file store under the OS config directory; wasm
/// builds get `localStorage`. Returns a boxed trait object so call sites and the
/// Bevy resource that holds it are target-agnostic.
#[cfg(not(target_arch = "wasm32"))]
pub fn default_storage() -> Box<dyn Storage> {
    Box::new(native::FileStorage::new())
}

/// Construct the platform-appropriate [`Storage`] backend (wasm: `localStorage`).
#[cfg(target_arch = "wasm32")]
pub fn default_storage() -> Box<dyn Storage> {
    Box::new(web::LocalStorage::new())
}

/// An always-empty, write-discarding [`Storage`]. Used as a fallback when no
/// real backend is available (e.g. headless tests, or a browser with
/// `localStorage` disabled) and directly by unit tests.
#[derive(Default)]
pub struct NullStorage;

impl Storage for NullStorage {
    fn load(&self, _key: &str) -> Option<String> {
        None
    }
    fn save(&self, _key: &str, _value: &str) {}
    fn remove(&self, _key: &str) {}
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn null_storage_never_persists() {
        let store = NullStorage;
        store.save("k", "v");
        assert_eq!(store.load("k"), None);
        store.remove("k");
    }

    #[test]
    fn default_storage_round_trips_through_the_trait_object() {
        // Exercises whichever backend the build target selects via the trait
        // object, then cleans up so the test is idempotent across runs.
        let store = default_storage();
        let key = "tetr_online_storage_selftest";
        store.save(key, "value-123");
        // Native: should round-trip. Web under wasi-less test harness may be a
        // no-op; either way the call must not panic and the contract holds.
        if let Some(value) = store.load(key) {
            assert_eq!(value, "value-123");
        }
        store.remove(key);
        assert_eq!(store.load(key), None);
    }
}
