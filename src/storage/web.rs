//! Web [`Storage`] backend: the browser's `window.localStorage`.
//!
//! Gated to `target_arch = "wasm32"`. All operations are best-effort: if the
//! window or `localStorage` is unavailable (private mode, disabled storage),
//! reads return `None` and writes are dropped, so the game falls back to
//! defaults instead of crashing.

use super::Storage;

/// `localStorage`-backed key/value store.
pub struct LocalStorage;

impl LocalStorage {
    pub fn new() -> Self {
        Self
    }

    /// Resolve `window.localStorage`, or `None` if unavailable.
    fn store() -> Option<web_sys::Storage> {
        web_sys::window()?.local_storage().ok().flatten()
    }
}

impl Default for LocalStorage {
    fn default() -> Self {
        Self::new()
    }
}

impl Storage for LocalStorage {
    fn load(&self, key: &str) -> Option<String> {
        Self::store()?.get_item(key).ok().flatten()
    }

    fn save(&self, key: &str, value: &str) {
        if let Some(store) = Self::store() {
            let _ = store.set_item(key, value);
        }
    }

    fn remove(&self, key: &str) {
        if let Some(store) = Self::store() {
            let _ = store.remove_item(key);
        }
    }
}
