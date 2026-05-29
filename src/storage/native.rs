//! Native [`Storage`] backend: one file per key under the OS config dir.
//!
//! Uses the `directories` crate to locate a per-app config directory
//! (`~/.config/tetr_online` on Linux, `~/Library/Application Support/…` on
//! macOS, `%APPDATA%\…` on Windows). Each key maps to a `<key>.dat` file. All
//! I/O is best-effort: errors degrade to `None`/no-op so a read-only or missing
//! directory never crashes the game.

use std::fs;
use std::path::PathBuf;

use directories::ProjectDirs;

use super::Storage;

/// File-backed key/value store rooted at the app config directory.
pub struct FileStorage {
    /// `None` if the platform config dir could not be resolved; every operation
    /// then degrades to a no-op (in-memory-less, like [`super::NullStorage`]).
    root: Option<PathBuf>,
}

impl FileStorage {
    pub fn new() -> Self {
        let root = ProjectDirs::from("dev", "tetr_online", "tetr_online")
            .map(|dirs| dirs.config_dir().to_path_buf());
        Self { root }
    }

    /// Map a logical key to its on-disk path, ensuring the parent dir exists.
    /// Returns `None` if there is no root or the dir cannot be created.
    fn path_for(&self, key: &str) -> Option<PathBuf> {
        let root = self.root.as_ref()?;
        if fs::create_dir_all(root).is_err() {
            return None;
        }
        Some(root.join(format!("{key}.dat")))
    }
}

impl Default for FileStorage {
    fn default() -> Self {
        Self::new()
    }
}

impl Storage for FileStorage {
    fn load(&self, key: &str) -> Option<String> {
        let path = self.path_for(key)?;
        fs::read_to_string(path).ok()
    }

    fn save(&self, key: &str, value: &str) {
        if let Some(path) = self.path_for(key) {
            let _ = fs::write(path, value);
        }
    }

    fn remove(&self, key: &str) {
        if let Some(path) = self.path_for(key) {
            let _ = fs::remove_file(path);
        }
    }
}
