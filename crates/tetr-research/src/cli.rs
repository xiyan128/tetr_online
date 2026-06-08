//! Shared config + RNG helpers for the research `bin/` tools.
//!
//! Every bin is configured purely by environment variables — so a run is a function
//! of its env plus the engine seed, reproducible with no flags to thread — and the
//! hill-climbers need a dependency-free deterministic PRNG. Both were copy-pasted
//! into each bin (where the combo-bug-style desync risk lives); this is their single
//! home.

use std::str::FromStr;

/// Parse environment variable `key` as `T`, falling back to `default` when it is
/// unset or fails to parse. The bins' one knob-reading primitive.
pub fn env_or<T: FromStr>(key: &str, default: T) -> T {
    std::env::var(key)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

/// [`env_or`] specialized to `usize` (the common case: seed counts, depths, widths).
pub fn env_usize(key: &str, default: usize) -> usize {
    env_or(key, default)
}

/// A tiny deterministic [SplitMix64](https://prng.di.unimi.it/splitmix64.c) PRNG —
/// the hill-climbers' mutation / jitter source. No `rand` dependency and fully
/// reproducible from the seed, so a climb replays bit-for-bit.
///
/// The seed is the running state and the standard increment is folded in on the
/// first [`next_u64`](Self::next_u64) — identical to the per-bin free-function form
/// it replaced (`SplitMix64::new(s).next_u64()` == old `next_u64(&mut s)`), so a
/// refactored climb produces the same sequence.
pub struct SplitMix64(u64);

impl SplitMix64 {
    /// Seed the generator.
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }

    /// Wrap a bare running-state word as a generator — the inverse of
    /// [`into_raw`](Self::into_raw). Identical to [`new`](Self::new) (both store the
    /// word as the running state folded forward on the next [`next_u64`](Self::next_u64));
    /// the distinct name documents intent at call sites that thread a raw `u64` PRNG
    /// state through `&mut u64` rather than seeding a fresh stream.
    pub fn from_raw(state: u64) -> Self {
        Self(state)
    }

    /// Unwrap the running-state word — the inverse of [`from_raw`](Self::from_raw) — so
    /// a caller holding a bare `u64` can read the advanced state back after stepping.
    pub fn into_raw(self) -> u64 {
        self.0
    }

    /// The next 64-bit output.
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}
