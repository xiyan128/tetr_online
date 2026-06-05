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

/// [`env_or`] specialized to `f32` (step sizes, jitter magnitudes).
pub fn env_f32(key: &str, default: f32) -> f32 {
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

    /// The next 64-bit output.
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// The next float in `[0, 1)` (24-bit mantissa resolution).
    pub fn next_unit(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
}
