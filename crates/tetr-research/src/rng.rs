//! The crate's dependency-free deterministic PRNG.
//!
//! Everything that jitters or samples (hill-climbs, simulation tests, cheese
//! holes) draws from this one generator, seeded explicitly — no `rand`
//! dependency, so streams are stable across toolchain and dependency bumps
//! and every consumer replays bit-for-bit from its seed.

/// A tiny deterministic [SplitMix64](https://prng.di.unimi.it/splitmix64.c) PRNG.
///
/// The seed is the running state and the standard increment is folded in on
/// the first [`next_u64`](Self::next_u64) — identical to the per-bin
/// free-function form it replaced (`SplitMix64::new(s).next_u64()` == old
/// `next_u64(&mut s)`), so historical streams reproduce.
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
