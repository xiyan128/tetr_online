//! Deterministic seed sets and the crate's seed-region discipline.
//!
//! Every experiment draws its game seeds from a deterministic stream over
//! *indices*, so a result is reproducible from `(code, env, region, count)`
//! alone. Disjoint index ranges give disjoint seed sets — the foundation of
//! the train / validation / confirmation separation that keeps verdicts
//! honest.
//!
//! # The region map
//!
//! The index space is partitioned by purpose ([`regions`]). The rule: **a
//! number may only be reported on seeds that did not influence any decision
//! that produced it.** Training selects, validation checks, confirmation
//! proves — three different regions, never shared. A new experiment that
//! needs fresh seeds claims a new region constant here rather than inventing
//! an offset inline.

use crate::cli::SplitMix64;

/// The crate's seed-index partition. Regions are starting indices into the
/// [`seed_set_from`] stream; each consumer documents its stride so regions
/// can be audited for overlap at a glance.
///
/// Fixed-size regions sit low; the two regions that GROW with iteration count
/// (the climb's rotation blocks and its per-accept confirmations) get
/// power-of-two starts with explicit headroom, because writing the old map
/// down exposed a latent collision: rotation at `8192 + iter × 24` walked
/// into the SPRT region after ~340 iterations — never hit in the recorded
/// runs (≤127 iters), but one overnight climb away. Headroom now:
/// rotation reaches CONFIRM after ~44 billion iterations at default block
/// size; confirmations never collide below ~10^14 iterations.
pub mod regions {
    /// Training / screening seeds (the climb's fixed-seed mode, quick A/Bs).
    pub const TRAIN: usize = 0;
    /// Held-out validation — the honest verdict after an optimization run.
    pub const VALIDATION: usize = 4096;
    /// The standalone SPRT racer (`versus_sprt`; stride: one block per SPRT
    /// block, bounded by the race length — well under the next region).
    pub const SPRT: usize = 16384;
    /// The climb's per-iteration rotating screen blocks
    /// (stride: one `SEEDS`-sized block per iteration).
    pub const ROTATION: usize = 1 << 20;
    /// The climb's per-accept SPRT confirmations (stride: 4096 per iteration).
    pub const CONFIRM: usize = 1 << 50;
}

/// A deterministic, well-distributed set of `count` seeds (SplitMix64 over indices).
pub fn seed_set(count: usize) -> Vec<u64> {
    seed_set_from(0, count)
}

/// Like [`seed_set`] but over indices `start..start+count` — for **disjoint**
/// train / held-out validation seed sets (`seed_set(n)` and `seed_set_from(s, n)`
/// share no seeds when `s >= n`), so a hillclimb can be checked for overfitting.
pub fn seed_set_from(start: usize, count: usize) -> Vec<u64> {
    // Per-index SplitMix64 seeding: `new(i).next_u64()` reproduces the old inline fold
    // (`new` stores `i`, then `next_u64` folds in the golden increment) bit-for-bit.
    (start as u64..(start + count) as u64)
        .map(|i| SplitMix64::new(i).next_u64())
        .collect()
}
