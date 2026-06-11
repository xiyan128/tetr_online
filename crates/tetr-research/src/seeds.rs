//! Deterministic seed sets and the crate's seed-region discipline.
//!
//! Every experiment draws its game seeds from a deterministic stream over
//! *indices*, so a result is reproducible from `(code, env, region, count)`
//! alone. Disjoint index ranges give disjoint seed sets — the foundation of
//! the train / validation / confirmation separation that keeps verdicts
//! honest.

use crate::cli::SplitMix64;

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
