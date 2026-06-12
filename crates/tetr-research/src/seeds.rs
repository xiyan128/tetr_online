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
//!
//! # Campaigns: freshness across runs, not just within one
//!
//! The static map keeps one *run* honest; it cannot keep a *researcher* — or
//! an autonomous agent — honest across runs. Every look at a fixed
//! validation region followed by a design change leaks a few bits into it,
//! and over a long optimization campaign the region quietly becomes training
//! data. A [`Campaign`] therefore owns a private slab of the index space,
//! derived deterministically from its name: fresh validation / anchor /
//! promotion / rotation / confirmation sub-regions per campaign, disjoint
//! from the static map by construction and from other campaigns up to the
//! slot-collision odds documented on [`Campaign::derive`]. Re-using a
//! campaign name deliberately resumes that campaign's regions; that is the
//! feature, not the hazard — the hazard is iterating against seeds a
//! previous decision already saw, which fresh names make cheap to avoid.
//!
//! [`regions::FINAL`] sits above every campaign slab: the one region nothing
//! reads during iteration, reserved for a last, never-tuned-against verdict
//! before an external claim. Tooling must gate any read of it behind an
//! explicit, recorded opt-in (the `promote` bin's `FINAL_VALIDATION`).

use crate::cli::SplitMix64;

// The region map needs a 64-bit index space (CONFIRM = 1<<50); fail loudly
// with a reason on any 32-bit target instead of overflowing usize.
const _: () = assert!(
    usize::BITS >= 64,
    "tetr-research's seed regions need 64-bit usize"
);

/// The crate's seed-index partition. Regions are starting indices into the
/// [`seed_set_from`] stream; each consumer documents its stride so regions
/// can be audited for overlap at a glance.
///
/// Fixed-size regions sit low; the two regions that GROW with iteration count
/// (the climb's rotation blocks and its per-accept confirmations) get
/// power-of-two starts with explicit headroom, because writing the old map
/// down exposed a latent collision: rotation at `8192 + iter × 24` walked
/// into the SPRT region after ~340 iterations — never hit in the recorded
/// runs (≤127 iters), but one overnight climb away. Headroom now: rotation
/// reaches CONFIRM after (2^50 − 2^20)/24 ≈ 4.7×10^13 iterations at the
/// default block size, and confirmations stay below the campaign space
/// because the climb's u32 iteration counter caps their growth at
/// 2^50 + 2^32×4096 = 2^50 + 2^44 < 2^51 (see [`Campaign`]).
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
    /// The never-iterated final region: NOTHING reads it during optimization,
    /// review, or promotion practice — it backs exactly one verdict per
    /// external claim, after which the claim cites it and the discipline
    /// resets with the next claim. Tooling gates it behind an explicit,
    /// recorded opt-in; quoting a number from here in any tuning loop is the
    /// one unrecoverable way to spend it. (Stride: small fixed suites only.)
    pub const FINAL: usize = 1 << 63;
}

/// Campaign-slab geometry: the space `[2^51, 2^51 + 2^30·2^32)` holds 2^30
/// slabs of 2^32 indices, starting above the static map's worst-case growth
/// (CONFIRM tops out below 2^51 — see [`regions`]) and ending below
/// [`regions::FINAL`] at 2^63.
const SLAB_BASE: usize = 1 << 51;
const SLAB_SIZE: usize = 1 << 32;
const SLAB_SLOTS: usize = 1 << 30;

// Sub-region offsets inside a slab. Fixed-size regions sit low, the two
// growing regions high, mirroring the static map's layout logic. Capacities
// (documented on the accessors) bind at ~524k climb iterations per campaign
// — weeks of continuous climbing; the accessors fail loudly at the bound.
const VALIDATION_OFF: usize = 0;
const VALIDATION_END: usize = 1 << 16;
const ANCHOR_OFF: usize = 1 << 16;
const ANCHOR_END: usize = 1 << 24;
const PROMOTE_OFF: usize = 1 << 24;
const PROMOTE_END: usize = 1 << 25;
const ROTATION_OFF: usize = 1 << 25;
const ROTATION_END: usize = 1 << 31;
const CONFIRM_OFF: usize = 1 << 31;
const CONFIRM_END: usize = SLAB_SIZE;

// The slab space must clear the static map's worst-case growth below and
// FINAL above, and the sub-regions must tile a slab in order; a layout edit
// that breaks any of this fails here, not in a run.
const _: () = assert!(SLAB_BASE >= regions::CONFIRM + (1 << 44));
const _: () = assert!(SLAB_BASE + SLAB_SLOTS * SLAB_SIZE <= regions::FINAL);
const _: () = assert!(VALIDATION_END <= ANCHOR_OFF);
const _: () = assert!(ANCHOR_END <= PROMOTE_OFF);
const _: () = assert!(PROMOTE_END <= ROTATION_OFF);
const _: () = assert!(ROTATION_END <= CONFIRM_OFF);
const _: () = assert!(CONFIRM_END <= SLAB_SIZE);

/// A named optimization campaign and its private slab of seed-index space.
///
/// Everything an agent runs while pursuing one goal — climbs, anchor races,
/// validation, promotion — draws from the campaign's sub-regions, so no seed
/// that influenced any decision in the campaign is ever re-used to judge it,
/// and no two campaigns share seeds to iterate against. The id is part of
/// the run record: a result is reproducible from `(code, env, campaign)`.
#[derive(Clone, Debug)]
pub struct Campaign {
    pub id: String,
    /// The slab index in `[0, 2^30)` — derived, recorded for auditability.
    pub slot: usize,
}

impl Campaign {
    /// Derive a campaign's slab from its name: FNV-1a 64 over the bytes,
    /// finished through one SplitMix64 step for avalanche, reduced mod 2^30.
    ///
    /// FROZEN: changing this mapping re-homes every recorded campaign; it is
    /// pinned by the `derive_is_frozen` canary test. Collisions: two distinct
    /// names share a slab with probability ~C²/2^31 over C campaigns (~0.05%
    /// at a thousand campaigns) — acceptable because a collision costs
    /// freshness, not correctness, and the recorded `slot` makes one
    /// auditable after the fact.
    pub fn derive(id: &str) -> Self {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for b in id.bytes() {
            h ^= u64::from(b);
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
        let slot = (SplitMix64::new(h).next_u64() % SLAB_SLOTS as u64) as usize;
        Self {
            id: id.to_string(),
            slot,
        }
    }

    fn slab(&self) -> usize {
        SLAB_BASE + self.slot * SLAB_SIZE
    }

    /// Held-out validation base (capacity 2^16 seeds).
    pub fn validation(&self, count: usize) -> usize {
        assert!(
            count <= VALIDATION_END - VALIDATION_OFF,
            "campaign '{}': validation wants {count} seeds, capacity {}",
            self.id,
            VALIDATION_END - VALIDATION_OFF
        );
        self.slab() + VALIDATION_OFF
    }

    /// Base of anchor race `event` (stride chosen by the caller, like the
    /// climb's confirmations; capacity ~4080 events at the default 4096).
    pub fn anchor_base(&self, event: u32, stride: usize) -> usize {
        let off = ANCHOR_OFF + event as usize * stride;
        assert!(
            off + stride <= ANCHOR_END,
            "campaign '{}': anchor sub-slab exhausted (event {event}, stride {stride})",
            self.id
        );
        self.slab() + off
    }

    /// Promotion-suite base (capacity 2^24 indices — far above any panel).
    pub fn promote(&self, count: usize) -> usize {
        assert!(
            count <= PROMOTE_END - PROMOTE_OFF,
            "campaign '{}': promotion wants {count} seed indices, capacity {}",
            self.id,
            PROMOTE_END - PROMOTE_OFF
        );
        self.slab() + PROMOTE_OFF
    }

    /// Base of the climb's rotating screen block for `iter`
    /// (stride: one `block`-sized set per iteration).
    pub fn rotation_block(&self, iter: u32, block: usize) -> usize {
        let off = ROTATION_OFF + iter as usize * block;
        assert!(
            off + block <= ROTATION_END,
            "campaign '{}': rotation sub-slab exhausted (iter {iter}, block {block}) — \
             start a fresh campaign",
            self.id
        );
        self.slab() + off
    }

    /// Base of the climb's confirmation race for `iter`
    /// (stride: caller-owned, ≥ the race's worst-case seed consumption).
    pub fn confirm_base(&self, iter: u32, stride: usize) -> usize {
        let off = CONFIRM_OFF + iter as usize * stride;
        assert!(
            off + stride <= CONFIRM_END,
            "campaign '{}': confirmation sub-slab exhausted (iter {iter}, stride {stride}) — \
             start a fresh campaign",
            self.id
        );
        self.slab() + off
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    /// The slot mapping is part of every campaign's reproducibility contract;
    /// this canary pins it. If it ever fails, the hash changed — revert the
    /// change rather than updating the literal.
    #[test]
    fn derive_is_frozen() {
        assert_eq!(
            Campaign::derive("scratch").slot,
            Campaign::derive("scratch").slot
        );
        let pinned = Campaign::derive("garbage-aware-2026").slot;
        assert_eq!(pinned, FROZEN_SLOT, "campaign slot hash changed");
    }
    const FROZEN_SLOT: usize = 1_062_951_010;

    #[test]
    fn distinct_names_take_distinct_slots() {
        let names = [
            "scratch",
            "garbage-aware-2026",
            "nn-round-2",
            "mcgs",
            "downstack-v2",
        ];
        let mut slots: Vec<usize> = names.iter().map(|n| Campaign::derive(n).slot).collect();
        slots.sort_unstable();
        slots.dedup();
        assert_eq!(
            slots.len(),
            names.len(),
            "slot collision among canary names"
        );
    }

    /// A derived slab sits between the static map and FINAL (the sub-region
    /// tiling itself is compile-time asserted next to the layout constants).
    #[test]
    fn slab_layout_is_disjoint_and_bounded() {
        let c = Campaign::derive("layout-probe");
        let slab = c.validation(1);
        assert!(slab >= SLAB_BASE);
        assert!(slab + SLAB_SIZE <= regions::FINAL);
        // Static regions' worst-case growth stays below the slab space:
        // confirmations stride 4096 per u32 iteration at most.
        assert!(regions::CONFIRM + (u32::MAX as usize) * 4096 < SLAB_BASE);
    }

    #[test]
    fn sub_regions_of_one_campaign_are_disjoint() {
        let c = Campaign::derive("disjointness-probe");
        let slab_end = c.validation(1) + SLAB_SIZE;
        let val = c.validation(1 << 16);
        let anchor = c.anchor_base(0, 4096);
        let promote = c.promote(1 << 24);
        let rotation = c.rotation_block(0, 24);
        let confirm = c.confirm_base(0, 4096);
        assert!(val + (1 << 16) <= anchor);
        assert!(c.anchor_base(4079, 4096) + 4096 <= promote);
        assert!(promote + (1 << 24) <= rotation);
        assert!(rotation + 24 <= confirm);
        assert!(confirm + 4096 <= slab_end);
    }

    #[test]
    #[should_panic(expected = "rotation sub-slab exhausted")]
    fn rotation_exhaustion_is_loud() {
        Campaign::derive("exhaust").rotation_block(u32::MAX, 1 << 20);
    }

    #[test]
    #[should_panic(expected = "confirmation sub-slab exhausted")]
    fn confirm_exhaustion_is_loud() {
        Campaign::derive("exhaust").confirm_base(1 << 20, 4096);
    }

    #[test]
    fn seed_streams_from_disjoint_regions_share_nothing() {
        let a = seed_set_from(Campaign::derive("a").validation(32), 32);
        let b = seed_set_from(Campaign::derive("b").validation(32), 32);
        let statics = seed_set_from(regions::VALIDATION, 32);
        for s in &a {
            assert!(!b.contains(s));
            assert!(!statics.contains(s));
        }
    }
}
