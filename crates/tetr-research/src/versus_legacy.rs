//! The **legacy** harness-side garbage scheduler — quarantined on purpose.
//!
//! Before the engine owned the versus rules (see `docs/adr-versus-rules.md`),
//! this module's `GarbageQueue` *was* the versus implementation. It survives
//! for exactly two consumers, both of which would be invalidated by switching
//! them to the engine path:
//!
//! 1. **Scripted pressure scenarios** ([`crate::behavior`]'s Faucet): inject
//!    garbage on a schedule, with the harness doing the bookkeeping.
//! 2. **The TBP referee** (`cc2_baseline`): Cold Clear 2 runs as an external
//!    process with no garbage messages, so the referee inserts raw and keeps
//!    cancellation accounting outside both engines.
//!
//! NOTE the deliberate divergence from the engine rules: this queue settles
//! the OLDEST garbage lowest ([`GarbageQueue::drain_newest_first`]), while the
//! engine's chronological rising leaves the NEWEST batch lowest — boards from
//! the two paths are not comparable for multi-batch deliveries, and
//! `cc2_baseline` win rates are not like-for-like with
//! [`play_versus`](crate::versus::play_versus) (the referee also keeps the old
//! wholesale dump timing). Changing either would invalidate every recorded
//! CC2 baseline. New experiments should use [`crate::versus`].

use std::collections::VecDeque;

use tetr_core::engine::Engine;
use tetr_core::player::PlayerController;

use crate::accounting::controller_seed;
use crate::marathon::marathon_config;
use crate::rng::SplitMix64;
use crate::versus::versus_step_piece;

/// Garbage queued against a player: a FIFO of `(lines, hole_col)` batches, one per
/// un-cancelled opponent attack. Your own clears cancel the oldest batches first;
/// whatever you fail to cancel is dumped onto your board.
#[derive(Default)]
pub struct GarbageQueue {
    batches: VecDeque<(u32, usize)>,
}

impl GarbageQueue {
    /// Total garbage lines currently queued.
    pub fn pending(&self) -> u32 {
        self.batches.iter().map(|&(n, _)| n).sum()
    }

    pub fn push(&mut self, lines: u32, hole: usize) {
        if lines > 0 {
            self.batches.push_back((lines, hole));
        }
    }

    /// Cancel up to `attack` lines from the front; return the un-cancelled remainder.
    pub fn cancel(&mut self, mut attack: u32) -> u32 {
        while attack > 0 {
            let Some(front) = self.batches.front_mut() else {
                break;
            };
            let c = attack.min(front.0);
            front.0 -= c;
            attack -= c;
            if front.0 == 0 {
                self.batches.pop_front();
            }
        }
        attack
    }

    /// Remove all queued batches, newest first — so a caller inserting them one by
    /// one (each landing at the bottom) settles the oldest garbage lowest.
    pub fn drain_newest_first(&mut self) -> Vec<(u32, usize)> {
        let mut out = Vec::with_capacity(self.batches.len());
        while let Some(batch) = self.batches.pop_back() {
            out.push(batch);
        }
        out
    }

    /// Dump all queued garbage onto `engine` (newest batch first, see above).
    /// Returns true if the rising stack tops the player out.
    pub fn dump(&mut self, engine: &mut Engine) -> bool {
        let mut topped = false;
        for (lines, hole) in self.drain_newest_first() {
            topped |= engine.insert_garbage(lines as usize, hole);
        }
        topped
    }
}

/// Next seeded garbage-hole column (SplitMix64 over a per-match stream).
pub fn versus_hole(rng: &mut u64) -> usize {
    // Thread the caller's bare `u64` state through the shared SplitMix64 step: one
    // `next_u64` advances the word exactly as the inlined fold did, then write it back.
    let mut generator = SplitMix64::from_raw(*rng);
    let hole = (generator.next_u64() % 10) as usize;
    *rng = generator.into_raw();
    hole
}

/// Salt folding the match seed into the HARNESS-side garbage-hole RNG used by the
/// TBP referee (`cc2_baseline`), decorrelating hole placement from the
/// (same-seeded) piece stream. Engine-rules matches never draw holes here:
/// each receiver engine uses its own internal stream (tetr-core's garbage
/// module, its own salt).
pub const VERSUS_HOLE_SALT: u64 = 0xA5A5_5A5A_DEAD_BEEF;

/// One side of a versus match: an engine + its bot. Exposed so an external referee
/// (e.g. the Cold Clear 2 driver, which runs the opponent over TBP) can pit our bot
/// against another protocol bot using the same garbage rules as the recorded
/// CC2 baselines.
pub struct VersusEngine {
    engine: Engine,
    bot: Box<dyn PlayerController>,
}

impl VersusEngine {
    pub fn new(make_bot: &dyn Fn(u64) -> Box<dyn PlayerController>, seed: u64) -> Self {
        Self {
            engine: Engine::new(marathon_config(), seed),
            bot: make_bot(controller_seed(seed)),
        }
    }

    /// Place one piece; return `(attack produced, topped_out)`. The referee
    /// inserts garbage raw ([`receive`](Self::receive)), so this engine's
    /// pending queue stays empty and the attack reported here is gross — the
    /// referee does its own cancellation bookkeeping externally.
    pub fn step_piece(&mut self) -> (u32, bool) {
        versus_step_piece(&mut self.engine, &mut *self.bot)
    }

    /// Receive one garbage batch (`lines` rows, hole at `hole_col`); return true if
    /// it tops this player out.
    pub fn receive(&mut self, lines: u32, hole_col: usize) -> bool {
        self.engine.insert_garbage(lines as usize, hole_col)
    }
}
