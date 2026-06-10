//! Incoming-garbage rules for versus play: the pending queue, cancellation, and
//! rising.
//!
//! Guideline versus is an exchange of garbage lines: a clear *sends* attack
//! ([`attack_lines`](super::attack_lines)), and an opponent's attack arrives
//! here as **pending** garbage — queued, visible to the player, but not yet on
//! the board. Three rules govern the queue, and the engine owns all of them so
//! every surface (headless versus, a future versus UI, netplay) gets identical
//! behavior:
//!
//! 1. **Cancellation (offset).** Attack you send first cancels your own pending
//!    garbage line-for-line, **oldest batch first**; only the remainder leaves
//!    the board as [`EngineEvent::AttackSent`](super::EngineEvent::AttackSent).
//! 2. **Rising.** Pending garbage enters the board after a lock that cleared
//!    **no** lines (clearing defers entry — the window in which cancellation can
//!    still save you), capped per lock by
//!    [`EngineConfig::garbage_cap`](super::EngineConfig::garbage_cap). A batch
//!    split by the cap keeps its hole column: it is the same attack, entering in
//!    two steps.
//! 3. **Holes.** Each queued batch gets one hole column drawn from the
//!    *receiver's* own seeded stream at queue time — self-contained determinism:
//!    a `(seed, queued-attack sequence)` fully reproduces a board, with the
//!    stream salted so it can never align with the piece generator's.
//!
//! The queue itself never touches the board; [`Engine`](super::Engine) applies
//! rising batches through the same `Board::insert_garbage_lines` primitive the
//! out-of-band harness seam uses.

use std::collections::VecDeque;

use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

/// Decorrelates the hole stream from the piece generator: both are seeded from
/// the engine seed, and identical streams would let a player predict holes from
/// the bag (or vice versa).
const HOLE_SALT: u64 = 0x6172_6261_6765_5F68; // "garbage_h", truncated

/// One queued attack: `lines` garbage rows sharing a single `hole_col`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct GarbageBatch {
    pub lines: u32,
    pub hole_col: usize,
}

/// The pending-garbage queue plus the receiver-owned hole stream.
pub(crate) struct PendingGarbage {
    /// FIFO of queued batches, oldest at the front — cancellation and rising
    /// both consume from the front (oldest attack lands or is offset first).
    batches: VecDeque<GarbageBatch>,
    /// Seeded hole stream; advanced once per queued batch.
    rng: StdRng,
}

impl PendingGarbage {
    pub(crate) fn new(engine_seed: u64) -> Self {
        Self {
            batches: VecDeque::new(),
            rng: StdRng::seed_from_u64(engine_seed ^ HOLE_SALT),
        }
    }

    /// Total pending lines (what a versus UI shows as the incoming meter).
    pub(crate) fn total(&self) -> u32 {
        self.batches.iter().map(|b| b.lines).sum()
    }

    /// Queue an incoming attack of `lines`, drawing its hole column from the
    /// receiver's stream. A zero-line attack queues nothing (and draws nothing,
    /// so no-op calls cannot perturb the hole sequence).
    pub(crate) fn queue(&mut self, lines: u32, board_width: usize) {
        if lines == 0 {
            return;
        }
        let hole_col = self.rng.random_range(0..board_width.max(1));
        self.batches.push_back(GarbageBatch { lines, hole_col });
    }

    /// Cancel pending garbage with `attack` lines of outgoing attack, oldest
    /// batch first, line-for-line. Returns the attack left over after
    /// cancellation — the lines that actually leave the board.
    pub(crate) fn cancel(&mut self, mut attack: u32) -> u32 {
        while attack > 0 {
            let Some(front) = self.batches.front_mut() else {
                break;
            };
            let cancelled = front.lines.min(attack);
            front.lines -= cancelled;
            attack -= cancelled;
            if front.lines == 0 {
                self.batches.pop_front();
            }
        }
        attack
    }

    /// Take the batches that rise after a clear-less lock: oldest first, at most
    /// `cap` total lines. A batch split by the cap leaves its remainder (same
    /// hole column) at the front of the queue.
    pub(crate) fn rise(&mut self, cap: u32) -> Vec<GarbageBatch> {
        let mut rising = Vec::new();
        let mut budget = cap;
        while budget > 0 {
            let Some(front) = self.batches.front_mut() else {
                break;
            };
            if front.lines <= budget {
                budget -= front.lines;
                rising.push(self.batches.pop_front().expect("front exists"));
            } else {
                front.lines -= budget;
                rising.push(GarbageBatch {
                    lines: budget,
                    hole_col: front.hole_col,
                });
                budget = 0;
            }
        }
        rising
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn queued(batches: &[(u32, usize)]) -> PendingGarbage {
        let mut q = PendingGarbage::new(7);
        for &(lines, hole) in batches {
            q.batches.push_back(GarbageBatch {
                lines,
                hole_col: hole,
            });
        }
        q
    }

    #[test]
    fn cancel_consumes_oldest_first_and_returns_leftover() {
        let mut q = queued(&[(3, 1), (4, 2)]);
        // 5 lines of attack: kills the 3-batch, eats 2 of the 4-batch.
        assert_eq!(q.cancel(5), 0);
        assert_eq!(q.total(), 2);
        assert_eq!(q.batches.front().unwrap().hole_col, 2);

        // 7 attack against the remaining 2: 5 lines leave the board.
        assert_eq!(q.cancel(7), 5);
        assert_eq!(q.total(), 0);
    }

    #[test]
    fn rise_respects_the_cap_and_splits_keeping_the_hole() {
        let mut q = queued(&[(3, 1), (6, 4)]);
        let rising = q.rise(5);
        // 3 from the first batch + 2 split off the second, hole preserved.
        assert_eq!(
            rising,
            vec![
                GarbageBatch {
                    lines: 3,
                    hole_col: 1
                },
                GarbageBatch {
                    lines: 2,
                    hole_col: 4
                },
            ]
        );
        // The remainder of the split batch stays queued, same hole.
        assert_eq!(q.total(), 4);
        assert_eq!(q.batches.front().unwrap().hole_col, 4);
    }

    #[test]
    fn holes_are_seed_deterministic_and_in_range() {
        let draw = |seed: u64| {
            let mut q = PendingGarbage::new(seed);
            (0..32)
                .map(|_| {
                    q.queue(1, 10);
                    q.batches.back().unwrap().hole_col
                })
                .collect::<Vec<_>>()
        };
        let a = draw(42);
        assert_eq!(a, draw(42), "same seed, same hole stream");
        assert_ne!(a, draw(43), "different seed diverges");
        assert!(a.iter().all(|&h| h < 10), "holes stay inside the board");
    }

    #[test]
    fn zero_line_queue_is_a_true_no_op() {
        let mut a = PendingGarbage::new(7);
        let mut b = PendingGarbage::new(7);
        a.queue(0, 10); // must not advance the hole stream
        a.queue(2, 10);
        b.queue(2, 10);
        assert_eq!(a.batches, b.batches, "a zero queue cannot perturb holes");
    }
}
