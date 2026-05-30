//! The native off-thread decision runner (reference impl of the async seam).
//!
//! [`ThreadedRunner`] runs the policy on a worker thread and hands the
//! [`Decision`] back over a channel, so a frame-expensive decision never blocks the
//! game loop. Tier-1 greedy does **not** need this (it is microseconds —
//! [`SyncRunner`](super::SyncRunner) is the right default), but it exists now to
//! *prove the [`DecisionRunner`] seam is real*: a future Tier-2 beam (or a heavy
//! neural forward) can switch the controller to an off-thread runner with no caller
//! change.
//!
//! # Why `std::thread` + `mpsc` and not `AsyncComputeTaskPool`
//!
//! The AI core is deliberately **Bevy-free** (so it stays deterministic and
//! unit-testable like [`crate::engine`]), so this reference runner uses only `std`:
//! a worker thread and a one-shot [`mpsc`] channel — no Bevy, no extra crate. The
//! Bevy-task variant is a drop-in alternative for the integration layer when one is
//! wanted:
//!
//! ```ignore
//! // Sketch of the AsyncComputeTaskPool variant (Appendix A), for the Bevy layer:
//! let task = AsyncComputeTaskPool::get().spawn(async move { policy.decide(&obs) });
//! // each frame: block_on(future::poll_once(&mut task)) -> Some(decision) when done.
//! ```
//!
//! `cfg(not(target_arch = "wasm32"))` only: `wasm32-unknown-unknown` has no threads,
//! so on web the controller uses a cooperative time-sliced runner instead (see
//! `runner/mod.rs`).
//!
//! # Determinism
//!
//! The worker is fed an **owned** [`Observation`] moved across the channel — never
//! live engine state — and the policy carries its own seeded RNG, so the off-thread
//! decision is bit-identical to the synchronous one. Thread *timing* is
//! nondeterministic, but it only affects *which frame* the decision arrives on, not
//! *what* it is; the controller tolerates a late decision by emitting neutral frames
//! until it lands.

use std::sync::mpsc::{self, Receiver, TryRecvError};
use std::thread::JoinHandle;

use crate::ai::policy::{Decision, Observation, Policy};
use crate::ai::runner::DecisionRunner;

/// A [`DecisionRunner`] that decides on a worker thread.
///
/// Holds the policy between decisions (the worker borrows it by moving it into the
/// thread and moving it back out on completion). One decision is in flight at a
/// time; [`submit`](DecisionRunner::submit) supersedes a previous one.
pub struct ThreadedRunner {
    /// The policy, parked here between decisions. `None` while the worker owns it.
    idle: Option<Box<dyn Policy>>,
    /// The in-flight worker, if any: its join handle and the result channel.
    inflight: Option<Worker>,
}

/// One in-flight decision: the worker thread and the channel it reports on. The
/// worker sends back `(decision, policy)` so the runner reclaims its owned policy.
struct Worker {
    handle: JoinHandle<()>,
    rx: Receiver<DecisionResult>,
}

/// What the worker sends back: the decision plus the moved-back policy.
type DecisionResult = (Decision, Box<dyn Policy>);

impl ThreadedRunner {
    /// Build a runner around a policy.
    pub fn new(policy: Box<dyn Policy>) -> Self {
        Self {
            idle: Some(policy),
            inflight: None,
        }
    }

    /// Reclaim the policy from a finished worker so the next `submit` can reuse it.
    /// Joins the worker thread (already finished — it just sent its result).
    fn reclaim(&mut self, worker: Worker, result: DecisionResult) -> Decision {
        let (decision, policy) = result;
        // The thread has sent its result, so this join returns promptly.
        let _ = worker.handle.join();
        self.idle = Some(policy);
        decision
    }
}

impl DecisionRunner for ThreadedRunner {
    fn submit(&mut self, obs: Observation) {
        // Supersede any in-flight decision: drop its receiver (the worker's send
        // will fail harmlessly). We cannot reclaim the policy from a still-running
        // worker without blocking, so a superseded decision leaves us with no idle
        // policy — the common path (controller cancels on piece change before
        // re-submitting) avoids this.
        let _ = self.inflight.take();

        let Some(mut policy) = self.idle.take() else {
            // No policy available (a prior decision was superseded mid-flight and
            // never reclaimed). Skip this submit; the controller retries next frame.
            return;
        };

        let (tx, rx) = mpsc::channel::<DecisionResult>();
        let handle = std::thread::spawn(move || {
            let decision = policy.decide(&obs);
            // Move the policy back so the runner can reuse it. If the receiver was
            // dropped (superseded), the send just errors and the values drop here.
            let _ = tx.send((decision, policy));
        });

        self.inflight = Some(Worker { handle, rx });
    }

    fn poll(&mut self) -> Option<Decision> {
        let worker = self.inflight.take()?;
        match worker.rx.try_recv() {
            Ok(result) => Some(self.reclaim(worker, result)),
            Err(TryRecvError::Empty) => {
                // Still working — put the worker back and report "not ready".
                self.inflight = Some(worker);
                None
            }
            Err(TryRecvError::Disconnected) => {
                // The worker panicked without sending. Surface as a completed
                // decision with no move (the bot idles) rather than hanging.
                let _ = worker.handle.join();
                Some(Decision::None)
            }
        }
    }

    fn cancel(&mut self) {
        // Block briefly to reclaim the policy so the next decision has one to use —
        // the decision is bounded and short, so this join is cheap; a Tier-2 beam
        // would instead be made cooperatively cancellable.
        if let Some(worker) = self.inflight.take() {
            if let Ok((_, policy)) = worker.rx.recv() {
                let _ = worker.handle.join();
                self.idle = Some(policy);
            } else {
                let _ = worker.handle.join();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::movegen;
    use crate::ai::policy::SearchPolicy;
    use crate::ai::state::SearchState;
    use crate::engine::{Board, CellKind, PieceType};
    use std::collections::VecDeque;

    fn runner() -> ThreadedRunner {
        ThreadedRunner::new(Box::new(SearchPolicy::greedy(0.0, 1)))
    }

    fn tetris_state() -> Observation {
        let mut board = Board::new(4, 10);
        for y in 0..4 {
            for x in 0..3 {
                board.set(x, y, CellKind::Some(PieceType::O));
            }
        }
        let active = movegen::spawn_piece(PieceType::I, 4, 10);
        SearchState::for_test(board, active, None, VecDeque::new())
    }

    /// Poll until the worker reports a decision (bounded so the test can't hang).
    fn poll_until_ready(runner: &mut ThreadedRunner) -> Decision {
        for _ in 0..100_000 {
            if let Some(decision) = runner.poll() {
                return decision;
            }
            std::thread::yield_now();
        }
        panic!("worker never produced a result");
    }

    #[test]
    fn off_thread_decision_returns_a_placement() {
        let mut runner = runner();
        runner.submit(tetris_state());
        assert!(matches!(
            poll_until_ready(&mut runner),
            Decision::Place(_)
        ));
    }

    #[test]
    fn off_thread_decision_matches_synchronous() {
        // The threaded runner must produce the *same* decision the synchronous
        // runner does — proving timing is the only difference, not the result.
        use crate::ai::runner::SyncRunner;
        let mut threaded = runner();
        let mut sync = SyncRunner::new(Box::new(SearchPolicy::greedy(0.0, 1)));

        threaded.submit(tetris_state());
        sync.submit(tetris_state());

        match (poll_until_ready(&mut threaded), sync.poll().unwrap()) {
            (Decision::Place(off), Decision::Place(on)) => {
                assert_eq!(off.path, on.path);
                assert_eq!(off.origin(), on.origin());
            }
            _ => panic!("expected placements"),
        }
    }

    #[test]
    fn cancel_then_resubmit_reuses_the_policy() {
        let mut runner = runner();
        runner.submit(tetris_state());
        runner.cancel(); // reclaims the policy
                         // A fresh submit must still work (the policy was returned, not lost).
        runner.submit(tetris_state());
        assert!(matches!(
            poll_until_ready(&mut runner),
            Decision::Place(_)
        ));
    }
}
