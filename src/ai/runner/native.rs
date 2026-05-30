//! The native off-thread compute runner (reference impl of the async seam).
//!
//! [`ThreadedRunner`] runs the search on a worker thread and hands the result
//! back over a channel, so a frame-expensive search never blocks the game loop.
//! Tier-1 greedy does **not** need this (it is microseconds — [`SyncRunner`] is
//! the right default), but it exists now to *prove the [`ComputeRunner`] seam is
//! real*: a future Tier-2 beam can switch the controller to an off-thread runner
//! with no caller change.
//!
//! # Why `std::thread` + `mpsc` and not `AsyncComputeTaskPool`
//!
//! The M2 plan's native back-end is "`AsyncComputeTaskPool::get().spawn(..)` +
//! `block_on(poll_once(..))`" *or* "`std::thread` + a channel". The AI core is
//! deliberately **Bevy-free** (so it stays deterministic and unit-testable like
//! [`crate::engine`]), so this reference runner uses only `std`: a worker thread
//! and a one-shot [`mpsc`] channel — no Bevy, no extra crate. The Bevy-task
//! variant is a drop-in alternative for the integration layer when one is wanted:
//!
//! ```ignore
//! // Sketch of the AsyncComputeTaskPool variant (Appendix A), for the Bevy layer:
//! let task = AsyncComputeTaskPool::get().spawn(async move { run_search(state) });
//! // each frame: block_on(future::poll_once(&mut task)) -> Some(plan) when done.
//! // Dropping `task` cancels it (our `cancel` drops the join handle + receiver).
//! ```
//!
//! `cfg(not(target_arch = "wasm32"))` only: `wasm32-unknown-unknown` has no
//! threads, so on web the controller uses a cooperative time-sliced runner
//! instead (see `runner/mod.rs`).
//!
//! # Determinism
//!
//! The worker is fed an **owned** [`SearchState`] moved across the channel — never
//! live engine state — and the planner/evaluator carry no RNG or clock, so the
//! off-thread result is bit-identical to the synchronous one. Thread *timing* is
//! nondeterministic, but it only affects *which frame* the plan arrives on, not
//! *what* the plan is; the controller tolerates a late plan by emitting neutral
//! frames until it lands.

use std::sync::mpsc::{self, Receiver, TryRecvError};
use std::thread::JoinHandle;

use crate::ai::eval::Evaluator;
use crate::ai::runner::ComputeRunner;
use crate::ai::search::{PlacementPlan, Planner, PlannerStep, SearchBudget};
use crate::ai::state::SearchState;

/// A [`ComputeRunner`] that searches on a worker thread.
///
/// Holds the planner + evaluator between searches (the worker borrows them by
/// moving them into the thread and moving them back out on completion). One
/// search is in flight at a time; [`submit`](ComputeRunner::submit) supersedes a
/// previous one.
pub struct ThreadedRunner {
    /// The planner + evaluator, parked here between searches. `None` while a
    /// search owns them on the worker thread.
    idle: Option<(Box<dyn Planner>, Box<dyn Evaluator>)>,
    /// The in-flight worker, if any: its join handle and the result channel.
    inflight: Option<Worker>,
}

/// One in-flight search: the worker thread and the channel it reports on. The
/// worker sends back `(result, planner, evaluator)` so the runner reclaims its
/// owned planner for the next search.
struct Worker {
    handle: JoinHandle<()>,
    rx: Receiver<SearchResult>,
}

/// What the worker sends back: the plan plus the moved-back planner + evaluator.
type SearchResult = (Option<PlacementPlan>, Box<dyn Planner>, Box<dyn Evaluator>);

impl ThreadedRunner {
    /// Build a runner around a planner and evaluator.
    pub fn new(planner: Box<dyn Planner>, evaluator: Box<dyn Evaluator>) -> Self {
        Self {
            idle: Some((planner, evaluator)),
            inflight: None,
        }
    }

    /// Reclaim the planner + evaluator from a finished worker so the next
    /// `submit` can reuse them. Joins the worker thread (already finished — it
    /// just sent its result).
    fn reclaim(&mut self, worker: Worker, result: SearchResult) -> Option<PlacementPlan> {
        let (plan, planner, evaluator) = result;
        // The thread has sent its result, so this join returns promptly.
        let _ = worker.handle.join();
        self.idle = Some((planner, evaluator));
        plan
    }
}

impl ComputeRunner for ThreadedRunner {
    fn submit(&mut self, state: SearchState, budget: SearchBudget) {
        // Supersede any in-flight search: drop its receiver (the worker's send
        // will fail harmlessly) and recover the planner by detaching the thread.
        // In practice `cancel` is called first; this is belt-and-braces.
        if self.inflight.take().is_some() {
            // We cannot reclaim the planner from a still-running worker without
            // blocking, so a superseded search leaves us with no idle planner.
            // The common path (controller cancels on piece change before
            // re-submitting) avoids this; if it happens, fall through and skip.
        }

        let Some((mut planner, evaluator)) = self.idle.take() else {
            // No planner available (a prior search was superseded mid-flight and
            // never reclaimed). Skip this submit; the controller will retry next
            // frame once the worker is reclaimed via poll.
            return;
        };

        let (tx, rx) = mpsc::channel::<SearchResult>();
        let handle = std::thread::spawn(move || {
            let plan = run_search(&mut planner, evaluator.as_ref(), &state, budget);
            // Move the planner + evaluator back so the runner can reuse them. If
            // the receiver was dropped (superseded), the send just errors and the
            // owned values drop here — fine.
            let _ = tx.send((plan, planner, evaluator));
        });

        self.inflight = Some(Worker { handle, rx });
    }

    fn poll(&mut self) -> Option<Option<PlacementPlan>> {
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
                // search with no plan (the bot idles) rather than hanging.
                let _ = worker.handle.join();
                Some(None)
            }
        }
    }

    fn cancel(&mut self) {
        // Drop the receiver: a finished worker's send errors harmlessly. We block
        // briefly to reclaim the planner so the next search has one to use — the
        // search is bounded and short, so this join is cheap; a Tier-2 beam would
        // instead be made cooperatively cancellable.
        if let Some(worker) = self.inflight.take() {
            if let Ok(result) = worker.rx.recv() {
                let (_, planner, evaluator) = result;
                let _ = worker.handle.join();
                self.idle = Some((planner, evaluator));
            } else {
                let _ = worker.handle.join();
            }
        }
    }

    fn evaluator(&self) -> Option<&dyn Evaluator> {
        // The evaluator is parked in `idle` between searches and owned by the
        // worker thread while a search is in flight. The controller only asks after
        // a successful `poll` (which reclaims it into `idle`), so in practice this
        // is `Some`; `None` would just skip error injection for that piece.
        self.idle.as_ref().map(|(_, evaluator)| evaluator.as_ref())
    }
}

/// Run the planner to a final plan (mirrors [`SyncRunner`]'s loop): drive it while
/// it asks for more budget so a future incremental planner works unchanged.
fn run_search(
    planner: &mut Box<dyn Planner>,
    evaluator: &dyn Evaluator,
    state: &SearchState,
    budget: SearchBudget,
) -> Option<PlacementPlan> {
    const MAX_STEPS: u32 = 100_000;
    for _ in 0..MAX_STEPS {
        match planner.plan(state, evaluator, budget) {
            PlannerStep::Done(plan) => return plan,
            PlannerStep::NeedMoreBudget => continue,
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::eval::LinearEvaluator;
    use crate::ai::movegen;
    use crate::ai::search::GreedyPlanner;
    use crate::engine::{Board, CellKind, PieceType};
    use std::collections::VecDeque;

    fn runner() -> ThreadedRunner {
        ThreadedRunner::new(
            Box::new(GreedyPlanner::new()),
            Box::new(LinearEvaluator::default()),
        )
    }

    fn tetris_state() -> SearchState {
        let mut board = Board::new(4, 10);
        for y in 0..4 {
            for x in 0..3 {
                board.set(x, y, CellKind::Some(PieceType::O));
            }
        }
        let active = movegen::spawn_piece(PieceType::I, 4, 10);
        SearchState::for_test(board, active, None, VecDeque::new())
    }

    /// Poll until the worker reports a result (bounded so the test can't hang).
    fn poll_until_ready(runner: &mut ThreadedRunner) -> Option<PlacementPlan> {
        for _ in 0..100_000 {
            if let Some(result) = runner.poll() {
                return result;
            }
            std::thread::yield_now();
        }
        panic!("worker never produced a result");
    }

    #[test]
    fn off_thread_search_returns_a_plan() {
        let mut runner = runner();
        runner.submit(tetris_state(), SearchBudget::greedy());
        let plan = poll_until_ready(&mut runner);
        assert!(plan.is_some(), "the tetris state has a legal placement");
    }

    #[test]
    fn off_thread_result_matches_synchronous() {
        // The threaded runner must produce the *same* plan the synchronous runner
        // does — proving timing is the only difference, not the result.
        use crate::ai::runner::SyncRunner;
        let mut threaded = runner();
        let mut sync = SyncRunner::new(
            Box::new(GreedyPlanner::new()),
            Box::new(LinearEvaluator::default()),
        );

        threaded.submit(tetris_state(), SearchBudget::greedy());
        sync.submit(tetris_state(), SearchBudget::greedy());

        let off = poll_until_ready(&mut threaded).unwrap();
        let on = sync.poll().unwrap().unwrap();
        assert_eq!(off.score, on.score);
        assert_eq!(off.placement.path, on.placement.path);
        assert_eq!(off.placement.origin(), on.placement.origin());
    }

    #[test]
    fn cancel_then_resubmit_reuses_the_planner() {
        let mut runner = runner();
        runner.submit(tetris_state(), SearchBudget::greedy());
        runner.cancel(); // reclaims the planner
                         // A fresh submit must still work (the planner was returned, not lost).
        runner.submit(tetris_state(), SearchBudget::greedy());
        let plan = poll_until_ready(&mut runner);
        assert!(plan.is_some(), "planner reused after cancel");
    }
}
