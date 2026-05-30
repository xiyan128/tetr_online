//! The synchronous compute runner (shipped Tier-1 back-end).
//!
//! [`SyncRunner`] runs the search inline: [`submit`](super::ComputeRunner::submit)
//! plans to completion right then and buffers the result, and the next
//! [`poll`](super::ComputeRunner::poll) returns it. The Tier-1 greedy planner is
//! microseconds per call, so there is nothing to gain from threads or time-slicing
//! yet — and a synchronous runner keeps the whole AI path trivially deterministic
//! and unit-testable. The off-thread (`native.rs`) and cooperative-WASM (`web.rs`)
//! runners drop in behind the same [`ComputeRunner`](super::ComputeRunner) trait
//! when a Tier-2 beam makes the search frame-expensive.
//!
//! The runner **owns** the planner and evaluator: the [`Planner`] trait takes
//! `&mut self` (an incremental search carries state between calls), so it cannot
//! be shared by reference across a poll boundary; the runner is its home.

use crate::ai::eval::Evaluator;
use crate::ai::runner::ComputeRunner;
use crate::ai::search::{PlacementPlan, Planner, PlannerStep, SearchBudget};
use crate::ai::state::SearchState;

/// A [`ComputeRunner`] that searches inline and buffers the result for the next
/// poll. Owns the planner + evaluator it drives.
pub struct SyncRunner {
    planner: Box<dyn Planner>,
    evaluator: Box<dyn Evaluator>,
    /// The buffered result of the last [`submit`](ComputeRunner::submit), taken by
    /// the next [`poll`](ComputeRunner::poll). `Some(opt)` is a finished search
    /// (`opt == None` ⇒ no legal placement); the outer `None` is "nothing pending".
    pending: Option<Option<PlacementPlan>>,
}

impl SyncRunner {
    /// Build a runner around a planner and evaluator.
    pub fn new(planner: Box<dyn Planner>, evaluator: Box<dyn Evaluator>) -> Self {
        Self {
            planner,
            evaluator,
            pending: None,
        }
    }

    /// Drive the planner to a final plan under `budget`.
    ///
    /// Loops while the planner asks for more budget so this works unchanged for a
    /// future incremental planner; the greedy Tier-1 planner returns
    /// [`PlannerStep::Done`] on the first call. The `NeedMoreBudget` loop is
    /// bounded by the node budget plus a hard cap, so a misbehaving incremental
    /// planner can never spin forever here.
    fn run_to_completion(
        &mut self,
        state: &SearchState,
        budget: SearchBudget,
    ) -> Option<PlacementPlan> {
        // A generous safety cap: even an incremental planner that yields one node
        // at a time terminates. Greedy never iterates more than once.
        const MAX_STEPS: u32 = 100_000;
        for _ in 0..MAX_STEPS {
            match self.planner.plan(state, self.evaluator.as_ref(), budget) {
                PlannerStep::Done(plan) => return plan,
                PlannerStep::NeedMoreBudget => continue,
            }
        }
        // Defensive: an incremental planner that never converged. Treat as "no
        // plan" rather than looping forever (it would surface as the bot idling).
        None
    }
}

impl ComputeRunner for SyncRunner {
    fn submit(&mut self, state: SearchState, budget: SearchBudget) {
        // Synchronous: compute now, hand back on the next poll. This models the
        // async contract (submit then poll) without the async machinery.
        self.pending = Some(self.run_to_completion(&state, budget));
    }

    fn poll(&mut self) -> Option<Option<PlacementPlan>> {
        self.pending.take()
    }

    fn cancel(&mut self) {
        self.pending = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::eval::LinearEvaluator;
    use crate::ai::movegen;
    use crate::ai::search::GreedyPlanner;
    use crate::engine::{Board, CellKind, PieceType};
    use std::collections::VecDeque;

    fn runner() -> SyncRunner {
        SyncRunner::new(
            Box::new(GreedyPlanner::new()),
            Box::new(LinearEvaluator::default()),
        )
    }

    /// A search state with a vertical-I-clears-Tetris board (cols 0-2 filled, well
    /// at col 3) so there is an unambiguous best placement to plan.
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

    #[test]
    fn submit_then_poll_returns_a_plan() {
        let mut runner = runner();
        runner.submit(tetris_state(), SearchBudget::greedy());
        let result = runner.poll().expect("a completed search is pending");
        assert!(result.is_some(), "the tetris state has a legal placement");
    }

    #[test]
    fn poll_without_submit_is_none() {
        let mut runner = runner();
        assert!(runner.poll().is_none(), "nothing submitted yet");
    }

    #[test]
    fn result_is_consumed_by_the_first_poll() {
        let mut runner = runner();
        runner.submit(tetris_state(), SearchBudget::greedy());
        assert!(runner.poll().is_some(), "first poll takes the result");
        assert!(
            runner.poll().is_none(),
            "second poll without a re-submit is empty"
        );
    }

    #[test]
    fn cancel_drops_a_buffered_result() {
        let mut runner = runner();
        runner.submit(tetris_state(), SearchBudget::greedy());
        runner.cancel();
        assert!(runner.poll().is_none(), "cancel discarded the pending plan");
    }

    #[test]
    fn synchronous_runner_is_deterministic() {
        // The same state submitted to two runners yields identical plans (score +
        // path), proving the sync path carries no hidden RNG/clock.
        let mut a = runner();
        let mut b = runner();
        a.submit(tetris_state(), SearchBudget::greedy());
        b.submit(tetris_state(), SearchBudget::greedy());
        let pa = a.poll().unwrap().unwrap();
        let pb = b.poll().unwrap().unwrap();
        assert_eq!(pa.score, pb.score);
        assert_eq!(pa.placement.path, pb.placement.path);
        assert_eq!(pa.placement.origin(), pb.placement.origin());
    }
}
