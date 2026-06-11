//! The synchronous decision runner (the blocking direct-drive venue).
//!
//! [`SyncRunner`] runs the policy inline: [`submit`](super::DecisionRunner::submit)
//! decides right then and buffers the [`Decision`], and the next
//! [`poll`](super::DecisionRunner::poll) returns it. This is the venue for
//! **headless** drivers — benchmarks, tests, research bots — where there is no
//! frame to hitch and a blocked "frame" is free throughput: exact budgets, zero
//! pacing, trivially deterministic. Interactive surfaces use the cooperative
//! [`SlicedRunner`](super::SlicedRunner) instead, which spreads the same work
//! (and the same decision) across polls.
//!
//! The runner **owns** the policy: [`Policy::decide`] takes `&mut self` (the policy
//! carries its seeded RNG, and an incremental search carries state between
//! calls), so it cannot be shared by reference across a poll boundary — the runner
//! is its home.

use crate::ai::policy::{Decision, Observation, Policy};
use crate::ai::runner::DecisionRunner;

/// A [`DecisionRunner`] that decides inline and buffers the result for the next
/// poll. Owns the [`Policy`] it drives.
pub struct SyncRunner {
    policy: Box<dyn Policy>,
    /// The buffered decision from the last [`submit`](DecisionRunner::submit), taken
    /// by the next [`poll`](DecisionRunner::poll).
    pending: Option<Decision>,
}

impl SyncRunner {
    /// Build a runner around a policy.
    pub fn new(policy: Box<dyn Policy>) -> Self {
        Self {
            policy,
            pending: None,
        }
    }
}

impl DecisionRunner for SyncRunner {
    fn submit(&mut self, obs: Observation) {
        // Synchronous: decide now, hand back on the next poll. This models the async
        // contract (submit then poll) without the async machinery.
        self.pending = Some(self.policy.decide(&obs));
    }

    fn poll(&mut self) -> Option<Decision> {
        self.pending.take()
    }

    fn cancel(&mut self) {
        self.pending = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::movegen;
    use crate::ai::policy::SearchPolicy;
    use crate::ai::state::SearchState;
    use crate::engine::{Board, CellKind, PieceType};

    fn runner() -> SyncRunner {
        SyncRunner::new(Box::new(SearchPolicy::greedy(0.0, 1)))
    }

    /// A search state with a vertical-I-clears-Tetris board (cols 0-2 filled, well
    /// at col 3) so there is an unambiguous best placement to decide.
    fn tetris_state() -> Observation {
        let mut board = Board::new(4, 10);
        for y in 0..4 {
            for x in 0..3 {
                board.set(x, y, CellKind::Some(PieceType::O));
            }
        }
        let active = movegen::spawn_piece(PieceType::I, 4, 10);
        SearchState::for_test(board, active, None, std::iter::empty())
    }

    #[test]
    fn submit_then_poll_returns_a_decision() {
        let mut runner = runner();
        runner.submit(tetris_state());
        let decision = runner.poll().expect("a decision is pending");
        assert!(
            matches!(decision, Decision::Place(_)),
            "the tetris state has a legal placement"
        );
    }

    #[test]
    fn poll_without_submit_is_none() {
        let mut runner = runner();
        assert!(runner.poll().is_none(), "nothing submitted yet");
    }

    #[test]
    fn decision_is_consumed_by_the_first_poll() {
        let mut runner = runner();
        runner.submit(tetris_state());
        assert!(runner.poll().is_some(), "first poll takes the decision");
        assert!(
            runner.poll().is_none(),
            "second poll without a re-submit is empty"
        );
    }

    #[test]
    fn cancel_drops_a_buffered_decision() {
        let mut runner = runner();
        runner.submit(tetris_state());
        runner.cancel();
        assert!(
            runner.poll().is_none(),
            "cancel discarded the pending decision"
        );
    }

    #[test]
    fn synchronous_runner_is_deterministic() {
        // The same state submitted to two runners yields identical decisions,
        // proving the sync path carries no hidden RNG/clock.
        let mut a = runner();
        let mut b = runner();
        a.submit(tetris_state());
        b.submit(tetris_state());
        match (a.poll().unwrap(), b.poll().unwrap()) {
            (Decision::Place(pa), Decision::Place(pb)) => {
                assert_eq!(pa.path, pb.path);
                assert_eq!(pa.origin(), pb.origin());
            }
            _ => panic!("expected placements"),
        }
    }
}
