//! The cooperative time-sliced decision runner (the interactive venue).
//!
//! [`SlicedRunner`] spends one bounded node quantum on the policy's in-flight
//! thinking per [`poll`](super::DecisionRunner::poll) and returns the decision on
//! the poll whose quantum meets the policy's budget contract. A heavy search thus
//! spreads across frames — ~a few milliseconds each — instead of stalling the
//! frame that submitted it, which is what makes a strong bot watchable on the
//! main thread (native *and* wasm; the browser has no other thread to give).
//!
//! # The quantum
//!
//! The per-poll quantum is a **configured node count, never a measured time**
//! (the runner determinism rule: no clocks). The platform defaults convert one
//! "comfortable slice of a 60 Hz frame" into nodes at the measured release-build
//! search rate; an explicit [`with_quantum`](SlicedRunner::with_quantum) overrides
//! them for tests and tuning. Total work per decision is the policy's budget
//! either way — the quantum only chooses *suspension points*, so slicing never
//! changes a decision (pinned by the policy/mind layers), only the poll on which
//! it lands.
//!
//! # Work placement
//!
//! [`submit`](super::DecisionRunner::submit) is free: it only stores the
//! observation. All work — including the policy's seeding reroot — happens inside
//! `poll`, so the per-frame cost is bounded by `seed + quantum` on the first poll
//! and `quantum` after, and a venue swap never changes *when* work runs relative
//! to the controller's frame.

use crate::ai::policy::{Decision, Observation, Policy, PolicyProgress};
use crate::ai::runner::DecisionRunner;

/// Nodes per poll on native: ≈3–5 ms of best-first search at the measured
/// release-build rate (~6 node expansions/ms with the CC2 evaluator) —
/// comfortably inside a 16.6 ms frame alongside rendering.
pub const NATIVE_QUANTUM: u32 = 32;

/// Nodes per poll on wasm: the browser main thread runs the same search ~2–4×
/// slower and shares the frame with the renderer, so half the native quantum
/// keeps the worst poll near ~5–10 ms. As the smaller of the two defaults this
/// is also the **cross-platform worst case** that one-budget operating points
/// size against (see `controller::ATTACK_NODE_BUDGET`).
pub const WASM_QUANTUM: u32 = 16;

/// The platform default ([`NATIVE_QUANTUM`] / [`WASM_QUANTUM`]) — exposed so a
/// venue-tuned construction seam (`AiController::interactive_with`) can name
/// "the default" without duplicating the platform split.
#[cfg(not(target_arch = "wasm32"))]
pub(crate) const DEFAULT_QUANTUM: u32 = NATIVE_QUANTUM;
#[cfg(target_arch = "wasm32")]
pub(crate) const DEFAULT_QUANTUM: u32 = WASM_QUANTUM;

/// A [`DecisionRunner`] that thinks one bounded quantum per poll. Owns the
/// [`Policy`] it drives.
pub struct SlicedRunner {
    policy: Box<dyn Policy>,
    /// The observation being decided; `None` when idle (nothing submitted, the
    /// decision was delivered, or the think was cancelled).
    obs: Option<Observation>,
    /// Node quantum per poll (≥ 1).
    quantum: u32,
}

impl SlicedRunner {
    /// Build a runner around a policy with the platform-default quantum.
    pub fn new(policy: Box<dyn Policy>) -> Self {
        Self::with_quantum(policy, DEFAULT_QUANTUM)
    }

    /// Build a runner with an explicit per-poll node `quantum` (clamped to ≥ 1).
    pub fn with_quantum(policy: Box<dyn Policy>, quantum: u32) -> Self {
        Self {
            policy,
            obs: None,
            quantum: quantum.max(1),
        }
    }
}

impl DecisionRunner for SlicedRunner {
    fn submit(&mut self, obs: Observation) {
        // Free by design: the work (seeding included) happens in the polls. The
        // policy's own root fingerprint discards a stale in-flight search when
        // this observation differs from the one it was thinking about.
        self.obs = Some(obs);
    }

    fn poll(&mut self) -> Option<Decision> {
        let obs = self.obs.as_ref()?;
        // Re-assert the root every poll (a no-op fingerprint compare after the
        // first), then spend this frame's quantum.
        self.policy.reroot(obs);
        if self.policy.think(self.quantum) == PolicyProgress::Ready {
            let obs = self.obs.take().expect("checked above");
            return Some(self.policy.take(&obs));
        }
        None
    }

    fn cancel(&mut self) {
        self.obs = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::eval::LinearEvaluator;
    use crate::ai::policy::SearchPolicy;
    use crate::ai::runner::SyncRunner;
    use crate::ai::search::{BestFirstPlanner, SearchBudget};
    use crate::ai::state::SearchState;
    use crate::engine::{Engine, EngineConfig, InputFrame};

    /// The interactive operating point's shape: best-first, 150 nodes, depth 6.
    fn attack_shaped_policy(seed: u64) -> SearchPolicy {
        SearchPolicy::new(
            Box::new(BestFirstPlanner::new()),
            Box::new(LinearEvaluator::default()),
            SearchBudget::best_first(150, 6),
            0.0,
            seed,
        )
    }

    /// A real engine observation (hold + full queue present).
    fn engine_obs(seed: u64) -> Observation {
        let mut engine = Engine::new(EngineConfig::default(), seed);
        engine.step(InputFrame::default());
        SearchState::from_snapshot(&engine.snapshot()).expect("active piece present")
    }

    fn placement(decision: Decision) -> crate::ai::movegen::Placement {
        match decision {
            Decision::Place(p) => p,
            Decision::None => panic!("expected a placement"),
        }
    }

    #[test]
    fn spreads_a_heavy_decision_across_polls_and_matches_the_blocking_venue() {
        // The venue contract end to end: a 150-node decision at quantum 16 spans
        // exactly ceil(150/16) = 10 polls — and the delivered decision is
        // identical to the same policy run in the blocking SyncRunner.
        let obs = engine_obs(7);

        let mut sliced = SlicedRunner::with_quantum(Box::new(attack_shaped_policy(1)), 16);
        sliced.submit(obs.clone());
        let mut polls = 1;
        let sliced_decision = loop {
            match sliced.poll() {
                Some(d) => break d,
                None => polls += 1,
            }
            assert!(polls <= 100, "the decision must land");
        };
        assert_eq!(polls, 10, "150 nodes at quantum 16 is 10 polls");

        let mut blocking = SyncRunner::new(Box::new(attack_shaped_policy(1)));
        blocking.submit(obs);
        let blocking_decision = blocking.poll().expect("sync decides at submit");

        let (s, b) = (placement(sliced_decision), placement(blocking_decision));
        assert_eq!(s.origin(), b.origin());
        assert_eq!(s.path, b.path);
    }

    #[test]
    fn one_shot_policies_deliver_on_the_first_poll() {
        // A policy with no incremental thinking (greedy: default Ready) behaves
        // exactly like the sync venue — submit, then the first poll delivers.
        let mut runner = SlicedRunner::new(Box::new(SearchPolicy::greedy(0.0, 1)));
        runner.submit(engine_obs(7));
        assert!(runner.poll().is_some(), "greedy completes in one quantum");
        assert!(runner.poll().is_none(), "the decision was taken");
    }

    #[test]
    fn cancel_discards_the_in_flight_think() {
        let mut runner = SlicedRunner::with_quantum(Box::new(attack_shaped_policy(1)), 16);
        runner.submit(engine_obs(7));
        assert!(runner.poll().is_none(), "first quantum: still working");
        runner.cancel();
        assert!(runner.poll().is_none(), "cancelled: nothing to deliver");

        // A fresh submit after cancel works (the policy re-roots away the stale run).
        runner.submit(engine_obs(42));
        let mut delivered = false;
        for _ in 0..100 {
            if runner.poll().is_some() {
                delivered = true;
                break;
            }
        }
        assert!(delivered, "a resubmitted decision lands");
    }

    #[test]
    fn sliced_venue_is_deterministic() {
        // Same policy seed, same observation, same quantum ⇒ the same decision on
        // the same poll — the venue adds no RNG and no clock.
        let run = |seed: u64| {
            let mut runner = SlicedRunner::with_quantum(Box::new(attack_shaped_policy(seed)), 16);
            runner.submit(engine_obs(7));
            let mut polls = 1u32;
            loop {
                if let Some(d) = runner.poll() {
                    return (polls, placement(d).origin());
                }
                polls += 1;
                assert!(polls <= 100);
            }
        };
        assert_eq!(run(99), run(99));
    }
}
