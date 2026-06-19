//! The time-budgeted cooperative decision runner (the *responsive* interactive venue).
//!
//! [`BudgetedRunner`] is [`SlicedRunner`](super::SlicedRunner) with one change: each
//! [`poll`](super::DecisionRunner::poll) keeps spending node quanta until either the
//! decision is ready **or** a per-frame wall-clock budget is exhausted — instead of
//! exactly one quantum. The motivation: a one-quantum-per-frame venue throttles a
//! heavy bot to the frame loop, not the CPU (the APP champion needs ~30 quanta, so at
//! one per 60 Hz frame it deliberates ~0.5 s per piece while each poll leaves most of
//! the frame idle). Spending the idle frame budget instead collapses that to a handful
//! of frames, at full strength, without stalling the renderer.
//!
//! # The determinism rule, and why this venue is the deliberate exception
//!
//! [`SlicedRunner`](super::SlicedRunner)'s quantum is a *configured node count, never a
//! measured time*, so a sliced game is reproducible from `(seed, quantum, cadence)`.
//! A budgeted poll instead spends *however many* quanta fit a wall-clock window, so the
//! poll on which a decision lands depends on machine speed — it is **timing
//! nondeterministic by design**, the same trade the future thread/worker venue makes
//! (`docs/adr-ai-compute-architecture.md`). It is therefore the *game's* venue only:
//! benchmarks, research, and the venue-equivalence gate stay on the blocking
//! [`SyncRunner`](super::SyncRunner) / [`SlicedRunner`](super::SlicedRunner). What does
//! **not** change is the
//! *decision*: the budget only chooses suspension points, so the delivered decision is
//! identical to the blocking one for any budget (pinned by the tests below).
//!
//! # The injected clock
//!
//! The engine-agnostic core never reads a platform clock itself — `std::time::Instant`
//! panics on wasm, and keeping timing out of the core is what lets every other venue
//! stay reproducible. The host injects a platform-correct [`MonotonicClock`] (Bevy's
//! web-time-backed `Instant` on the game side); the core only ever takes *differences*
//! between [`elapsed`](MonotonicClock::elapsed) readings. A fake clock makes the
//! budgeting logic deterministically testable.

use std::time::Duration;

use crate::ai::policy::{Decision, Observation, Policy, PolicyProgress};
use crate::ai::runner::DecisionRunner;

/// Native per-frame compute budget: ~half a 16.6 ms frame, leaving the rest for the
/// renderer and the catch-up margin. The champion's ~90 ms decision lands in ~12
/// frames at this budget — inside its 200 ms reaction window, so it acts on time.
pub const NATIVE_BUDGET: Duration = Duration::from_micros(8_000);

/// Wasm per-frame compute budget: smaller, because the browser main thread runs the
/// search slower *and* shares the frame with the renderer. Tune against
/// `scripts/fps-probe.js` (and note the `opt-level="z"` web profile inflates the
/// per-node cost — a separate lever).
pub const WASM_BUDGET: Duration = Duration::from_micros(4_000);

/// The platform default ([`NATIVE_BUDGET`] / [`WASM_BUDGET`]).
#[cfg(not(target_arch = "wasm32"))]
pub(crate) const DEFAULT_BUDGET: Duration = NATIVE_BUDGET;
#[cfg(target_arch = "wasm32")]
pub(crate) const DEFAULT_BUDGET: Duration = WASM_BUDGET;

/// The granularity at which a budgeted poll checks the clock: small, so a poll
/// overshoots its [`DEFAULT_BUDGET`] by at most one step's worth of search *or* one
/// clock-resolution tick — whichever is larger (coarse wasm `Performance.now`, when
/// cross-origin isolation is off, can be clamped to ~1 ms) — keeping the worst poll
/// comfortably inside a frame. The *total* per-poll work is governed by the budget, not
/// this step — so unlike the sliced quantum it needs no per-bot tuning, and production
/// never varies it (only the granularity-sweep tests do).
const BUDGET_STEP: u32 = 8;

/// A monotonic wall-clock source for the time-budgeted venue, **injected by the host**.
///
/// Only differences between [`elapsed`](Self::elapsed) readings are used, so any fixed
/// epoch is fine — but it must track **real elapsed time**: a budgeted poll is bounded by
/// the clock reaching the budget, so a clock that never advances would let one poll run
/// until the policy finishes (a full-decision frame), or — for a pathological
/// never-terminating policy — forever. The shipped host clock is `Instant`-backed and
/// always advances. `Send` (not `Sync`) matches [`DecisionRunner`]; the controller seat
/// it lives behind is single-threaded.
pub trait MonotonicClock: Send {
    /// Real elapsed time since an arbitrary fixed epoch — monotonic and advancing.
    fn elapsed(&self) -> Duration;
}

/// A [`DecisionRunner`] that spends node quanta until a per-poll wall-clock `budget`
/// is hit (or the decision lands). Owns the [`Policy`] it drives, exactly like
/// [`SlicedRunner`](super::SlicedRunner); the additions are the `budget` and the
/// injected [`MonotonicClock`].
pub struct BudgetedRunner {
    policy: Box<dyn Policy>,
    /// The observation being decided; `None` when idle (nothing submitted, decision
    /// delivered, or cancelled) — same lifecycle as [`SlicedRunner`](super::SlicedRunner).
    obs: Option<Observation>,
    /// Node quantum per `think` step (≥ 1); the granularity at which the budget is
    /// checked. Keep it small so a poll overshoots the budget by at most one step.
    quantum: u32,
    /// Wall-clock compute allowed per poll before yielding the frame.
    budget: Duration,
    clock: Box<dyn MonotonicClock>,
}

impl BudgetedRunner {
    /// Build a runner with a per-poll wall-clock `budget` and injected `clock`. The
    /// per-step granularity is fixed at `BUDGET_STEP`: the budget governs total per-poll
    /// work, so (unlike the sliced quantum) the step is an implementation detail, not a
    /// caller-tuned dial.
    pub fn new(policy: Box<dyn Policy>, budget: Duration, clock: Box<dyn MonotonicClock>) -> Self {
        Self::with_step(policy, BUDGET_STEP, budget, clock)
    }

    /// Build with an explicit per-step `quantum` — only the granularity at which the
    /// budget is checked, never the total work. Private: production goes through
    /// [`new`](Self::new) at `BUDGET_STEP`; the tests vary the step to exercise the
    /// budget arithmetic across granularities.
    fn with_step(
        policy: Box<dyn Policy>,
        quantum: u32,
        budget: Duration,
        clock: Box<dyn MonotonicClock>,
    ) -> Self {
        Self {
            policy,
            obs: None,
            quantum: quantum.max(1),
            budget,
            clock,
        }
    }
}

impl DecisionRunner for BudgetedRunner {
    fn submit(&mut self, obs: Observation) {
        // Free, like the sliced venue: all work (seeding included) happens in `poll`.
        self.obs = Some(obs);
    }

    fn poll(&mut self) -> Option<Decision> {
        let obs = self.obs.as_ref()?;
        // Re-assert the root (a no-op fingerprint compare after the first poll), then
        // spend quanta until the decision lands or this frame's budget is gone.
        self.policy.reroot(obs);
        let start = self.clock.elapsed();
        loop {
            if self.policy.think(self.quantum) == PolicyProgress::Ready {
                let obs = self.obs.take().expect("checked above");
                return Some(self.policy.take(&obs));
            }
            // Post-`think` check: at least one quantum always runs, so even a zero
            // budget makes progress (one quantum/poll, like the sliced venue) and the
            // loop can never spin without advancing the search.
            if self.clock.elapsed().saturating_sub(start) >= self.budget {
                return None;
            }
        }
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
    use crate::ai::runner::{SlicedRunner, SyncRunner};
    use crate::ai::search::{BestFirstPlanner, SearchBudget};
    use crate::ai::state::SearchState;
    use crate::engine::{Engine, EngineConfig, InputFrame};
    use std::sync::atomic::{AtomicU64, Ordering};

    /// A deterministic fake clock: each `elapsed()` reads the running total and then
    /// advances it by `step`, so N readings span `(N-1) * step`. This lets a test fix
    /// exactly how many quanta a poll's budget admits, with zero wall-clock.
    struct FakeClock {
        nanos: AtomicU64,
        step: u64,
    }
    impl FakeClock {
        fn new(step: Duration) -> Self {
            Self {
                nanos: AtomicU64::new(0),
                step: step.as_nanos() as u64,
            }
        }
    }
    impl MonotonicClock for FakeClock {
        fn elapsed(&self) -> Duration {
            Duration::from_nanos(self.nanos.fetch_add(self.step, Ordering::Relaxed))
        }
    }

    /// A frozen clock: never advances, so the budget is never reached — the poll runs
    /// to completion in one call (the blocking-equivalent extreme).
    struct FrozenClock;
    impl MonotonicClock for FrozenClock {
        fn elapsed(&self) -> Duration {
            Duration::ZERO
        }
    }

    fn attack_shaped_policy(seed: u64) -> SearchPolicy {
        SearchPolicy::new(
            Box::new(BestFirstPlanner::new()),
            Box::new(LinearEvaluator::default()),
            SearchBudget::best_first(150, 6),
            0.0,
            seed,
        )
    }

    fn engine_obs(seed: u64) -> Observation {
        let mut engine = Engine::new(EngineConfig::default(), seed);
        engine.step(InputFrame::default());
        SearchState::from_snapshot(&engine.snapshot()).expect("active piece present")
    }

    fn origin(decision: Decision) -> (isize, isize) {
        match decision {
            Decision::Place(p) => p.origin(),
            Decision::None => panic!("expected a placement"),
        }
    }

    /// Drive any runner to its decision, returning `(polls, origin)`.
    fn drive(runner: &mut dyn DecisionRunner, obs: Observation) -> (u32, (isize, isize)) {
        runner.submit(obs);
        let mut polls = 1u32;
        loop {
            if let Some(d) = runner.poll() {
                return (polls, origin(d));
            }
            polls += 1;
            assert!(polls <= 1000, "the decision must land");
        }
    }

    #[test]
    fn the_decision_is_invariant_under_the_budget() {
        // The load-bearing property: for ANY budget, the budgeted venue delivers the
        // SAME decision as the blocking venue — the budget moves only *when* the
        // decision lands, never *what* it is.
        let obs = engine_obs(7);
        let blocking = {
            let mut r = SyncRunner::new(Box::new(attack_shaped_policy(1)));
            r.submit(obs.clone());
            origin(r.poll().expect("sync decides at submit"))
        };
        // A spread of (quantum, budget-step) operating points.
        for &(q, step_us) in &[(1u32, 1u64), (8, 50), (16, 250)] {
            let clock = Box::new(FakeClock::new(Duration::from_micros(step_us)));
            let mut r = BudgetedRunner::with_step(
                Box::new(attack_shaped_policy(1)),
                q,
                Duration::from_micros(500),
                clock,
            );
            let (_, got) = drive(&mut r, obs.clone());
            assert_eq!(
                got, blocking,
                "budget (q={q}, step={step_us}us) changed the decision"
            );
        }
    }

    #[test]
    fn an_unreachable_budget_completes_in_one_poll_like_blocking() {
        // Frozen clock ⇒ budget never reached ⇒ the whole decision runs in poll #1,
        // exactly like the blocking venue.
        let obs = engine_obs(7);
        let mut r = BudgetedRunner::new(
            Box::new(attack_shaped_policy(1)),
            Duration::from_micros(8_000),
            Box::new(FrozenClock),
        );
        let (polls, _) = drive(&mut r, obs);
        assert_eq!(polls, 1, "an unreachable budget is a one-shot decision");
    }

    #[test]
    fn a_tight_budget_matches_the_sliced_venue() {
        // A budget that admits exactly one quantum per poll must reproduce the sliced
        // venue's poll count AND decision: step == budget ⇒ the post-think check trips
        // after the first quantum every poll.
        let obs = engine_obs(7);
        let mut sliced = SlicedRunner::with_quantum(Box::new(attack_shaped_policy(3)), 16);
        let (sliced_polls, sliced_origin) = drive(&mut sliced, obs.clone());

        let step = Duration::from_micros(100);
        let mut budgeted = BudgetedRunner::with_step(
            Box::new(attack_shaped_policy(3)),
            16,
            step, // budget == step ⇒ one quantum then yield
            Box::new(FakeClock::new(step)),
        );
        let (budgeted_polls, budgeted_origin) = drive(&mut budgeted, obs);
        assert_eq!(
            budgeted_polls, sliced_polls,
            "one-quantum budget should match sliced cadence"
        );
        assert_eq!(budgeted_origin, sliced_origin);
    }

    #[test]
    fn a_wider_budget_finishes_in_fewer_polls() {
        // The whole point: admitting K quanta per poll cuts the poll count ~K-fold
        // versus one-quantum slicing. step=1us, budget=5us ⇒ ~5 quanta/poll.
        let obs = engine_obs(7);
        let step = Duration::from_micros(1);
        let mut wide = BudgetedRunner::with_step(
            Box::new(attack_shaped_policy(3)),
            16,
            step * 5,
            Box::new(FakeClock::new(step)),
        );
        let (wide_polls, _) = drive(&mut wide, obs.clone());

        let mut tight = BudgetedRunner::with_step(
            Box::new(attack_shaped_policy(3)),
            16,
            step,
            Box::new(FakeClock::new(step)),
        );
        let (tight_polls, _) = drive(&mut tight, obs);
        assert!(
            wide_polls * 2 <= tight_polls,
            "a 5-quantum budget ({wide_polls} polls) should be well under half the 1-quantum cadence ({tight_polls})"
        );
    }

    #[test]
    fn cancel_discards_the_in_flight_think() {
        let obs = engine_obs(7);
        let mut r = BudgetedRunner::new(
            Box::new(attack_shaped_policy(1)),
            Duration::from_micros(100),
            Box::new(FakeClock::new(Duration::from_micros(100))),
        );
        r.submit(obs);
        assert!(r.poll().is_none(), "first poll: still working");
        r.cancel();
        assert!(r.poll().is_none(), "cancelled: nothing to deliver");
    }
}
