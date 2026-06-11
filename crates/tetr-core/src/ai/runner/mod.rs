//! The compute seam: the **venue** where the AI's decision runs (AI3.5).
//!
//! [`DecisionRunner`] decouples *what* the AI decides (a [`Policy`](crate::ai::Policy)
//! over an [`Observation`]) from *where* that decision is computed. The controller
//! submits an observation and polls for the [`Decision`]; it never blocks.
//!
//! Two venues ship, one per regime:
//!
//! - **[`SyncRunner`]** — blocking direct-drive: the policy runs inline, to
//!   completion, in [`submit`](DecisionRunner::submit). The venue for headless
//!   benchmarks and tests, where exact budgets and zero pacing matter and a
//!   frame doesn't exist to hitch.
//! - **[`SlicedRunner`]** — cooperative interactive: each
//!   [`poll`](DecisionRunner::poll) spends one bounded node quantum on the
//!   policy's in-flight thinking, so a heavy search spreads across frames
//!   instead of stalling the one that submitted it. The venue for the game and
//!   the wasm embed.
//!
//! The trait shape (submit / non-blocking poll / cancel) is exactly Cold Clear's
//! own off-thread `request` / `poll` / cancel model, so the remaining venues — a
//! native thread that thinks continuously between polls, a Web Worker speaking
//! the same protocol over `postMessage` — drop in as a controller-internal
//! change that no caller sees. [`take_now`](DecisionRunner::take_now) is the
//! anytime valve those venues share: the best decision available *right now*,
//! for a deadline-pressed caller (lock-timer pressure, a versus pace cap).
//!
//! # Determinism
//!
//! A runner is pure plumbing: it must not introduce RNG or a clock. It feeds the
//! policy an owned [`Observation`] (never live engine state); the policy's own
//! seeded RNG makes the decision reproducible regardless of which thread runs it.
//! [`SlicedRunner`]'s quantum is a *configured node count*, never a measured
//! time, so a sliced game is reproducible from `(seed, quantum, poll cadence)`.

pub mod sliced;
pub mod sync;

pub use sliced::SlicedRunner;
pub use sync::SyncRunner;

use crate::ai::policy::{Decision, Observation};

/// Where the AI's decision is computed. The controller drives it as `submit` once
/// per piece, then `poll` every frame until a [`Decision`] appears;
/// [`cancel`](DecisionRunner::cancel) drops an in-flight computation whose
/// observation went stale (the active piece changed).
///
/// `Send` so an off-thread implementation can live behind the same controller field;
/// the shipped runners are trivially `Send`.
pub trait DecisionRunner: Send {
    /// Begin (or replace) a decision for `obs`. A previous in-flight computation is
    /// superseded. For [`SyncRunner`] this runs the policy immediately and stashes
    /// the decision for the next [`poll`](Self::poll); for [`SlicedRunner`] the
    /// work happens in the polls.
    fn submit(&mut self, obs: Observation);

    /// Non-blocking: the finished [`Decision`] if one is ready, else `None` ("still
    /// working, ask again next frame"). A completed computation that found no legal
    /// move yields `Some(Decision::None)`. The controller takes the decision, so a
    /// second `poll` without an intervening `submit` returns `None`.
    fn poll(&mut self) -> Option<Decision>;

    /// Abandon any in-flight or buffered decision (its observation is stale). After
    /// this, [`poll`](Self::poll) returns `None` until the next
    /// [`submit`](Self::submit). An off-thread runner drops its worker here.
    fn cancel(&mut self);
}
