//! The compute seam: where the AI's decision is computed (AI3.5).
//!
//! [`DecisionRunner`] decouples *what* the AI decides (a [`Policy`](crate::ai::Policy)
//! over an [`Observation`]) from *where* that decision is computed. The controller
//! submits an observation and polls for the [`Decision`]; it never blocks. This one
//! interface backs three implementations:
//!
//! - **[`SyncRunner`] (shipped).** Runs the policy inline, to completion, in
//!   [`submit`](DecisionRunner::submit) and hands the decision to the next
//!   [`poll`](DecisionRunner::poll). Tier-1 greedy is microseconds, so off-thread
//!   machinery would be pure overhead — this is the right tool today.
//! - **[`ThreadedRunner`] (native, `native.rs`).** Runs the policy on a worker
//!   thread (proves the seam; a Tier-2 beam or a heavy neural forward would use it).
//! - **Web cooperative time-slice (future, `web.rs`).** `wasm32` is single-threaded,
//!   so advance the policy a bounded slice per `poll` and resume next frame.
//!
//! The trait shape (submit / non-blocking poll / cancel) is exactly Cold Clear's
//! own off-thread `request` / `poll` / cancel model, so swapping `SyncRunner` for an
//! async runner later is a controller-internal change — no caller sees it.
//!
//! # Determinism
//!
//! A runner is pure plumbing: it must not introduce RNG or a clock. It feeds the
//! policy an owned [`Observation`] (never live engine state); the policy's own
//! seeded RNG makes the decision reproducible regardless of which thread runs it.

#[cfg(not(target_arch = "wasm32"))]
pub mod native;
pub mod sync;

#[cfg(not(target_arch = "wasm32"))]
pub use native::ThreadedRunner;
pub use sync::SyncRunner;

use crate::ai::policy::{Decision, Observation};

/// Where the AI's decision is computed. The controller drives it as `submit` once
/// per piece, then `poll` every frame until a [`Decision`] appears;
/// [`cancel`](DecisionRunner::cancel) drops an in-flight computation whose
/// observation went stale (the active piece changed).
///
/// `Send` so an off-thread implementation can live behind the same controller field;
/// the shipped [`SyncRunner`] is trivially `Send`.
pub trait DecisionRunner: Send {
    /// Begin (or replace) a decision for `obs`. A previous in-flight computation is
    /// superseded. For [`SyncRunner`] this runs the policy immediately and stashes
    /// the decision for the next [`poll`](Self::poll).
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
