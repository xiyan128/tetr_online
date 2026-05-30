//! The compute seam: where the placement search runs (AI3.5).
//!
//! [`ComputeRunner`] decouples *what* the AI searches (the [`Planner`] over a
//! [`SearchState`]) from *where* that search executes. The controller submits a
//! search and polls for the result; it never blocks. This one interface is meant
//! to back three implementations (M2 plan, "Cross-platform compute"):
//!
//! - **[`SyncRunner`] (shipped now).** Runs the planner inline, to completion, in
//!   [`submit`](ComputeRunner::submit) and hands the plan straight to the next
//!   [`poll`](ComputeRunner::poll). Tier-1 greedy is microseconds, so off-thread
//!   machinery would be pure overhead — this is the right tool today.
//! - **Native off-thread (future, `native.rs`).** Spawn the search on
//!   `AsyncComputeTaskPool::get().spawn(..)`, hold the `Task<PlacementPlan>`, and
//!   poll it with `block_on(future::poll_once(&mut task))` each frame (Appendix A).
//!   Dropping the task on [`cancel`](ComputeRunner::cancel) aborts a stale search.
//!   Worth building only when a Tier-2 beam makes the search frame-expensive.
//! - **Web cooperative time-slice (future, `web.rs`).** `wasm32` is single-threaded,
//!   so advance the planner a bounded [`SearchBudget::nodes`] per `poll` and resume
//!   next frame (this is why [`Planner::plan`] can return
//!   [`PlannerStep::NeedMoreBudget`]). Cooperative slicing keeps determinism: a
//!   fixed per-frame node budget with no wall-clock is reproducible.
//!
//! The trait shape (submit / non-blocking poll / cancel) is exactly Cold Clear's
//! own off-thread `request` / `poll` / cancel model, so swapping `SyncRunner` for
//! an async runner later is a controller-internal change — no caller sees it.
//!
//! # Determinism
//!
//! A runner is pure plumbing: it must not introduce RNG or a clock and must feed
//! the planner an owned [`SearchState`] snapshot, never live engine state. The
//! synchronous runner trivially preserves the engine's no-rand / no-time
//! determinism; an off-thread runner preserves it by moving an owned snapshot into
//! the worker (Appendix A).

#[cfg(not(target_arch = "wasm32"))]
pub mod native;
pub mod sync;

#[cfg(not(target_arch = "wasm32"))]
pub use native::ThreadedRunner;
pub use sync::SyncRunner;

use crate::ai::search::{PlacementPlan, SearchBudget};
use crate::ai::state::SearchState;

/// Where the placement search runs. The controller drives it as
/// `submit` once per planning round, then `poll` every frame until a plan
/// appears; [`cancel`](ComputeRunner::cancel) drops an in-flight search whose
/// state went stale (the active piece changed).
///
/// `Send` so an off-thread implementation can live behind the same controller
/// field; the shipped [`SyncRunner`] is trivially `Send`.
pub trait ComputeRunner: Send {
    /// Begin (or replace) a search of `state` under `budget`. A previous
    /// in-flight search is superseded. For [`SyncRunner`] this runs the whole
    /// search immediately and stashes the result for the next [`poll`](Self::poll).
    fn submit(&mut self, state: SearchState, budget: SearchBudget);

    /// Non-blocking: return the finished plan if one is ready, else `None`.
    ///
    /// `Some(Some(plan))` is a placement to play; `Some(None)` is a *completed*
    /// search that found **no** legal placement (board topped out); `None` means
    /// "still working, ask again next frame". The controller takes the result, so
    /// a second `poll` without an intervening `submit` returns `None`.
    fn poll(&mut self) -> Option<Option<PlacementPlan>>;

    /// Abandon any in-flight or buffered result (the state it was computed from is
    /// stale). After this, [`poll`](Self::poll) returns `None` until the next
    /// [`submit`](Self::submit). An off-thread runner drops its `Task` here.
    fn cancel(&mut self);
}
