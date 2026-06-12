//! The experiment commands behind the `tetr-research` binary.
//!
//! Three registries, three concerns, kept apart:
//!
//! - **Bots** ([`crate::bots`]) say WHO plays — named `BotSpec` instances.
//! - **Evals** (these modules) say WHAT is measured — serde-serialized
//!   `Spec`s with bot SLOTS, bound to bot names by a [`crate::registry`]
//!   entry. A command is `run(spec, bots…, rt) -> Value`: measurement in,
//!   human context on stderr, the result RETURNED — the runner prints the
//!   one-JSON-line stdout contract ({run, eval, bots, …metrics}).
//! - **Tracking** is not a participant: the runner writes the receipt
//!   ([`crate::ledger`]) before dispatch; commands never see it.
//!
//! Optimizers are NOT here: search (climbs, promotion gates) was removed
//! pending a first-principles redesign and will return wrapping these
//! primitives.

use std::path::PathBuf;
use std::time::Duration;

use tetr_core::ai::Cc2Weights;

pub mod cc2_baseline;
pub mod climb_app;
pub mod downstack;
pub mod marathon;
pub mod race;
pub mod versus;

/// Machine-local circumstances of one invocation — everything here may vary
/// between hosts and runs of the SAME experiment without changing what the
/// experiment *is*. Budgets gate stopping points only (a
/// budget-cut run is a prefix of the unbounded one).
#[derive(Clone, Debug, Default, serde::Serialize)]
pub struct Runtime {
    /// Wall-clock bound override; each command documents its default.
    pub budget_secs: Option<u64>,
    /// Path to a Cold Clear 2 binary (`cc2-baseline` only; machine-local).
    pub cc2_bin: Option<PathBuf>,
}

impl Runtime {
    /// The effective wall-clock budget given a command's default.
    pub fn budget(&self, default_secs: u64) -> Duration {
        Duration::from_secs(self.budget_secs.unwrap_or(default_secs))
    }
}

/// CC2 board-parameter vector (the climb's search space).
pub type BoardParams = [f32; Cc2Weights::BOARD_PARAM_COUNT];
