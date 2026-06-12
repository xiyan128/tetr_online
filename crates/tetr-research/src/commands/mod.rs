//! The experiment commands behind the `tetr-research` binary.
//!
//! Three registries, three concerns, kept apart:
//!
//! - **Bots** ([`crate::bots`]) say WHO plays — named `BotSpec` instances.
//! - **Evals** (these modules) say WHAT is measured — serde-serialized
//!   `Spec`s with bot SLOTS, bound to bot names by a [`crate::registry`]
//!   entry. A command is `run(spec, bots…, rt)`: measurement in, machine
//!   lines on stdout, context on stderr, nothing else.
//! - **Tracking** is not a participant: the runner writes the receipt
//!   ([`crate::ledger`]) before dispatch; only the climb touches a run
//!   directory, and only for its resume checkpoint.
//!
//! The climb is the one non-eval: an optimizer that MUTATES a subject bot's
//! params against an objective, gated by screen → confirm → anchor.

use std::path::PathBuf;
use std::time::Duration;

use tetr_core::ai::Cc2Weights;

pub mod awareness;
pub mod behavior;
pub mod cc2_baseline;
pub mod climb;
pub mod downstack;
pub mod marathon;
pub mod panel;
pub mod race;
pub mod runs;
pub mod versus;

/// Machine-local circumstances of one invocation — everything here may vary
/// between hosts and runs of the SAME experiment without changing what the
/// experiment *is*. Budgets gate stopping points only (a budget-cut walk is
/// a prefix of the unbounded one; `resume` continues it bit-identically).
#[derive(Clone, Debug, Default, serde::Serialize)]
pub struct Runtime {
    /// Wall-clock bound override; each command documents its default.
    pub budget_secs: Option<u64>,
    /// Climb only: new iterations allowed this invocation (0 = unbounded).
    pub max_iters: u32,
    /// Path to a Cold Clear 2 binary (`cc2-baseline` only; machine-local).
    pub cc2_bin: Option<PathBuf>,
    /// Set by the `resume` verb: the prior run directory whose checkpoint
    /// this invocation continues.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub resume_from: Option<PathBuf>,
}

impl Runtime {
    /// The effective wall-clock budget given a command's default.
    pub fn budget(&self, default_secs: u64) -> Duration {
        Duration::from_secs(self.budget_secs.unwrap_or(default_secs))
    }
}

/// CC2 board-parameter vector (the climb's search space).
pub type BoardParams = [f32; Cc2Weights::BOARD_PARAM_COUNT];
