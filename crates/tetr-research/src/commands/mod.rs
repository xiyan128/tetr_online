//! The experiment commands behind the `tetr-research` binary.
//!
//! ```text
//! cargo run --release -p tetr-research -- run cc2-board-climb
//! ```
//!
//! Design rules (the crate conventions in `lib.rs` bind every command):
//!
//! - **Configuration is the registry, not arguments.** Everything that can
//!   change a result lives in a named [`crate::registry`] entry — a plain
//!   Rust literal. A recorded run reproduces from `(commit, name)`; changing
//!   results means adding a name, never mutating one with recorded runs
//!   (the resume path enforces this by refusing drifted specs).
//! - **Runtime ≠ identity.** The only command-line flags are machine-local
//!   circumstances that bound *how much* of a deterministic experiment this
//!   invocation materializes — wall-clock budget, iteration cap, resume
//!   pointer, binary paths — never *which* experiment runs ([`Runtime`]).
//! - **Specs are data.** Each command owns a serde-serialized `Spec` struct;
//!   the ledger records it verbatim in `spec.json`, so the manifest IS the
//!   configuration, typed.
//! - Commands stay thin: take the spec, compose library pieces, print the
//!   machine lines on stdout and human context on stderr, write outcomes and
//!   a summary into the run ledger handed to them.

use std::path::PathBuf;
use std::time::Duration;

use tetr_core::ai::Cc2Weights;

pub mod ab;
pub mod behavior;
pub mod cc2_baseline;
pub mod cc2_native;
pub mod climb;
pub mod marathon;
pub mod metric;
pub mod promote;
pub mod runs;
pub mod sprt;

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

/// CC2 board-parameter vector (the climb's search space and the candidate
/// payload of races and promotions).
pub type BoardParams = [f32; Cc2Weights::BOARD_PARAM_COUNT];

/// Beam search shape — the bot-construction pair every spec spells the same
/// way.
#[derive(Clone, Copy, Debug, serde::Serialize)]
pub struct Beam {
    pub width: usize,
    pub depth: u8,
}

impl Default for Beam {
    fn default() -> Self {
        Self {
            width: 16,
            depth: 2,
        }
    }
}
