//! `tetr-research` — the headless experiment platform behind the versus bot.
//!
//! Bevy-free: depends only on `tetr-core` (engine + AI seam), so everything
//! here compiles and runs fast enough for iterative loops. The crate is the
//! *instrument layer*: match harnesses, benchmark suites, statistics, and bot
//! construction. Experiments themselves live in `src/bin/` and stay thin —
//! parse env, compose library pieces, print a record.
//!
//! # The conventions (read before adding an experiment)
//!
//! - **Determinism.** A game is a pure function of `(bot spec, seed)`: the
//!   engine's 7-bag and the policy RNG are both seeded. Every reported number
//!   must be reproducible from code + env.
//! - **Seed-region discipline** ([`seeds`]). Suites draw seeds from disjoint
//!   index regions (train / validation / rotation / SPRT / confirmation) so no
//!   verdict is ever quoted on seeds that influenced a decision.
//! - **Self-bounding.** Every long-running bin honours `TIME_BUDGET_SECS` and
//!   ends with an honest partial verdict rather than running unbounded.
//! - **Arm-swapping + CRN.** Paired comparisons play each seed from both
//!   chairs (seed luck and chair order cancel) on common random numbers.
//! - **Death decides; the cap tiebreak is biased.** The net-attack tiebreak in
//!   capped games is structurally anti-defensive (cancelled lines count for
//!   nothing), so survival verdicts must come from death-decisive matches —
//!   see [`sprt`] — never from bare capped win rates.
//! - **Run records.** A bin's doc header carries the durable results of its
//!   runs (with settings and a [`ledger`] run id), so conclusions outlive
//!   sessions and are never silently re-derived.
//! - **Run manifests.** Every bin writes resolved config, provenance, per-seed
//!   outcomes, and a terminal summary through [`ledger`].
//!
//! # Layout
//!
//! | module | role |
//! |---|---|
//! | [`marathon`] | solo scoring/APP suite (the original benchmark) |
//! | [`downstack`] | cheese-clearing suite (digging skill, not gameable by combos) |
//! | [`versus`] | head-to-head under the **engine's** garbage rules |
//! | [`versus_legacy`] | the pre-engine harness scheduler, quarantined for the TBP referee + scripted scenarios |
//! | [`behavior`] | APP / DS-P metrics across garbage scenarios |
//! | [`sprt`] | Wald's sequential test over death-decisive matches |
//! | [`bots`] | bot construction (one home for the strength conventions) |
//! | [`seeds`] | deterministic seed sets + region discipline |
//! | [`cc2`] | TBP client for baselining Cold Clear 2 as a subprocess |
//! | [`cli`] | env-config + deterministic-RNG helpers for the bins |
//! | [`ledger`] | machine-readable run specs, outcomes, checkpoints, and summaries |

pub mod behavior;
pub mod bots;
pub mod cc2;
pub mod cli;
pub mod downstack;
pub mod ledger;
pub mod marathon;
pub mod seeds;
pub mod sprt;
pub mod versus;
pub mod versus_legacy;

pub(crate) mod accounting;

// No flat re-exports: every item is imported by module path (one import
// style, and the path carries meaning — `versus_legacy::` is a warning label,
// `seeds::regions::` is the partition). This crate has no external consumers
// and keeps no compatibility surface.
