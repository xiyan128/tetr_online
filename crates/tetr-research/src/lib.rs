//! `tetr-research` — the headless experiment platform behind the versus bot.
//!
//! Bevy-free: depends only on `tetr-core` (engine + AI seam), so everything
//! here compiles and runs fast enough for iterative loops. The crate is the
//! *instrument layer*: match harnesses, benchmark suites, statistics, and bot
//! construction. An experiment is a NAMED eval ([`registry`]) run on NAMED
//! bots ([`bots`]) — `run <eval> [bots…]`; the [`commands`] behind the evals
//! stay thin: take the spec and the bots, compose library pieces, print a
//! record.
//!
//! # The conventions (read before adding an experiment)
//!
//! - **Determinism.** A game is a pure function of `(bot spec, seed)`: the
//!   engine's 7-bag and the policy RNG are both seeded. Every reported number
//!   must be reproducible from `(commit, eval, bots…)` — all names.
//! - **The registries are the configuration** ([`registry`], [`bots`]).
//!   Everything that can change a result lives in a named eval spec or a
//!   named bot; command-line flags carry only machine-local circumstance
//!   (budgets, paths, resume pointers). Changing results means registering
//!   a new name — `resume` refuses a drifted spec, and dirty-tree runs are
//!   stamped as such.
//! - **Seed-region discipline** ([`seeds`]). Suites draw seeds from disjoint
//!   index regions (train / validation / rotation / SPRT / confirmation) so no
//!   verdict is ever quoted on seeds that influenced a decision.
//! - **Self-bounding.** Every long-running command honours its wall-clock
//!   budget and ends with an honest partial verdict rather than running
//!   unbounded.
//! - **Arm-swapping + CRN.** Paired comparisons play each seed from both
//!   chairs (seed luck and chair order cancel) on common random numbers —
//!   and sequential verdicts treat the chair-swapped pair as ONE observation
//!   ([`sprt`]): the two games share the seed, so counting them as
//!   independent voids the test's error bounds.
//! - **Death decides; the cap tiebreak is biased.** The net-attack tiebreak in
//!   capped games is structurally anti-defensive (cancelled lines count for
//!   nothing), so survival verdicts must come from death-decisive matches —
//!   see [`sprt`] — never from bare capped win rates.
//! - **Run records.** A command's doc header carries the durable results of
//!   its runs (with settings and a [`ledger`] run id), so conclusions outlive
//!   sessions and are never silently re-derived.
//! - **Receipts + events.** The runner stamps every run's coordinates (eval,
//!   bots, spec, runtime, git state) into one `spec.json` ([`ledger`]) and
//!   sinks the game stream into `games.jsonl` ([`events`]): games are the
//!   facts, receipts the parameters, metrics the duckdb queries over both —
//!   nothing is stored that either already determines.
//!
//! # Layout
//!
//! | module | role |
//! |---|---|
//! | [`registry`] | named eval specs as code (one of the two configuration surfaces) |
//! | [`commands`] | the eval executors behind `tetr-research run` |
//! | [`marathon`] | solo scoring/APP suite (the original benchmark) |
//! | [`downstack`] | cheese-clearing suite (digging skill, not gameable by combos) |
//! | [`pc`] | clean-board perfect-clear suite (PPC + per-PC lock indices) |
//! | [`versus`] | head-to-head under the **engine's** garbage rules |
//! | [`versus_legacy`] | the pre-engine harness scheduler, quarantined for the TBP referee + scripted scenarios |
//! | [`sprt`] | pair-level GSPRT over death-decisive seed pairs |
//! | [`bots`] | bot construction + the named bot registry (the other surface) |
//! | [`seeds`] | deterministic seed sets + region discipline |
//! | [`cc2`] | TBP client for baselining Cold Clear 2 as a subprocess |
//! | [`rng`] | the dependency-free deterministic PRNG (SplitMix64) |
//! | [`progress`] | stderr progress bars (cosmetic only, hidden off-TTY) |
//! | [`events`] | the normalized game stream (`games.jsonl`, duckdb-ready) |
//! | [`ledger`] | run receipts |

pub mod bots;
pub mod cc2;
pub mod commands;
pub mod downstack;
pub mod events;
pub mod ledger;
pub mod marathon;
pub mod pc;
pub mod progress;
pub mod registry;
pub mod rng;
pub mod seeds;
pub mod sprt;
pub mod versus;
pub mod versus_legacy;

pub(crate) mod accounting;

// No flat re-exports: every item is imported by module path (one import
// style, and the path carries meaning — `versus_legacy::` is a warning label,
// `seeds::regions::` is the partition). This crate has no external consumers
// and keeps no compatibility surface.
