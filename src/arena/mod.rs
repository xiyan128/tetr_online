//! Reliable, deterministic play-evaluation for any [`PlayerController`].
//!
//! This is the measurement foundation: it plays a controller through the headless
//! engine and records *how it played*, as trustworthy, reproducible numbers. It is
//! deliberately **bot-agnostic** — it depends only on [`crate::engine`] and the
//! [`PlayerController`] seam, and knows nothing about any particular AI. What you
//! evaluate (a set of greedy weights, a future beam search, a scripted baseline)
//! is expressed as a [`Contender`]; the harness just measures it.
//!
//! ```no_run
//! use tetr_online::arena::{self, Contender, GameSetup, evaluate};
//! use tetr_online::ai::AiController;
//!
//! let bot = Contender::new("greedy-default", |seed| Box::new(AiController::new(
//!     tetr_online::ai::Handicap::perfect(), seed,
//! )));
//! let setup = GameSetup::standard("marathon", 200);   // 200 pieces / game
//! let result = evaluate(&bot, &setup, &arena::seed_set(30));
//! println!("lines: {:.1} ± {:.1}", result.lines.mean, result.lines.std_dev);
//! ```
//!
//! # Reliability guarantees
//!
//! - **Deterministic.** A run is a pure function of `(Contender, GameSetup, seed)`:
//!   same inputs → identical [`GameOutcome`], every time. The engine and the
//!   controllers carry the no-clock / seeded-RNG contract; this harness adds no
//!   entropy of its own. Pinned by tests.
//! - **Terminating.** Every game stops — on top-out, on a piece budget, or on a
//!   hard frame cap — and records *why* ([`Termination`]). No run can hang.
//! - **Self-checking.** Headline totals (lines, score, level) come from the
//!   engine's authoritative snapshot; the per-clear breakdown is tallied from the
//!   event stream; [`play`] reconciles the two and panics on disagreement, so a
//!   miscount surfaces as a failure, never a silently-wrong number.
//!
//! # What this is *not* (yet)
//!
//! It measures *one* implementation reliably (across seeds, with variance — see
//! [`evaluate`]). Comparing many implementations, ranking them, persisting runs
//! across sessions, and the hill-climbing loop are a separate layer built **on
//! top** of this one — kept out so the foundation stays small and trustworthy.
//!
//! Compiled only under the `arena` feature, so it never enters the shipped game.
//!
//! [`PlayerController`]: crate::player::PlayerController

mod outcome;
mod runner;
mod summary;

pub use outcome::{ClearCounts, GameOutcome, Termination};
pub use runner::play;
pub use summary::{evaluate, Evaluation, MetricSummary};

use crate::engine::EngineConfig;
use crate::player::PlayerController;

/// A named AI implementation to evaluate.
///
/// A contender is a **factory**: given a seed, it builds a fresh
/// [`PlayerController`]. Building fresh per game is what keeps games independent
/// and reproducible — no state leaks between runs. The harness never inspects the
/// controller, so *any* `PlayerController` is a valid contender: the real bot, a
/// scripted baseline, or a future search.
pub struct Contender {
    name: String,
    #[allow(clippy::type_complexity)]
    factory: Box<dyn Fn(u64) -> Box<dyn PlayerController> + Send + Sync>,
}

impl Contender {
    /// Define a contender from a display name and a `seed -> controller` factory.
    pub fn new(
        name: impl Into<String>,
        factory: impl Fn(u64) -> Box<dyn PlayerController> + Send + Sync + 'static,
    ) -> Self {
        Self {
            name: name.into(),
            factory: Box::new(factory),
        }
    }

    /// The contender's display name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Build a fresh controller seeded for one game.
    pub fn build(&self, seed: u64) -> Box<dyn PlayerController> {
        (self.factory)(seed)
    }
}

/// The game a run plays: the engine configuration plus how the game terminates.
///
/// `max_pieces` is the evaluation budget — the placement count that defines "a
/// game" for measurement (an endless variant would otherwise never stop). If the
/// controller tops out first, that is recorded instead. `max_frames` is an
/// independent hard safety cap so a stuck controller can never loop forever.
#[derive(Clone, Debug)]
pub struct GameSetup {
    name: String,
    config: EngineConfig,
    max_pieces: usize,
    max_frames: usize,
}

impl GameSetup {
    /// A setup ending after `max_pieces` placements (or an earlier top-out), using
    /// `config`. The frame cap defaults to a generous `512 * max_pieces` (floored
    /// at 4096) — far above any legitimate per-piece frame count.
    pub fn new(name: impl Into<String>, config: EngineConfig, max_pieces: usize) -> Self {
        let max_frames = max_pieces.saturating_mul(512).max(4096);
        Self {
            name: name.into(),
            config,
            max_pieces,
            max_frames,
        }
    }

    /// A setup on the default engine configuration.
    pub fn standard(name: impl Into<String>, max_pieces: usize) -> Self {
        Self::new(name, EngineConfig::default(), max_pieces)
    }

    /// Override the hard frame-cap safety bound.
    pub fn with_frame_cap(mut self, max_frames: usize) -> Self {
        self.max_frames = max_frames;
        self
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn config(&self) -> &EngineConfig {
        &self.config
    }

    pub fn max_pieces(&self) -> usize {
        self.max_pieces
    }

    pub fn max_frames(&self) -> usize {
        self.max_frames
    }
}

/// A deterministic, well-distributed set of `count` seeds.
///
/// Seeds are derived by SplitMix64 over the indices `0..count`, so adjacent seeds
/// don't correlate (raw `0, 1, 2, …` can produce suspiciously similar early piece
/// sequences). The set is fixed — `seed_set(n)` is the same every run — so an
/// evaluation over it is reproducible.
pub fn seed_set(count: usize) -> Vec<u64> {
    (0..count as u64)
        .map(|i| {
            let mut z = i.wrapping_add(0x9E37_79B9_7F4A_7C15);
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{EngineSnapshot, InputFrame};

    struct Idle;
    impl PlayerController for Idle {
        fn poll(&mut self, _snapshot: &EngineSnapshot) -> InputFrame {
            InputFrame {
                dt_seconds: 1.0 / 60.0,
                ..InputFrame::default()
            }
        }
    }

    #[test]
    fn seed_set_is_deterministic_and_distinct() {
        let a = seed_set(50);
        let b = seed_set(50);
        assert_eq!(a, b);
        assert_eq!(a.len(), 50);
        let unique: std::collections::HashSet<_> = a.iter().collect();
        assert_eq!(unique.len(), 50, "seeds must be distinct");
    }

    #[test]
    fn contender_builds_a_fresh_controller() {
        let contender = Contender::new("idle", |_seed| Box::new(Idle));
        assert_eq!(contender.name(), "idle");
        let _controller = contender.build(1);
    }

    #[test]
    fn evaluate_aggregates_over_the_seed_set() {
        let contender = Contender::new("idle", |_seed| Box::new(Idle));
        // A tiny frame cap keeps each game instant; we only check aggregation here.
        let setup = GameSetup::standard("standard", 5).with_frame_cap(3);
        let seeds = seed_set(4);

        let first = evaluate(&contender, &setup, &seeds);
        assert_eq!(first.games.len(), 4);
        assert_eq!(first.pieces.n, 4);
        assert_eq!(first.contender, "idle");

        // Deterministic: re-running the same evaluation reproduces every game.
        let second = evaluate(&contender, &setup, &seeds);
        assert_eq!(first.games, second.games);
    }
}
