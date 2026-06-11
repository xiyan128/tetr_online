//! Headless **marathon** evaluation: score/sec and attack-per-piece for a solo
//! bot playing Marathon (`GoalSystem::Variable`, end at `MAX_LEVEL`), measured
//! at `Handicap::perfect()` so it reflects *policy quality*, not the in-game
//! reaction handicap.
//!
//! Determinism: a game is a pure function of `(bot factory, seed)` — the
//! engine's 7-bag and the policy RNG are both seeded. Re-running an evaluation
//! reproduces every number.

use tetr_core::engine::{Engine, EngineConfig, EngineEvent, GoalSystem, MAX_LEVEL};
use tetr_core::player::{drive_engine, PlayerController};

use crate::accounting::{controller_seed, fold_combo};

/// Fixed simulation rate: one engine step (one `drive_engine` poll) = 1/60 s.
pub const SIM_HZ: f32 = 60.0;

/// Generous per-game frame cap (≈ 4.6 hours of sim time) so a stalling bot can
/// never hang the harness. Marathon normally ends far sooner (level 15 or top-out).
pub const DEFAULT_MAX_FRAMES: u32 = 1_000_000;

/// Engine configuration for a Marathon game: the Variable goal system (what
/// `Variant::Marathon` applies) on the default board. Marathon ends when the
/// snapshot level reaches `MAX_LEVEL`.
pub fn marathon_config() -> EngineConfig {
    EngineConfig {
        goal_system: GoalSystem::Variable,
        ..EngineConfig::default()
    }
}

/// The result of one Marathon game.
#[derive(Debug, Clone, Copy)]
pub struct MarathonOutcome {
    pub seed: u64,
    pub score: u32,
    pub level: u8,
    pub lines: u32,
    pub pieces: u32,
    pub frames: u32,
    pub topped_out: bool,
    /// Reached `MAX_LEVEL` without topping out (a "won" marathon).
    pub completed: bool,
    /// Total garbage lines sent (guideline attack table) over the game — the
    /// versus-relevant quantity. `attack_per_piece()` is the APP efficiency metric
    /// we compare against Cold Clear 2.
    pub total_attack: u32,
}

impl MarathonOutcome {
    pub fn elapsed_seconds(&self) -> f32 {
        self.frames as f32 / SIM_HZ
    }

    /// The headline marathon metric: score accumulated per simulated second.
    pub fn score_per_second(&self) -> f32 {
        let t = self.elapsed_seconds();
        if t > 0.0 {
            self.score as f32 / t
        } else {
            0.0
        }
    }

    /// Attack per piece (APP): garbage lines sent ÷ pieces placed — the standard
    /// offensive-efficiency metric for comparing versus bots (vs Cold Clear 2).
    pub fn attack_per_piece(&self) -> f32 {
        if self.pieces > 0 {
            self.total_attack as f32 / self.pieces as f32
        } else {
            0.0
        }
    }
}

/// Play one Marathon game to completion (`MAX_LEVEL`), top-out, the frame cap, or
/// `max_pieces` placements — the fast `/autoresearch` metric path. A piece cap turns
/// a full ~930-piece marathon into a quick proxy while keeping score/sec meaningful
/// as an early-game scoring rate. Pass `u32::MAX` for the full, uncapped marathon
/// (final validation). `make_bot` builds a fresh controller per (controller) seed,
/// so games stay independent and reproducible.
pub fn play_marathon_capped(
    make_bot: &dyn Fn(u64) -> Box<dyn PlayerController>,
    seed: u64,
    max_frames: u32,
    max_pieces: u32,
) -> MarathonOutcome {
    let mut engine = Engine::new(marathon_config(), seed);
    let mut bot = make_bot(controller_seed(seed));

    let mut pieces = 0u32;
    let mut frames = 0u32;
    let mut topped = false;
    // Versus attack accounting (guideline table). `combo` = consecutive
    // line-clearing placements; `total_attack` sums garbage lines sent.
    let mut combo = 0u32;
    let mut total_attack = 0u32;

    while frames < max_frames {
        frames += 1;
        let mut locked = false;
        for event in drive_engine(&mut engine, &mut *bot) {
            if let Some(clear) = fold_combo(&event, &engine, &mut combo) {
                total_attack += clear.attack;
            }
            match &event {
                EngineEvent::Locked { .. } => {
                    pieces += 1;
                    locked = true;
                }
                EngineEvent::GameOver { .. } => topped = true,
                _ => {}
            }
        }
        if topped {
            break;
        }
        // Level only rises on a line clear (at a lock), so checking then suffices
        // and avoids a full snapshot every frame.
        if locked && engine.snapshot().level >= MAX_LEVEL {
            break;
        }
        // Fast-metric cap: stop after a bounded number of placements (u32::MAX = off).
        if pieces >= max_pieces {
            break;
        }
    }

    let snap = engine.snapshot();
    MarathonOutcome {
        seed,
        score: snap.score as u32,
        level: snap.level,
        lines: snap.lines as u32,
        pieces,
        frames,
        topped_out: topped,
        completed: snap.level >= MAX_LEVEL && !topped,
        total_attack,
    }
}

/// Aggregate statistics over a seed set.
#[derive(Debug, Clone)]
pub struct MarathonStats {
    pub games: usize,
    pub mean_score_per_second: f32,
    pub mean_score: f32,
    pub mean_level: f32,
    pub mean_pieces: f32,
    pub completion_rate: f32,
    pub topout_rate: f32,
    /// Mean attack per piece (APP) — the versus offensive-efficiency metric.
    pub mean_attack_per_piece: f32,
    /// Mean total attack (garbage lines sent) per game.
    pub mean_attack: f32,
    pub outcomes: Vec<MarathonOutcome>,
}

/// Evaluate a bot over `seeds`, returning aggregate Marathon stats.
pub fn evaluate(
    make_bot: &dyn Fn(u64) -> Box<dyn PlayerController>,
    seeds: &[u64],
    max_frames: u32,
) -> MarathonStats {
    evaluate_capped(make_bot, seeds, max_frames, u32::MAX)
}

/// Like [`evaluate`] but with a per-game `max_pieces` cap — the fast metric path
/// the `/autoresearch` loop uses (full uncapped marathon = `u32::MAX`).
pub fn evaluate_capped(
    make_bot: &dyn Fn(u64) -> Box<dyn PlayerController>,
    seeds: &[u64],
    max_frames: u32,
    max_pieces: u32,
) -> MarathonStats {
    let outcomes: Vec<MarathonOutcome> = seeds
        .iter()
        .map(|&seed| play_marathon_capped(make_bot, seed, max_frames, max_pieces))
        .collect();

    let n = outcomes.len().max(1) as f32;
    let sum = |f: &dyn Fn(&MarathonOutcome) -> f32| outcomes.iter().map(f).sum::<f32>();

    MarathonStats {
        games: outcomes.len(),
        mean_score_per_second: sum(&|o| o.score_per_second()) / n,
        mean_score: sum(&|o| o.score as f32) / n,
        mean_level: sum(&|o| o.level as f32) / n,
        mean_pieces: sum(&|o| o.pieces as f32) / n,
        completion_rate: sum(&|o| if o.completed { 1.0 } else { 0.0 }) / n,
        topout_rate: sum(&|o| if o.topped_out { 1.0 } else { 0.0 }) / n,
        mean_attack_per_piece: sum(&|o| o.attack_per_piece()) / n,
        mean_attack: sum(&|o| o.total_attack as f32) / n,
        outcomes,
    }
}
