//! Headless perfect-clear evaluation on a clean board.
//!
//! The headline metric is **perfect clears per piece (PPC)**, counted only when
//! the engine's post-clear board is actually empty (`crate::accounting`'s
//! engine-true fold — never the planner's own claim). Attack per piece and
//! top-out rate ride along as failed-attempt quality measures: a PC policy is
//! useless if its misses destroy ordinary play. Every game also records the
//! piece index of each PC, so opener-vs-sustained behavior reads straight off
//! the game stream with no extra eval kind.

use tetr_core::engine::{Engine, EngineEvent};
use tetr_core::player::{PlayerController, drive_engine};

use crate::accounting::{controller_seed, fold_combo};
use crate::marathon::{DEFAULT_MAX_FRAMES, marathon_config};

/// One PC game's outcome (the per-game event row).
#[derive(Debug, Clone, serde::Serialize)]
pub struct PcOutcome {
    pub seed: u64,
    pub pieces: u32,
    pub perfect_clears: u32,
    /// Piece count at the moment of each PC (0-based lock index) — the trace
    /// that separates opener PCs from mid-game ones.
    pub pc_piece_indices: Vec<u32>,
    pub total_attack: u32,
    pub frames: u32,
    pub topped_out: bool,
}

/// Play one piece-capped game and count engine-true perfect clears.
pub fn play_pc_capped(
    make_bot: &dyn Fn(u64) -> Box<dyn PlayerController>,
    seed: u64,
    max_pieces: u32,
) -> PcOutcome {
    let mut engine = Engine::new(marathon_config(), seed);
    let mut bot = make_bot(controller_seed(seed));
    let mut pieces = 0u32;
    let mut perfect_clears = 0u32;
    let mut pc_piece_indices = Vec::new();
    let mut total_attack = 0u32;
    let mut frames = 0u32;
    let mut topped_out = false;
    let mut combo = 0u32;

    while pieces < max_pieces && frames < DEFAULT_MAX_FRAMES {
        frames += 1;
        for event in drive_engine(&mut engine, &mut *bot) {
            if let Some(clear) = fold_combo(&event, &engine, &mut combo) {
                total_attack += clear.attack;
                if clear.perfect_clear {
                    perfect_clears += 1;
                    pc_piece_indices.push(pieces);
                }
            }
            match event {
                EngineEvent::Locked { .. } => pieces += 1,
                EngineEvent::GameOver { .. } => topped_out = true,
                _ => {}
            }
        }
        if topped_out {
            break;
        }
    }

    PcOutcome {
        seed,
        pieces,
        perfect_clears,
        pc_piece_indices,
        total_attack,
        frames,
        topped_out,
    }
}

/// Aggregates over a set of PC games. PPC and APP are piece-weighted (totals
/// over totals), so long games count for what they are.
#[derive(Debug, Clone)]
pub struct PcStats {
    pub games: usize,
    pub pieces: u64,
    pub perfect_clears: u64,
    pub total_attack: u64,
    pub topouts: usize,
    pub outcomes: Vec<PcOutcome>,
}

impl PcStats {
    pub fn pc_per_piece(&self) -> f64 {
        if self.pieces == 0 {
            0.0
        } else {
            self.perfect_clears as f64 / self.pieces as f64
        }
    }

    pub fn attack_per_piece(&self) -> f64 {
        if self.pieces == 0 {
            0.0
        } else {
            self.total_attack as f64 / self.pieces as f64
        }
    }

    pub fn topout_rate(&self) -> f64 {
        if self.games == 0 {
            0.0
        } else {
            self.topouts as f64 / self.games as f64
        }
    }
}

pub fn summarize(outcomes: Vec<PcOutcome>) -> PcStats {
    PcStats {
        games: outcomes.len(),
        pieces: outcomes.iter().map(|o| u64::from(o.pieces)).sum(),
        perfect_clears: outcomes.iter().map(|o| u64::from(o.perfect_clears)).sum(),
        total_attack: outcomes.iter().map(|o| u64::from(o.total_attack)).sum(),
        topouts: outcomes.iter().filter(|o| o.topped_out).count(),
        outcomes,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aggregate_ppc_is_weighted_by_pieces() {
        let stats = summarize(vec![
            PcOutcome {
                seed: 1,
                pieces: 10,
                perfect_clears: 1,
                pc_piece_indices: vec![9],
                total_attack: 10,
                frames: 1,
                topped_out: false,
            },
            PcOutcome {
                seed: 2,
                pieces: 30,
                perfect_clears: 1,
                pc_piece_indices: vec![20],
                total_attack: 20,
                frames: 1,
                topped_out: true,
            },
        ]);
        assert_eq!(stats.pieces, 40);
        assert_eq!(stats.perfect_clears, 2);
        assert!((stats.pc_per_piece() - 0.05).abs() < f64::EPSILON);
        assert!((stats.attack_per_piece() - 0.75).abs() < f64::EPSILON);
        assert!((stats.topout_rate() - 0.5).abs() < f64::EPSILON);
    }
}
