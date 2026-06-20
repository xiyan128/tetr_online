//! Shared experiment fixtures.
//!
//! A single source of truth for the "realistic mid-game state bank" the compute-axis and
//! depth-stabilization experiments both measure over (`examples/elo_pareto.rs`,
//! `examples/depth_probe.rs`) — previously duplicated, which let the two studies silently
//! drift apart despite sharing a seed.

use tetr_core::ai::SearchState;
use tetr_core::engine::{Engine, EngineEvent};
use tetr_core::player::drive_engine;

use crate::bots::BotSpec;
use crate::marathon::marathon_config;

/// Fixed seed for the deterministic state bank.
const STATE_SEED: u64 = 0x0E10_0BEE;
/// Pieces played before the first snapshot — skips the near-empty opening boards (no real
/// decision pressure) that would otherwise contaminate a "mid-game" bank.
const WARMUP_PIECES: usize = 5;

/// A bank of `n` realistic mid-game [`SearchState`]s: the `driver` bot plays a solo marathon and
/// the board is snapshotted at the start of each piece, after a short warm-up. Deterministic.
pub fn state_bank(n: usize, driver: BotSpec) -> Vec<SearchState> {
    let mut engine = Engine::new(marathon_config(), STATE_SEED);
    let mut bot = driver.factory()(STATE_SEED);
    let mut states = Vec::with_capacity(n);
    let mut piece = 0usize;
    'outer: while states.len() < n {
        let snap = engine.snapshot();
        if snap.game_over.is_some() {
            break;
        }
        if piece >= WARMUP_PIECES
            && let Some(s) = SearchState::from_snapshot(&snap)
        {
            states.push(s);
        }
        // Drive one piece to lock.
        for _ in 0..4000 {
            let mut locked = false;
            for ev in drive_engine(&mut engine, &mut *bot) {
                match ev {
                    EngineEvent::Locked { .. } => locked = true,
                    EngineEvent::GameOver { .. } => break 'outer,
                    _ => {}
                }
            }
            if locked {
                break;
            }
        }
        piece += 1;
    }
    states
}
