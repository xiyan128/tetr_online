//! Shared combo / attack accounting — the one home of the conventions every
//! suite (marathon, downstack, behavior, versus tests) folds events through.
//!
//! The engine emits attack itself in versus play ([`EngineEvent::AttackSent`]);
//! this fold exists for the solo suites that predate it and as the recorded
//! convention the engine is gated against
//! (`engine_attack_events_match_the_research_fold` in [`crate::versus`]).

use tetr_core::engine::{attack_lines, Engine, EngineEvent, EngineScoreAction};

/// Derive the controller RNG seed from the game seed (decorrelated from the
/// engine's piece stream, but fully determined by it — matches the arena harness).
pub(crate) fn controller_seed(seed: u64) -> u64 {
    seed ^ 0x9E37_79B9_7F4A_7C15
}

/// Lines a scoring action actually cleared (0 for drops / no-clear / a 0-line spin).
///
/// The combo counter must advance on **line clears only**. A hard drop emits its own
/// `ScoreAwarded { action: HardDrop }` (engine `api.rs`), so a loop that bumps combo
/// on every `ScoreAwarded` would inflate combo (and thus attack) by ~1 per piece.
/// Gate combo + attack on `action_clear_lines(action) > 0`.
pub(crate) fn action_clear_lines(action: EngineScoreAction) -> usize {
    match action {
        EngineScoreAction::Single => 1,
        EngineScoreAction::Double => 2,
        EngineScoreAction::Triple => 3,
        EngineScoreAction::Tetris => 4,
        EngineScoreAction::TSpin { lines, .. } => lines,
        EngineScoreAction::SoftDrop
        | EngineScoreAction::HardDrop { .. }
        | EngineScoreAction::NoClear => 0,
    }
}

/// One line clear's accounting, produced by [`fold_combo`].
pub(crate) struct ClearInfo {
    pub action: EngineScoreAction,
    pub back_to_back_bonus: bool,
    pub perfect_clear: bool,
    /// Combo index used for this clear (pre-increment; `0` for the first in a chain).
    pub combo: u32,
    /// Garbage lines this clear sends ([`attack_lines`] with the pre-clear combo).
    pub attack: u32,
}

/// Fold one engine event into the running `combo`, returning the clear it produced (if
/// any). The single home for combo/attack accounting: combo advances on line clears
/// only — a hard drop emits its own `ScoreAwarded` that must NOT bump it — and resets
/// on a clear-less lock.
/// Callers still do their own piece counting / top-out / stats from the same event.
pub(crate) fn fold_combo(
    event: &EngineEvent,
    engine: &Engine,
    combo: &mut u32,
) -> Option<ClearInfo> {
    match event {
        EngineEvent::Locked { lines_cleared, .. } => {
            if *lines_cleared == 0 {
                *combo = 0; // a non-clearing placement breaks the chain
            }
            None
        }
        EngineEvent::ScoreAwarded {
            action,
            back_to_back_bonus,
            ..
        } if action_clear_lines(*action) > 0 => {
            // Post-clear board: empty ⇒ perfect clear. Cheap: no snapshot alloc.
            let perfect_clear = engine.board_is_empty();
            let index = *combo;
            let attack = attack_lines(*action, *back_to_back_bonus, index, perfect_clear);
            *combo += 1;
            Some(ClearInfo {
                action: *action,
                back_to_back_bonus: *back_to_back_bonus,
                perfect_clear,
                combo: index,
                attack,
            })
        }
        _ => None,
    }
}
