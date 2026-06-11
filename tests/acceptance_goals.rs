//! Acceptance suite for the guideline §25.9 "Goals (Fixed/Variable)".
//!
//! Each scenario from §25.9 maps to exactly one `#[test]` below. The pure-goal
//! helpers (`fixed_goal_for_level`, `variable_goal_for_level`,
//! `variable_goal_units`) are exercised directly, and the goal-bearing parts of
//! the `Engine` (`goal_remaining` from a fresh `snapshot()`, and the decrement
//! on a line clear) are driven only through the public `Engine::new` / `step` /
//! `snapshot` boundary so the suite stays deterministic.
//!
//! These names are reachable today via the flat `pub use` block in
//! `src/lib.rs` (`tetr_online::<Name>`), so this file does not depend on the
//! pending `pub mod engine;` change.

use tetr_online::{fixed_goal_for_level, variable_goal_for_level, variable_goal_units};
use tetr_online::{
    Engine, EngineConfig, EngineEvent, EngineScoreAction, GoalSystem, InputFrame, PieceType,
    TSpinKind,
};

/// Build a deterministic engine on a 4-wide well with the given goal system and
/// starting level. A 4-wide board is the narrowest well in which a horizontal
/// I-piece fills an entire row, which scenario 8 relies on.
fn engine_with(goal_system: GoalSystem, starting_level: u8) -> Engine {
    let config = EngineConfig {
        board_width: 4,
        goal_system,
        starting_level,
        ..Default::default()
    };
    Engine::new(config, 0)
}

// 1. Fixed: Level 1 goal is 10.
#[test]
fn fixed_level_1_goal_is_10() {
    // Pure helper: the level-1 fixed goal is 10 lines.
    assert_eq!(fixed_goal_for_level(1, 1), 10);

    // And a freshly constructed Fixed engine starting at level 1 reports the
    // same goal before any piece has been driven.
    let engine = engine_with(GoalSystem::Fixed, 1);
    let snapshot = engine.snapshot();
    assert_eq!(snapshot.level, 1);
    assert_eq!(snapshot.goal_remaining, 10);
}

// 2. Fixed: starting at Level 4, the first goal is 40.
#[test]
fn fixed_starting_at_level_4_first_goal_is_40() {
    // Pure helper: starting at level 4, the level-4 goal is prorated to 40.
    assert_eq!(fixed_goal_for_level(4, 4), 40);

    // A Fixed engine started at level 4 reports a remaining goal of 40.
    let engine = engine_with(GoalSystem::Fixed, 4);
    let snapshot = engine.snapshot();
    assert_eq!(snapshot.level, 4);
    assert_eq!(snapshot.goal_remaining, 40);
}

// 3. Fixed: later goals are 10 (e.g. level 5 when starting at level 4).
#[test]
fn fixed_later_goals_are_10() {
    assert_eq!(fixed_goal_for_level(4, 5), 10);
}

// 4. Variable: goal is Level * 5.
#[test]
fn variable_goal_is_level_times_5() {
    assert_eq!(variable_goal_for_level(1), 5);
    assert_eq!(variable_goal_for_level(15), 75);
}

// 5. Variable: total goal over levels 1..=15 is 600.
#[test]
fn variable_total_levels_1_to_15_is_600() {
    let total: usize = (1..=15).map(variable_goal_for_level).sum();
    assert_eq!(total, 600);
}

// 6. Variable: Tetris and T-Spin Single each award 8 units.
#[test]
fn variable_units_tetris_and_tspin_single_award_8() {
    // Tetris (four lines, no spin), without a back-to-back bonus.
    assert_eq!(variable_goal_units(None, 4, false), 8);
    // Full T-Spin Single (one line), without a back-to-back bonus.
    assert_eq!(variable_goal_units(Some(TSpinKind::Full), 1, false), 8);
}

// 7. Variable: T-Spin Double awards 12, T-Spin Triple awards 16.
#[test]
fn variable_units_tspin_double_12_triple_16() {
    assert_eq!(variable_goal_units(Some(TSpinKind::Full), 2, false), 12);
    assert_eq!(variable_goal_units(Some(TSpinKind::Full), 3, false), 16);
}

// 8. Engine: goal_remaining decrements on a line clear.
//
// Mirrors the in-crate unit test `lock_line_clear_scores_single_and_advances_
// fixed_goal` (api.rs), but reaches the result purely through the public
// `step()` boundary instead of the private lock seam: on a 4-wide well a
// horizontal I-piece hard-drops onto the floor, fills the single bottom row,
// and clears a Single. A Fixed engine starting at level 1 therefore moves from
// goal_remaining 10 to 9 with one physical line cleared.
#[test]
fn engine_goal_remaining_decrements_on_line_clear() {
    // The bag generator is seeded, so the first spawned piece is deterministic
    // per seed but not guaranteed to be an I. Search seeds for the first one
    // whose opening hard-drop on a 4-wide well clears exactly one line; the
    // search itself is deterministic (same seeds, same order, every run).
    let mut cleared = false;
    for seed in 0..64u64 {
        let config = EngineConfig {
            board_width: 4,
            goal_system: GoalSystem::Fixed,
            starting_level: 1,
            ..Default::default()
        };
        let mut engine = Engine::new(config, seed);

        // Confirm the precondition: a fresh Fixed L1 engine owes 10 lines.
        assert_eq!(engine.snapshot().goal_remaining, 10);

        // First step with no flags only spawns the opening piece (plus its
        // immediate one-row drop), so we learn its type — from the snapshot,
        // where spawns are observed — before committing.
        engine.step(InputFrame::default());
        let first_piece = engine.snapshot().active.map(|active| active.piece_type);
        // A horizontal I is the only piece that fills all four columns of a
        // 4-wide row in one drop; skip any other opening piece.
        if first_piece != Some(PieceType::I) {
            continue;
        }

        // Hard-drop the spawned I-piece straight down onto the empty floor.
        let drop_events = engine.step(InputFrame {
            hard_drop: true,
            ..Default::default()
        });

        // The lock must report exactly one cleared line and a Single score.
        assert!(
            drop_events.iter().any(|event| matches!(
                event,
                EngineEvent::Locked {
                    piece_type: PieceType::I,
                    lines_cleared: 1,
                }
            )),
            "expected a single-line Locked event, got {drop_events:?}"
        );
        assert!(
            drop_events.iter().any(|event| matches!(
                event,
                EngineEvent::ScoreAwarded {
                    action: EngineScoreAction::Single,
                    ..
                }
            )),
            "expected a Single ScoreAwarded event, got {drop_events:?}"
        );

        // The Single consumes one goal unit (10 -> 9) and records one physical
        // line cleared, matching api.rs::lock_line_clear_scores_single.
        let snapshot = engine.snapshot();
        assert_eq!(snapshot.lines, 1);
        assert_eq!(snapshot.goal_remaining, 9);

        cleared = true;
        break;
    }

    assert!(
        cleared,
        "no seed in 0..64 spawned an opening I-piece for the 4-wide well"
    );
}
