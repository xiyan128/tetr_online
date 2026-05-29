//! Acceptance suite for reference_guideline.md §25.8 "Scoring (Level 1)" plus
//! the canonical Back-to-Back example whose total is 5400 (§13.3).
//!
//! Every scenario is one `#[test]` driven through the public engine boundary.
//! The base scoring table (§13.1) at Level 1 is:
//!
//! | Action             | Points |
//! | ------------------ | -----: |
//! | Single             |    100 |
//! | Double             |    300 |
//! | Triple             |    500 |
//! | Tetris             |    800 |
//! | Mini T-Spin        |    100 |
//! | Mini T-Spin Single |    200 |
//! | T-Spin             |    400 |
//! | T-Spin Single      |    800 |
//! | T-Spin Double      |   1200 |
//! | T-Spin Triple      |   1600 |
//!
//! Back-to-Back (§13.2): the qualifying actions are Tetris, Mini T-Spin Single,
//! T-Spin Single/Double/Triple. The first qualifying clear starts the chain with
//! no bonus; each subsequent qualifying clear while the chain is active adds
//! `floor(0.5 * base)`. Single/Double/Triple break the chain; zero-line T-Spins
//! neither start nor break it.
//!
//! ## Determinism & board setup
//! A fixed seed feeds the seven-bag RNG, so the next-queue is reproducible. These
//! scenarios are board-precondition heavy: they need specific wells and T-slots
//! that cannot be reached deterministically through `step()` alone. They use the
//! public test seam on `Engine` (`set_cell` to paint the board, and
//! `lock_active_for_test` to lock a hand-placed `ActivePiece` through the real
//! lock/clear/score path — the same path the in-crate unit tests
//! `lock_line_clear_scores_single_and_advances_fixed_goal` and
//! `lock_tetris_scores_back_to_back_bonus_on_second_qualifying_clear` exercise
//! via private fields). The asserted values are the spec values; if the seam or
//! engine disagrees, the resulting red test is a real bug to surface, not
//! something to soften.

use tetr_online::engine::{
    ActivePiece, CellKind, Engine, EngineConfig, EngineEvent, EngineScoreAction, PieceRotation,
    PieceType, RotationDirection, TSpinKind,
};

/// Fixed seed shared by every scenario for determinism.
const SEED: u64 = 0;

/// A narrow Level-1 well: 4-wide, Fixed-goal, Extended lock-down (defaults).
fn narrow_engine() -> Engine {
    Engine::new(
        EngineConfig {
            board_width: 4,
            ..EngineConfig::default()
        },
        SEED,
    )
}

/// A full-width Level-1 well, needed where T-slot geometry must avoid clearing
/// rows it does not intend to (T-Spins with controlled line counts).
fn wide_engine() -> Engine {
    Engine::new(EngineConfig::default(), SEED)
}

/// Paints a single locked Block at `(x, y)`.
fn block(engine: &mut Engine, x: isize, y: isize) {
    assert!(
        engine.set_cell(x, y, CellKind::Some(PieceType::O)),
        "set_cell({x}, {y}) must land inside the board"
    );
}

/// Paints locked Blocks at every `(x, y)` in `cells`.
fn fill(engine: &mut Engine, cells: &[(isize, isize)]) {
    for &(x, y) in cells {
        block(engine, x, y);
    }
}

/// Clears a bounded rectangle (`0..width` x `0..rows`) back to empty so a chain
/// of locks can be re-seeded to a known state despite residue from prior locks.
fn clear_low_rows(engine: &mut Engine, width: isize, rows: isize) {
    for y in 0..rows {
        for x in 0..width {
            assert!(engine.set_cell(x, y, CellKind::None));
        }
    }
}

/// A horizontal I at R0 anchored so its four minos sit on row 0 of a 4-wide
/// well (origin (0, -2) -> cells (0,0),(1,0),(2,0),(3,0)). Mirrors the unit test
/// `lock_line_clear_scores_single_and_advances_fixed_goal`.
fn horizontal_i_on_floor() -> ActivePiece {
    ActivePiece::new(PieceType::I, (0, -2))
}

/// A vertical I (rotated to R90) whose four minos occupy a single column across
/// rows `oy..oy+4`. With origin `(ox, oy)` the column is `ox + 2`. Mirrors the
/// unit test `lock_tetris_scores_back_to_back_bonus_on_second_qualifying_clear`.
fn vertical_i_at(ox: isize, oy: isize) -> ActivePiece {
    let mut active = ActivePiece::new(PieceType::I, (ox, oy));
    active.rotate_to(
        PieceRotation::R90,
        (ox, oy),
        RotationDirection::Clockwise,
        1,
        false,
    );
    active
}

/// A T-piece placed at `origin` and "rotated" in place so its last successful
/// action is a rotation (the T-Spin recognition precondition, §12.2). `rotation`
/// is the final facing; kick 1 means no offset. Mirrors the unit-test idiom in
/// `lock_uses_t_spin_classifier_for_score_action`.
fn rotated_t_at(rotation: PieceRotation, origin: (isize, isize)) -> ActivePiece {
    let mut active = ActivePiece::new(PieceType::T, origin);
    active.rotate_to(
        rotation,
        origin,
        RotationDirection::Clockwise,
        1,
        false,
    );
    active
}

/// Extracts the single `ScoreAwarded` event from a lock's event vector.
fn score_awarded(events: &[EngineEvent]) -> (EngineScoreAction, usize, usize, bool) {
    events
        .iter()
        .find_map(|event| match event {
            EngineEvent::ScoreAwarded {
                action,
                score,
                total_score,
                back_to_back_bonus,
            } => Some((*action, *score, *total_score, *back_to_back_bonus)),
            _ => None,
        })
        .unwrap_or_else(|| panic!("expected a ScoreAwarded event in {events:?}"))
}

/// Asserts the lock produced exactly the expected `lines_cleared`, then returns
/// the `ScoreAwarded` tuple. Keeps the line count honest alongside the score.
fn assert_lock(
    events: &[EngineEvent],
    expected_lines: usize,
) -> (EngineScoreAction, usize, usize, bool) {
    let locked_lines = events
        .iter()
        .find_map(|event| match event {
            EngineEvent::Locked { lines_cleared, .. } => Some(*lines_cleared),
            _ => None,
        })
        .unwrap_or_else(|| panic!("expected a Locked event in {events:?}"));
    assert_eq!(
        locked_lines, expected_lines,
        "lock cleared {locked_lines} lines, expected {expected_lines}: {events:?}"
    );
    score_awarded(events)
}

// =============================================================================
// 1. Single / Double / Triple / Tetris all score the §13.1 base values at L1.
// =============================================================================

// §25.8 / §13.1 — On a 4-wide Level-1 well, a Single/Double/Triple/Tetris scores
// exactly 100/300/500/800 with no Back-to-Back bonus on the first clear. Mirrors
// `mod.rs::lock_line_clear_scores_single_and_advances_fixed_goal` for the Single
// and extends the same vertical-I-into-a-prefilled-well recipe to N rows.
#[test]
fn single_double_triple_tetris_at_level_1() {
    // Single: a horizontal I dropped onto an empty 4-wide floor fills row 0.
    {
        let mut engine = narrow_engine();
        let events = engine.lock_active_for_test(horizontal_i_on_floor());
        let (action, score, total, b2b) = assert_lock(&events, 1);
        assert_eq!(action, EngineScoreAction::Single);
        assert_eq!(score, 100, "Single scores 100 * 1 at level 1");
        assert_eq!(total, 100);
        assert!(!b2b, "the first qualifying clear gets no B2B bonus");
        let snapshot = engine.snapshot();
        assert_eq!(snapshot.score, 100);
        assert_eq!(snapshot.lines, 1);
        assert_eq!(snapshot.level, 1);
    }

    // Double / Triple / Tetris: prefill columns 0..3 across `rows` full rows,
    // then drop a vertical I into the open column 3 (origin (1, 0)). Exactly
    // `rows` rows complete.
    for (rows, expected_action, expected_score) in [
        (2usize, EngineScoreAction::Double, 300usize),
        (3, EngineScoreAction::Triple, 500),
        (4, EngineScoreAction::Tetris, 800),
    ] {
        let mut engine = narrow_engine();
        for y in 0..rows as isize {
            fill(&mut engine, &[(0, y), (1, y), (2, y)]);
        }
        let events = engine.lock_active_for_test(vertical_i_at(1, 0));
        let (action, score, total, b2b) = assert_lock(&events, rows);
        assert_eq!(action, expected_action, "{rows}-row clear action");
        assert_eq!(
            score, expected_score,
            "{rows}-row clear scores {expected_score} at level 1"
        );
        assert_eq!(total, expected_score, "first clear: total == base");
        assert!(
            !b2b,
            "the first qualifying clear of a run gets no B2B bonus"
        );
        assert_eq!(engine.snapshot().score, expected_score);
    }
}

// =============================================================================
// 2. Mini T-Spin (0 lines) = 100, Mini T-Spin Single = 200.
// =============================================================================

// §25.8 / §12.5 / §13.1 — A Mini T-Spin requires both back corners (C, D) and at
// least one front corner (A or B) blocked, while NOT satisfying the full-T-Spin
// pattern. A zero-line Mini scores 100; a Mini that clears one line scores 200.
//
// Geometry (T at R0, origin (4, 4); t-center (5, 5)):
//   corners  A=NW(4,6)  B=NE(6,6)  C=SW(4,4)  D=SE(6,4)
//   T minos  (4,5) (5,5) (5,6) (6,5)
// Blocking C, D and only A (leaving B clear) yields Mini, not Full.
#[test]
fn mini_tspin_and_mini_tspin_single_100_200() {
    // Mini T-Spin, 0 lines -> 100. Full-width well so no row fills.
    {
        let mut engine = wide_engine();
        // C = SW(4,4), D = SE(6,4), A = NW(4,6); B = NE(6,6) deliberately empty.
        fill(&mut engine, &[(4, 4), (6, 4), (4, 6)]);
        let events = engine.lock_active_for_test(rotated_t_at(PieceRotation::R0, (4, 4)));
        let (action, score, total, b2b) = assert_lock(&events, 0);
        assert_eq!(
            action,
            EngineScoreAction::TSpin {
                kind: TSpinKind::Mini,
                lines: 0,
            }
        );
        assert_eq!(score, 100, "Mini T-Spin scores 100 * 1 at level 1");
        assert_eq!(total, 100);
        assert!(!b2b, "a zero-line Mini cannot receive a B2B bonus");
        // A zero-line Mini does not start a B2B chain (§13.2).
        assert!(!engine.snapshot().back_to_back_active);
    }

    // Mini T-Spin Single -> 200. Prefill row 5 (the T's bottom row) outside the
    // T's three minos so it completes; keep the Mini corner pattern at y=4/y=6.
    {
        let mut engine = wide_engine();
        // Complete row 5 around T minos at columns 4,5,6.
        fill(
            &mut engine,
            &[(0, 5), (1, 5), (2, 5), (3, 5), (7, 5), (8, 5), (9, 5)],
        );
        // Mini corners: C=SW(4,4), D=SE(6,4), A=NW(4,6); B=NE(6,6) stays empty.
        fill(&mut engine, &[(4, 4), (6, 4), (4, 6)]);
        let events = engine.lock_active_for_test(rotated_t_at(PieceRotation::R0, (4, 4)));
        let (action, score, total, b2b) = assert_lock(&events, 1);
        assert_eq!(
            action,
            EngineScoreAction::TSpin {
                kind: TSpinKind::Mini,
                lines: 1,
            }
        );
        assert_eq!(score, 200, "Mini T-Spin Single scores 200 * 1 at level 1");
        assert_eq!(total, 200);
        assert!(!b2b, "first qualifying clear: no bonus yet");
        // Mini T-Spin Single is a qualifying action -> it starts the chain.
        assert!(engine.snapshot().back_to_back_active);
    }
}

// =============================================================================
// 3. Full T-Spin (0 lines) = 400, T-Spin Single = 800.
// =============================================================================

// §25.8 / §12.4 / §13.1 — A full T-Spin needs both front corners (A, B) and at
// least one back corner (C or D) blocked. Zero lines -> 400; one line -> 800.
// Mirrors `mod.rs::lock_uses_t_spin_classifier_for_score_action` (the 0-line
// case) and extends it to a one-line clear.
#[test]
fn t_spin_400_and_single_800() {
    // Full T-Spin, 0 lines -> 400. Blocks A=NW(4,6), B=NE(6,6), C=SW(4,4).
    {
        let mut engine = wide_engine();
        fill(&mut engine, &[(4, 6), (6, 6), (4, 4)]);
        let events = engine.lock_active_for_test(rotated_t_at(PieceRotation::R0, (4, 4)));
        let (action, score, total, b2b) = assert_lock(&events, 0);
        assert_eq!(
            action,
            EngineScoreAction::TSpin {
                kind: TSpinKind::Full,
                lines: 0,
            }
        );
        assert_eq!(score, 400, "T-Spin (no lines) scores 400 * 1 at level 1");
        assert_eq!(total, 400);
        assert!(!b2b);
        // Zero-line T-Spin does not start a B2B chain (§13.2).
        assert!(!engine.snapshot().back_to_back_active);
    }

    // T-Spin Single -> 800. Complete the T's bottom row (5) while keeping the
    // full-T-Spin corner pattern A=NW(4,6), B=NE(6,6), C=SW(4,4).
    {
        let mut engine = wide_engine();
        fill(
            &mut engine,
            &[(0, 5), (1, 5), (2, 5), (3, 5), (7, 5), (8, 5), (9, 5)],
        );
        fill(&mut engine, &[(4, 6), (6, 6), (4, 4)]);
        let events = engine.lock_active_for_test(rotated_t_at(PieceRotation::R0, (4, 4)));
        let (action, score, total, b2b) = assert_lock(&events, 1);
        assert_eq!(
            action,
            EngineScoreAction::TSpin {
                kind: TSpinKind::Full,
                lines: 1,
            }
        );
        assert_eq!(score, 800, "T-Spin Single scores 800 * 1 at level 1");
        assert_eq!(total, 800);
        assert!(!b2b, "first qualifying clear: no bonus yet");
        assert!(engine.snapshot().back_to_back_active);
    }
}

// =============================================================================
// 4. T-Spin Double = 1200, T-Spin Triple = 1600.
// =============================================================================

// §25.8 / §12.6 / §13.1 — A full T-Spin clearing two lines scores 1200; clearing
// three lines scores 1600. Both are the first qualifying clear of a fresh engine,
// so neither carries a B2B bonus.
#[test]
fn t_spin_double_1200_and_triple_1600() {
    // T-Spin Double -> 1200. T at R0, origin (4, 4): bottom row 5 (cols 4,5,6),
    // top row 6 (col 5). Complete both rows around the T; add C=SW(4,4) so the
    // front corners A=NW(4,6) + B=NE(6,6) + a back corner satisfy "full".
    {
        let mut engine = wide_engine();
        // Row 5 around T cols {4,5,6}.
        fill(
            &mut engine,
            &[(0, 5), (1, 5), (2, 5), (3, 5), (7, 5), (8, 5), (9, 5)],
        );
        // Row 6 around T col {5} (this also blocks A=NW(4,6) and B=NE(6,6)).
        fill(
            &mut engine,
            &[(0, 6), (1, 6), (2, 6), (3, 6), (4, 6), (6, 6), (7, 6), (8, 6), (9, 6)],
        );
        // Back corner C=SW(4,4) so full-by-corners holds (A & B & (C|D)).
        block(&mut engine, 4, 4);
        let events = engine.lock_active_for_test(rotated_t_at(PieceRotation::R0, (4, 4)));
        let (action, score, total, b2b) = assert_lock(&events, 2);
        assert_eq!(
            action,
            EngineScoreAction::TSpin {
                kind: TSpinKind::Full,
                lines: 2,
            }
        );
        assert_eq!(score, 1200, "T-Spin Double scores 1200 * 1 at level 1");
        assert_eq!(total, 1200);
        assert!(!b2b);
    }

    // T-Spin Triple -> 1600. A vertical T (R90) drops into a 3-deep tunnel. With
    // origin (4, 4) the spine occupies col 5 rows 4,5,6 and the nub is (6, 5).
    // Completing rows 4,5,6 around those minos blocks all four diagonal corners
    // (NE/SE front, NW/SW back) -> full T-Spin clearing three lines.
    {
        let mut engine = wide_engine();
        // Row 4 around T col {5}.
        fill(
            &mut engine,
            &[(0, 4), (1, 4), (2, 4), (3, 4), (4, 4), (6, 4), (7, 4), (8, 4), (9, 4)],
        );
        // Row 5 around T cols {5, 6} (spine + nub).
        fill(
            &mut engine,
            &[(0, 5), (1, 5), (2, 5), (3, 5), (4, 5), (7, 5), (8, 5), (9, 5)],
        );
        // Row 6 around T col {5}.
        fill(
            &mut engine,
            &[(0, 6), (1, 6), (2, 6), (3, 6), (4, 6), (6, 6), (7, 6), (8, 6), (9, 6)],
        );
        let events = engine.lock_active_for_test(rotated_t_at(PieceRotation::R90, (4, 4)));
        let (action, score, total, b2b) = assert_lock(&events, 3);
        assert_eq!(
            action,
            EngineScoreAction::TSpin {
                kind: TSpinKind::Full,
                lines: 3,
            }
        );
        assert_eq!(score, 1600, "T-Spin Triple scores 1600 * 1 at level 1");
        assert_eq!(total, 1600);
        assert!(!b2b);
    }
}

// =============================================================================
// 5. Canonical Back-to-Back example totals 5400 (§13.3).
// =============================================================================

// §25.8 / §13.3 — Replays the guideline's canonical chain at Level 1 and asserts
// the final snapshot score is exactly 5400:
//
//   Tetris        = 800           starts B2B, no bonus      (running 800)
//   T-Spin Double = 1200 + 600    B2B bonus                 (running 2600)
//   T-Spin (0)    = 400           no bonus, chain preserved (running 3000)
//   Tetris        = 800  + 400    B2B bonus                 (running 4200)
//   T-Spin Single = 800  + 400    B2B bonus                 (running 5400)
//
// Each clear is set up on a full-width well; the low rows are cleared back to
// empty between locks so residue from a previous lock cannot corrupt the next.
//
// The §13.3 example is computed explicitly "At Level 1" for every action. This
// chain clears 4+2+0+4+1 = 11 physical lines, which is more than the Level-1
// Fixed goal of 10 (§14.2), so a real continuous game would level up to 2 before
// the final T-Spin Single and score it at the 2x multiplier. To honor the spec's
// stated precondition (and isolate the B2B *bonus* arithmetic from level scaling,
// which §25.9 covers separately), the level/goal progression is rewound to the
// start after each lock via `reset_level_for_test`. This preserves the running
// score and the Back-to-Back chain — only the level multiplier is pinned at 1.
#[test]
fn b2b_example_totals_5400() {
    let mut engine = wide_engine();
    let width = engine.snapshot().config.board_width as isize; // 10

    // Reusable setups -----------------------------------------------------

    // A full-width Tetris well: rows 0..4 complete except column 9, where a
    // vertical I (origin (7, 0) -> column 9, rows 0..4) drops in.
    fn fill_tetris_well(engine: &mut Engine, width: isize) {
        for y in 0..4 {
            for x in 0..(width - 1) {
                block(engine, x, y);
            }
        }
    }
    fn tetris_i(width: isize) -> ActivePiece {
        // column = ox + 2 = width - 1 -> ox = width - 3.
        vertical_i_at(width - 3, 0)
    }

    // 1) Tetris -> 800, starts B2B.
    fill_tetris_well(&mut engine, width);
    let events = engine.lock_active_for_test(tetris_i(width));
    let (action, score, total, b2b) = assert_lock(&events, 4);
    assert_eq!(action, EngineScoreAction::Tetris);
    assert_eq!((score, total, b2b), (800, 800, false));
    assert!(engine.snapshot().back_to_back_active);
    // Pin Level 1 for the next action (preserves score + B2B chain); see header.
    engine.reset_level_for_test();

    // 2) T-Spin Double -> 1200 + 600 bonus.
    clear_low_rows(&mut engine, width, 8);
    fill(
        &mut engine,
        &[(0, 5), (1, 5), (2, 5), (3, 5), (7, 5), (8, 5), (9, 5)],
    );
    fill(
        &mut engine,
        &[(0, 6), (1, 6), (2, 6), (3, 6), (4, 6), (6, 6), (7, 6), (8, 6), (9, 6)],
    );
    block(&mut engine, 4, 4);
    let events = engine.lock_active_for_test(rotated_t_at(PieceRotation::R0, (4, 4)));
    let (action, score, total, b2b) = assert_lock(&events, 2);
    assert_eq!(
        action,
        EngineScoreAction::TSpin {
            kind: TSpinKind::Full,
            lines: 2,
        }
    );
    assert_eq!((score, total, b2b), (1800, 2600, true));
    engine.reset_level_for_test();

    // 3) T-Spin (0 lines) -> 400, no bonus, chain preserved.
    clear_low_rows(&mut engine, width, 8);
    fill(&mut engine, &[(4, 6), (6, 6), (4, 4)]);
    let events = engine.lock_active_for_test(rotated_t_at(PieceRotation::R0, (4, 4)));
    let (action, score, total, b2b) = assert_lock(&events, 0);
    assert_eq!(
        action,
        EngineScoreAction::TSpin {
            kind: TSpinKind::Full,
            lines: 0,
        }
    );
    assert_eq!((score, total, b2b), (400, 3000, false));
    assert!(
        engine.snapshot().back_to_back_active,
        "a zero-line T-Spin preserves the existing B2B chain"
    );
    engine.reset_level_for_test();

    // 4) Tetris -> 800 + 400 bonus.
    clear_low_rows(&mut engine, width, 8);
    fill_tetris_well(&mut engine, width);
    let events = engine.lock_active_for_test(tetris_i(width));
    let (action, score, total, b2b) = assert_lock(&events, 4);
    assert_eq!(action, EngineScoreAction::Tetris);
    assert_eq!((score, total, b2b), (1200, 4200, true));
    engine.reset_level_for_test();

    // 5) T-Spin Single -> 800 + 400 bonus.
    clear_low_rows(&mut engine, width, 8);
    fill(
        &mut engine,
        &[(0, 5), (1, 5), (2, 5), (3, 5), (7, 5), (8, 5), (9, 5)],
    );
    fill(&mut engine, &[(4, 6), (6, 6), (4, 4)]);
    let events = engine.lock_active_for_test(rotated_t_at(PieceRotation::R0, (4, 4)));
    let (action, score, total, b2b) = assert_lock(&events, 1);
    assert_eq!(
        action,
        EngineScoreAction::TSpin {
            kind: TSpinKind::Full,
            lines: 1,
        }
    );
    assert_eq!((score, total, b2b), (1200, 5400, true));

    assert_eq!(
        engine.snapshot().score,
        5400,
        "the canonical guideline B2B chain totals 5400 at level 1"
    );
}

// =============================================================================
// 6. Back-to-Back bonus is 1.5x on the second qualifying clear.
// =============================================================================

// §25.8 / §13.2 — Mirrors `mod.rs::lock_tetris_scores_back_to_back_bonus_on_
// second_qualifying_clear`: the first Tetris scores 800 with no bonus and starts
// the chain; a second Tetris while the chain is active scores 1200 (800 + 400
// bonus) with `back_to_back_bonus: true`, for a running total of 2000.
#[test]
fn back_to_back_bonus_is_1_5x_on_second_qualifying_clear() {
    let mut engine = narrow_engine();

    // First Tetris: cols 0..3 across rows 0..4, vertical I into column 3.
    for y in 0..4 {
        fill(&mut engine, &[(0, y), (1, y), (2, y)]);
    }
    let events = engine.lock_active_for_test(vertical_i_at(1, 0));
    let (action, score, total, b2b) = assert_lock(&events, 4);
    assert_eq!(action, EngineScoreAction::Tetris);
    assert_eq!(score, 800, "first Tetris scores 800 with no bonus");
    assert_eq!(total, 800);
    assert!(!b2b, "first Tetris of a run carries no B2B bonus");
    assert!(engine.snapshot().back_to_back_active);

    // Second Tetris while B2B is active: 800 base + 400 bonus = 1200.
    for y in 0..4 {
        fill(&mut engine, &[(0, y), (1, y), (2, y)]);
    }
    let events = engine.lock_active_for_test(vertical_i_at(1, 0));
    let (action, score, total, b2b) = assert_lock(&events, 4);
    assert_eq!(action, EngineScoreAction::Tetris);
    assert_eq!(
        score, 1200,
        "second qualifying Tetris scores 1.5x base = 1200"
    );
    assert_eq!(total, 2000);
    assert!(b2b, "the second qualifying clear must flag the B2B bonus");
    assert_eq!(engine.snapshot().score, 2000);
}

// =============================================================================
// 7. A non-qualifying line clear breaks the Back-to-Back chain.
// =============================================================================

// §25.8 / §13.2 — A Single (non-qualifying) clears the chain that a Tetris
// started, so a subsequent Tetris scores 800 again rather than 1200. The Single
// itself receives no bonus and leaves `back_to_back_active` false.
#[test]
fn non_qualifying_line_clear_breaks_b2b() {
    let mut engine = narrow_engine();

    // Tetris starts the chain (800).
    for y in 0..4 {
        fill(&mut engine, &[(0, y), (1, y), (2, y)]);
    }
    let events = engine.lock_active_for_test(vertical_i_at(1, 0));
    let (action, _score, total, _b2b) = assert_lock(&events, 4);
    assert_eq!(action, EngineScoreAction::Tetris);
    assert_eq!(total, 800);
    assert!(engine.snapshot().back_to_back_active);

    // A Single (cols 0..3 on row 0, vertical I fills column 3) clears one line
    // and breaks the chain.
    clear_low_rows(&mut engine, 4, 4);
    fill(&mut engine, &[(0, 0), (1, 0), (2, 0)]);
    let events = engine.lock_active_for_test(vertical_i_at(1, 0));
    let (action, score, total, b2b) = assert_lock(&events, 1);
    assert_eq!(action, EngineScoreAction::Single);
    assert_eq!(score, 100, "Single scores its flat 100 with no bonus");
    assert_eq!(total, 900);
    assert!(!b2b);
    assert!(
        !engine.snapshot().back_to_back_active,
        "a Single must break the B2B chain"
    );

    // The next Tetris therefore scores 800 again, not 1200.
    clear_low_rows(&mut engine, 4, 4);
    for y in 0..4 {
        fill(&mut engine, &[(0, y), (1, y), (2, y)]);
    }
    let events = engine.lock_active_for_test(vertical_i_at(1, 0));
    let (action, score, total, b2b) = assert_lock(&events, 4);
    assert_eq!(action, EngineScoreAction::Tetris);
    assert_eq!(
        score, 800,
        "after the chain breaks, the next Tetris scores 800 (no bonus)"
    );
    assert_eq!(total, 1700);
    assert!(
        !b2b,
        "the Tetris after a broken chain restarts B2B without a bonus"
    );
    assert_eq!(engine.snapshot().score, 1700);
}
