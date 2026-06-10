//! Acceptance tests for the guideline §25.7 (T-Spins).
//!
//! These exercise the T-Spin recognition rules and the zero-line T-Spin
//! scoring / back-to-back interaction through the engine's public boundary.
//!
//! Scenarios 1-5 drive the pure recognition helpers (`classify_t_spin`,
//! `t_spin_corners`) on hand-built `Board` + `ActivePiece` values — no `Engine`
//! required. Scenarios 6-7 need a pre-filled board, so they use the
//! `lock_active_for_test` test seam on `Engine` to run the real lock + scoring
//! path against a board we control.
//!
//! Spec anchors used for the assertions:
//!   * Full T-Spin (3 corners, front pattern a&b + a back corner) => Full.
//!   * Mini T-Spin (back corners c&d + a front corner) => Mini.
//!   * SRS Test 5 into the T-Slot overrides Mini to Full (point-5 exception).
//!   * Score: T-Spin (Full, 0 lines) = 400 * Level; at Level 1 => 400.
//!   * Zero-line T-Spins cannot START back-to-back, and do NOT BREAK an
//!     existing back-to-back chain.

use tetr_online::engine::{
    classify_t_spin, ActivePiece, Board, CellKind, Engine, EngineConfig, EngineEvent,
    EngineScoreAction, PieceAction, PieceRotation, PieceType, RotationDirection, TSpinKind,
};

/// Origin shared by the hand-built recognition scenarios. The T-Slot center is
/// `(origin.0 + 1, origin.1 + 1)`; with this origin the center is `(5, 5)`,
/// keeping all four corners comfortably inside a 10x20 board.
const ORIGIN: (isize, isize) = (4, 4);

/// Build a `T` `ActivePiece` whose last successful action is a rotation, so the
/// rotation precondition for a T-Spin is satisfied. `rotate_to` records
/// `PieceAction::Rotate` even when the target rotation equals the current one,
/// which is exactly how the engine flags a freshly-rotated piece.
fn rotated_t(
    rotation: PieceRotation,
    kick_number: u8,
    entered_t_slot_with_kick_5: bool,
) -> ActivePiece {
    let mut active_piece = ActivePiece::new(PieceType::T, ORIGIN);
    active_piece.rotate_to(
        rotation,
        ORIGIN,
        RotationDirection::Clockwise,
        kick_number,
        entered_t_slot_with_kick_5,
    );
    active_piece
}

/// A 10x20 board with the given corner offsets (relative to the T-Slot center)
/// filled with locked minos, marking those corners as "blocked".
fn board_with_blocked_corners(corner_offsets: &[(isize, isize)]) -> Board {
    let mut board = Board::new(10, 20);
    let center = (ORIGIN.0 + 1, ORIGIN.1 + 1);
    for (x_offset, y_offset) in corner_offsets {
        assert!(board.set(
            center.0 + x_offset,
            center.1 + y_offset,
            CellKind::Some(PieceType::O),
        ));
    }
    board
}

// -- Scenario 1 -------------------------------------------------------------

/// A non-T piece is never a T-Spin, regardless of how its corners are blocked.
#[test]
fn no_t_piece_means_no_t_spin() {
    // L piece, rotated, with three corners blocked — the classifier must still
    // bail out purely on piece type.
    let mut active_piece = ActivePiece::new(PieceType::L, ORIGIN);
    active_piece.rotate_to(
        PieceRotation::R90,
        ORIGIN,
        RotationDirection::Clockwise,
        1,
        false,
    );
    let board = board_with_blocked_corners(&[(-1, 1), (1, 1), (-1, -1)]);

    assert_eq!(classify_t_spin(&active_piece, &board), None);

    // And a J piece, same story.
    let mut j_piece = ActivePiece::new(PieceType::J, ORIGIN);
    j_piece.rotate_to(
        PieceRotation::R90,
        ORIGIN,
        RotationDirection::Clockwise,
        1,
        false,
    );
    assert_eq!(classify_t_spin(&j_piece, &board), None);
}

// -- Scenario 2 -------------------------------------------------------------

/// A T that has not rotated (last successful action is the spawn) is not a
/// T-Spin even with the corners blocked: the rotation precondition fails.
#[test]
fn t_piece_without_prior_rotation_is_not_t_spin() {
    let active_piece = ActivePiece::new(PieceType::T, ORIGIN);
    // Three corners blocked — geometry alone would look like a T-Spin.
    let board = board_with_blocked_corners(&[(-1, 1), (1, 1), (-1, -1)]);

    // No rotation has happened (last_successful_action == Spawn), so the
    // recognition rule rejects it.
    assert_eq!(active_piece.last_successful_action(), PieceAction::Spawn);
    assert_eq!(classify_t_spin(&active_piece, &board), None);
}

// -- Scenario 3 -------------------------------------------------------------

/// Rotated T with the two front corners (a & b) plus one back corner blocked
/// classifies as a Full T-Spin.
#[test]
fn three_corner_front_blocked_classifies_full() {
    let active_piece = rotated_t(PieceRotation::R0, 1, false);
    // At R0 the corner mapping is a=nw, b=ne, c=sw, d=se. Block nw + ne (the
    // front pair) and sw (a back corner): a & b & (c || d) => Full.
    let board = board_with_blocked_corners(&[(-1, 1), (1, 1), (-1, -1)]);

    assert_eq!(
        classify_t_spin(&active_piece, &board),
        Some(TSpinKind::Full)
    );
}

// -- Scenario 4 -------------------------------------------------------------

/// Rotated T with the two back corners (c & d) plus one front corner blocked
/// classifies as a Mini T-Spin.
#[test]
fn back_corner_pattern_classifies_mini() {
    let active_piece = rotated_t(PieceRotation::R0, 1, false);
    // At R0: c=sw, d=se, and a=nw is the front corner. Block sw + se (back
    // pair) and nw (a front corner): c & d & (a || b), and NOT the full
    // pattern => Mini.
    let board = board_with_blocked_corners(&[(-1, -1), (1, -1), (-1, 1)]);

    assert_eq!(
        classify_t_spin(&active_piece, &board),
        Some(TSpinKind::Mini)
    );
}

// -- Scenario 5 -------------------------------------------------------------

/// The point-5 exception: when SRS Test 5 (kick number 5) placed the T into the
/// slot, the result is upgraded to a Full T-Spin even though the blocked-corner
/// geometry would otherwise read as Mini.
#[test]
fn srs_test_5_into_t_slot_overrides_mini_to_full() {
    // Same Mini corner pattern as scenario 4 (back corners + one front), but
    // the piece arrived via kick 5 into the T-Slot.
    let active_piece = rotated_t(PieceRotation::R0, 5, true);
    let board = board_with_blocked_corners(&[(-1, -1), (1, -1), (-1, 1)]);

    // Sanity: the kick-5 flag is recorded on the piece.
    assert!(active_piece.used_kick_5_into_t_slot());
    assert_eq!(active_piece.last_rotation_kick_number(), Some(5));

    // Override: Mini geometry, but kick 5 forces Full.
    assert_eq!(
        classify_t_spin(&active_piece, &board),
        Some(TSpinKind::Full)
    );
}

// -- Scenario 6 -------------------------------------------------------------

/// A zero-line Full T-Spin scores `400 * Level` (= 400 at Level 1) but, per the
/// guideline, a zero-line T-Spin cannot START a back-to-back chain.
///
/// Needs the board-setup seam (a): we pre-fill three corner cells so the locked
/// T is recognized as a 3-corner Full T-Spin, with no row completed.
#[test]
fn zero_line_t_spin_scores_but_does_not_start_b2b() {
    let mut engine = Engine::new(EngineConfig::default(), 0);

    // T at origin (4,4), R0 => center (5,5). Block nw (4,6), ne (6,6) and the
    // back corner sw (4,4): the front pair plus a back corner => Full. These
    // three lone cells never complete a width-10 row, so 0 lines clear.
    for (x, y) in [(4, 6), (6, 6), (4, 4)] {
        engine.set_cell(x, y, CellKind::Some(PieceType::O));
    }

    let mut active = ActivePiece::new(PieceType::T, (4, 4));
    active.rotate_to(
        PieceRotation::R0,
        (4, 4),
        RotationDirection::Clockwise,
        1,
        false,
    );

    let events = engine.lock_active_for_test(active);

    // Lock with no clear, then a Full T-Spin (0 lines) scoring 400, then the
    // next piece spawns.
    assert!(
        matches!(
            events.as_slice(),
            [
                EngineEvent::Locked {
                    piece_type: PieceType::T,
                    lines_cleared: 0,
                },
                EngineEvent::ScoreAwarded {
                    action: EngineScoreAction::TSpin {
                        kind: TSpinKind::Full,
                        lines: 0,
                    },
                    score: 400,
                    total_score: 400,
                    back_to_back_bonus: false,
                },
                EngineEvent::Spawned { .. },
            ]
        ),
        "expected Locked(0) + Full T-Spin 0-line scoring 400 + Spawned, got {events:?}",
    );

    let snapshot = engine.snapshot();
    assert_eq!(snapshot.score, 400);
    assert_eq!(snapshot.lines, 0);
    // The defining assertion: a zero-line T-Spin does NOT start back-to-back.
    assert!(!snapshot.back_to_back_active);
}

// -- Scenario 7 -------------------------------------------------------------

/// A zero-line T-Spin preserves an already-active back-to-back chain: it neither
/// breaks it nor earns a back-to-back bonus for itself.
///
/// Needs the board-setup seam (a). We first establish back-to-back with a
/// Tetris, then perform a zero-line Full T-Spin on the (now-cleared) board and
/// assert the chain survives.
#[test]
fn zero_line_t_spin_preserves_existing_b2b() {
    // Narrow 4-wide well so a vertical I clears four rows at once.
    let config = EngineConfig {
        board_width: 4,
        ..EngineConfig::default()
    };
    let mut engine = Engine::new(config, 0);

    // Fill columns 0..3 on rows 0..4, leaving column 3 empty for four rows.
    for y in 0..4 {
        for x in 0..3 {
            engine.set_cell(x, y, CellKind::Some(PieceType::O));
        }
    }

    // Vertical I dropped into the empty column completes four rows = Tetris.
    let mut vertical_i = ActivePiece::new(PieceType::I, (1, 0));
    vertical_i.rotate_to(
        PieceRotation::R90,
        (1, 0),
        RotationDirection::Clockwise,
        1,
        false,
    );
    let tetris_events = engine.lock_active_for_test(vertical_i);
    assert!(
        matches!(
            tetris_events.as_slice(),
            [
                EngineEvent::Locked {
                    piece_type: PieceType::I,
                    lines_cleared: 4,
                },
                EngineEvent::ScoreAwarded {
                    action: EngineScoreAction::Tetris,
                    score: 800,
                    total_score: 800,
                    back_to_back_bonus: false,
                },
                EngineEvent::Spawned { .. },
            ]
        ),
        "expected Tetris establishing b2b, got {tetris_events:?}",
    );
    // Back-to-back is now armed by the Tetris.
    assert!(engine.snapshot().back_to_back_active);

    // The Tetris cleared rows 0..3, so the board is empty again. Build a
    // 3-corner Full T-Spin that clears zero lines in the 4-wide board.
    //
    // T at origin (1,4), R0 => center (2,5). Block nw (1,6), ne (3,6) and the
    // back corner sw (1,4): front pair + a back corner => Full. The T's own
    // cells are (1,5),(2,5),(2,6),(3,5); none of the three rows (4,5,6) reach
    // the 4-cell width, so 0 lines clear.
    for (x, y) in [(1, 6), (3, 6), (1, 4)] {
        engine.set_cell(x, y, CellKind::Some(PieceType::O));
    }
    let mut t_spin_piece = ActivePiece::new(PieceType::T, (1, 4));
    t_spin_piece.rotate_to(
        PieceRotation::R0,
        (1, 4),
        RotationDirection::Clockwise,
        1,
        false,
    );

    let t_spin_events = engine.lock_active_for_test(t_spin_piece);
    assert!(
        matches!(
            t_spin_events.as_slice(),
            [
                EngineEvent::Locked {
                    piece_type: PieceType::T,
                    lines_cleared: 0,
                },
                EngineEvent::ScoreAwarded {
                    action: EngineScoreAction::TSpin {
                        kind: TSpinKind::Full,
                        lines: 0,
                    },
                    score: 400,
                    total_score: 1200,
                    // Zero-line T-Spin earns no b2b bonus for itself...
                    back_to_back_bonus: false,
                },
                EngineEvent::Spawned { .. },
            ]
        ),
        "expected zero-line Full T-Spin preserving b2b, got {t_spin_events:?}",
    );

    let snapshot = engine.snapshot();
    // 800 (Tetris) + 400 (zero-line Full T-Spin, no bonus) = 1200.
    assert_eq!(snapshot.score, 1200);
    // ...and crucially the existing back-to-back chain is NOT broken.
    assert!(snapshot.back_to_back_active);
}

/// Scenario 8: a T-Spin Mini DOUBLE is a real scored clear — 400 × level — and
/// it starts a Back-to-Back chain, consistent with the attack table treating it
/// as a clear. (The Mini-Double row is unified across scoring, variable-goal
/// units, B2B qualification, and attack; it must never be "a clear in some
/// tables but not others".)
///
/// Geometry (10-wide board, right wall): a T at R270 (nub left) hugging the
/// wall, origin (8, 0) — cells (9,0),(9,1),(9,2),(8,1), center (9,1). Corners:
/// SE (10,0) and NE (10,2) are wall (blocked), SW (8,0) is pre-filled, NW (8,2)
/// is open — three corners, back pair + one front ⇒ Mini (the Full pattern
/// needs both front corners). Rows 0 and 1 complete on lock ⇒ a Mini Double.
#[test]
fn mini_t_spin_double_scores_400_and_starts_back_to_back() {
    let mut engine = Engine::new(EngineConfig::default(), 0);
    // Row 0 full except the T's (9,0); includes the SW corner (8,0).
    for x in 0..=8 {
        engine.set_cell(x, 0, CellKind::Some(PieceType::O));
    }
    // Row 1 full except the T's (8,1) and (9,1).
    for x in 0..=7 {
        engine.set_cell(x, 1, CellKind::Some(PieceType::O));
    }

    // The T arrives by rotation (the T-Spin precondition), resting at R270.
    let mut mini_double = ActivePiece::new(PieceType::T, (8, 0));
    mini_double.rotate_to(
        PieceRotation::R270,
        (8, 0),
        RotationDirection::Clockwise,
        1,
        false,
    );

    let events = engine.lock_active_for_test(mini_double);
    assert!(
        matches!(
            events.as_slice(),
            [
                EngineEvent::Locked {
                    piece_type: PieceType::T,
                    lines_cleared: 2,
                },
                EngineEvent::ScoreAwarded {
                    action: EngineScoreAction::TSpin {
                        kind: TSpinKind::Mini,
                        lines: 2,
                    },
                    score: 400, // 400 × level 1
                    total_score: 400,
                    back_to_back_bonus: false, // first qualifying clear: starts, not continues
                },
                EngineEvent::Spawned { .. },
            ]
        ),
        "expected a Mini Double worth 400 starting a B2B chain, got {events:?}",
    );

    let snapshot = engine.snapshot();
    assert_eq!(snapshot.score, 400);
    assert_eq!(snapshot.lines, 2);
    assert!(
        snapshot.back_to_back_active,
        "a Mini Double is a qualifying clear: the B2B chain must now be live",
    );
}
