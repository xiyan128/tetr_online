//! Acceptance tests for reference_guideline.md §25.10 "Game over".
//!
//! Spec mapping (the guideline numbers rows 1-based; the engine is 0-based):
//!   * §16.1 Block Out: a spawn cell of the freshly generated Tetrimino overlaps
//!     an existing Block, detected *before* the immediate one-row drop.
//!   * §16.2 Lock Out: the active Tetrimino locks down completely above the
//!     Skyline. 1-based spec `cell.y >= 21` == 0-based `cell.y >= visible_height (20)`.
//!   * §16.3 Top Out: incoming lines push existing Blocks above the Top Out Line,
//!     i.e. above row 40. 1-based "above row 40" == 0-based `y >= total_height (40)`.
//!   * §16.4 Not game over: a piece partly below the Skyline, and Blocks that sit
//!     inside the Buffer Zone but below the Top Out Line, are allowed.
//!
//! These exercise the public free-function layer (`is_block_out`, `is_lock_out`,
//! `is_top_out`) plus, where reachable, the real `Engine` end-to-end. The
//! pieces and boards are built deterministically; no RNG is involved in the
//! helper assertions, and the Engine paths use a fixed seed.

use tetr_online::engine::{
    is_block_out, is_lock_out, is_top_out, ActivePiece, Board, CellKind, Engine, EngineConfig,
    EngineEvent, GameOverStatus, Piece, PieceType,
};

/// §25.10 / §16.1: a Tetrimino whose spawn footprint overlaps an existing Block
/// triggers Block Out during the Generation Phase.
///
/// Mirrors the spec's "check Block Out at the spawn cells before the immediate
/// one-row drop" (§3 generation, §16.5 check order): a T spawned on a default
/// 10x20+20 well at origin (3, 19) occupies the mino at board cell (4, 20); when
/// that cell is already filled, `is_block_out` reports the overlap.
#[test]
fn spawn_overlap_triggers_block_out() {
    let mut board = Board::with_top_margin(10, 20, 20);
    let piece = Piece::from(PieceType::T);
    let spawn_origin = piece.spawn_coords(10, 20);

    // Empty well: the freshly generated piece does not overlap anything.
    assert!(
        !is_block_out(&piece, &board, spawn_origin),
        "T should generate cleanly on an empty well"
    );

    // Fill a cell inside the T's spawn footprint, then re-check.
    assert!(board.set(4, 20, CellKind::Some(PieceType::O)));
    assert!(
        is_block_out(&piece, &board, spawn_origin),
        "spawn overlap must be a Block Out before the immediate drop"
    );
}

/// §25.10 / §16.2: a piece whose every mino ends up above the Skyline is a Lock Out.
///
/// `is_lock_out` reports true when *all* of the piece's minos satisfy
/// `y + origin.y >= visible_height`. With visible_height = 20, placing a T at
/// origin y = 20 lifts the whole piece into the Buffer Zone (0-based y >= 20,
/// i.e. 1-based row >= 21), matching the spec's `lockedCells.every(c => c.y >= 21)`.
///
/// The Engine end-to-end variant ("lock a piece entirely above row 20 =>
/// GameOver{LockOut} after Locked") requires pre-filling the board so the piece
/// can only land above the Skyline. The Engine's `board`/`active` fields are
/// private and there is no public test seam, so that path is split out below as
/// an ignored stub.
#[test]
fn whole_piece_locked_above_skyline_is_lock_out() {
    let visible_height = 20;
    let piece = Piece::from(PieceType::T);

    // Origin x is irrelevant to the vertical Lock Out test; use the spawn column.
    let x = piece.spawn_coords(10, visible_height).0;

    assert!(
        is_lock_out(&piece, (x, visible_height as isize), visible_height),
        "a piece locked entirely at/above the Skyline must be a Lock Out"
    );
}

/// §25.10 / §16.2 Engine end-to-end Lock Out.
///
/// Spec: locking a Tetrimino entirely above row 20 must emit `Locked` followed
/// by `GameOver{LockOut}` (see Engine::lock_active_piece, which classifies
/// `is_lock_out` against the pre-lock state and ends the game). The board test
/// seam places a T entirely in the Buffer Zone (origin y == visible_height, so
/// every mino sits at 0-based y >= 20, matching the free-function case above) and
/// locks it through the real lock/clear/score path. No rows below the Skyline are
/// touched, so the lock clears nothing and the only outcome is the Lock Out.
#[test]
fn whole_piece_locked_above_skyline_is_lock_out_via_engine() {
    let config = EngineConfig::default();
    let visible_height = config.visible_height as isize;
    let mut engine = Engine::new(config, 0);

    // A T whose footprint lies entirely at/above the Skyline (y >= 20).
    let x = Piece::from(PieceType::T).spawn_coords(10, 20).0;
    let above_skyline = ActivePiece::new(PieceType::T, (x, visible_height));

    let events = engine.lock_active_for_test(above_skyline);

    // The piece must lock (zero lines cleared) and then end the game as a Lock Out.
    let locked_at = events
        .iter()
        .position(|e| {
            matches!(
                e,
                EngineEvent::Locked {
                    lines_cleared: 0,
                    ..
                }
            )
        })
        .unwrap_or_else(|| panic!("expected a zero-line Locked event, got {events:?}"));
    let game_over_at = events
        .iter()
        .position(|e| {
            matches!(
                e,
                EngineEvent::GameOver {
                    reason: GameOverStatus::LockOut
                }
            )
        })
        .unwrap_or_else(|| panic!("expected GameOver{{LockOut}}, got {events:?}"));
    assert!(
        locked_at < game_over_at,
        "Lock Out must be reported after the piece locks: {events:?}"
    );
    assert_eq!(
        engine.snapshot().game_over,
        Some(GameOverStatus::LockOut),
        "engine must record the Lock Out in its snapshot"
    );
    assert!(
        engine.snapshot().active.is_none(),
        "no piece should spawn after a Lock Out ends the game"
    );
}

/// §25.10 / §16.4: a piece that locks partly below the Skyline is NOT a Lock Out.
///
/// Lowering the same T two rows below the Skyline (origin y = visible_height - 2)
/// leaves minos straddling the boundary, so `is_lock_out` is false — the spec's
/// "a piece locks partly below and partly above the Skyline" allowance.
#[test]
fn piece_partly_below_skyline_is_not_lock_out() {
    let visible_height = 20;
    let piece = Piece::from(PieceType::T);
    let x = piece.spawn_coords(10, visible_height).0;

    assert!(
        !is_lock_out(&piece, (x, visible_height as isize - 2), visible_height),
        "a piece partly below the Skyline must not be a Lock Out"
    );
}

/// §25.10 / §16.4: Blocks resting in the Buffer Zone but below the Top Out Line
/// (row 40) are NOT a Top Out.
///
/// Total board height is 40 (visible 20 + buffer 20). A Block at 0-based y = 39
/// (the highest buffer row, 1-based row 40) is still inside the board, so
/// `is_top_out` is false — the spec's "Existing Blocks are above the Skyline but
/// still within the Buffer Zone" allowance.
#[test]
fn blocks_in_buffer_below_row_41_are_not_top_out() {
    assert!(
        !is_top_out([(0, 39)], 40),
        "a Block at the top of the Buffer Zone (below the Top Out Line) is not a Top Out"
    );
}

/// §25.10 / §16.3: garbage that forces existing Blocks above row 40 is a Top Out.
///
/// A Block pushed to 0-based y = 40 (1-based "above row 40") crosses the absolute
/// ceiling, so `is_top_out` reports the Top Out.
#[test]
fn garbage_pushing_blocks_above_row_40_is_top_out() {
    assert!(
        is_top_out([(0, 40)], 40),
        "a Block forced above the Top Out Line must be a Top Out"
    );
}
