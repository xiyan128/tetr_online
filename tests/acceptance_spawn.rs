//! Acceptance tests for the guideline §25.1 "Board and spawn".
//!
//! These exercise the real `Engine` end-to-end through its public
//! `InputFrame`/`step`/`snapshot` API plus the public spawn geometry helpers
//! (`Piece::spawn_coords`). Board-precondition scenarios (block-out overlap,
//! blocked-row-below) use the test seam on `Engine` (`set_cell`/`set_active`),
//! which is the only public way to stage a board for an integration test.
//!
//! Spec mapping (guideline rows are 1-based; the engine is 0-based with y
//! growing UP and the skyline at y >= visible_height):
//!
//! * guideline column `c` (1-based) == engine x `c - 1`
//! * guideline row    `r` (1-based) == engine y `r - 1`
//!
//! With the default config (visible_height = 20) the first buffer row 21
//! is engine y = 20 (the skyline row).
//!
//! NOTE on the immediate drop: §25.1 says a freshly generated piece drops
//! exactly one row if the row below is free. The engine performs this drop
//! inside the same `step()` that spawns the piece, so on an empty board the
//! observable `snapshot().active` sits one row BELOW the spawn row. Tests that
//! talk about the *spawn* row therefore reconstruct it as `observed_y + 1`
//! (exactly one drop happened) rather than softening the spec value.

use tetr_online::engine::{
    ActivePiece, CellKind, Engine, EngineConfig, EngineEvent, GameOverStatus, InputFrame, Piece,
    PieceType,
};

/// Build a default-config engine whose FIRST spawned piece is `wanted`.
///
/// The seven-bag generator is seeded deterministically (`StdRng::seed_from_u64`),
/// so scanning a fixed seed range and picking the first match yields the same
/// engine on every run. `next_queue[0]` (pre-step) is exactly the first piece
/// the engine will spawn.
fn engine_with_first_piece(wanted: PieceType) -> Engine {
    engine_with_first_piece_config(wanted, EngineConfig::default())
}

fn engine_with_first_piece_config(wanted: PieceType, config: EngineConfig) -> Engine {
    for seed in 0..10_000u64 {
        let engine = Engine::new(config.clone(), seed);
        if engine.snapshot().next_queue[0] == wanted {
            return engine;
        }
    }
    panic!("no seed in 0..10000 produced {wanted:?} as the first piece");
}

/// Global cells a piece occupies at its R0 spawn footprint, using the engine's
/// own public spawn geometry (`Piece::cells` at R0 translated by `origin`).
fn spawn_footprint(piece_type: PieceType, origin: (isize, isize)) -> Vec<(isize, isize)> {
    let piece = Piece::from(piece_type);
    let mut cells: Vec<(isize, isize)> = piece
        .cells()
        .into_iter()
        .map(|(x, y)| (x + origin.0, y + origin.1))
        .collect();
    cells.sort();
    cells
}

fn sorted_active_cells(engine: &Engine) -> Vec<(isize, isize)> {
    let active = engine.snapshot().active.expect("active piece after step");
    let mut cells: Vec<(isize, isize)> = active.cells.iter().map(|c| (c.x, c.y)).collect();
    cells.sort();
    cells
}

/// Step once with no inputs and assert it cleanly spawned `expected`: spawning
/// is snapshot state (not an event), so the observation is `active` going from
/// None to `Some(expected)` across a step that emits nothing.
fn step_spawns(engine: &mut Engine, expected: PieceType) {
    assert!(
        engine.snapshot().active.is_none(),
        "no piece is active before the spawning step"
    );
    let events = engine.step(InputFrame::default());
    assert!(
        events.is_empty(),
        "a clean spawn step emits no events, got {events:?}"
    );
    assert_eq!(
        engine
            .snapshot()
            .active
            .expect("active piece after the spawning step")
            .piece_type,
        expected,
        "the spawned piece must be the front of the next queue"
    );
}

/// 1. §25.1 "I spawns at (4..7,21)".
///
/// Guideline columns 4..7 (1-based) == engine x in 3..=6; guideline row 21 ==
/// engine y == visible_height (20) == the skyline row. The engine immediately
/// drops the piece one row on the empty board, so the spawn row is recovered as
/// `observed_y + 1`.
#[test]
fn i_spawns_across_columns_4_to_7_on_skyline_row() {
    let config = EngineConfig::default();
    let visible_height = config.visible_height as isize; // 20

    // Spawn origin per the engine's own public spawn geometry.
    let spawn_origin =
        Piece::from(PieceType::I).spawn_coords(config.board_width, config.visible_height);
    assert_eq!(
        spawn_origin,
        (3, 18),
        "I spawn origin for default 10-wide board"
    );

    // The pre-drop spawn footprint is the guideline position: columns 3..=6 at
    // the skyline row y == 20.
    let spawn_cells = spawn_footprint(PieceType::I, spawn_origin);
    assert_eq!(
        spawn_cells,
        vec![(3, 20), (4, 20), (5, 20), (6, 20)],
        "I spawns across columns 4..7 (0-based 3..=6) on the skyline row"
    );
    for &(x, y) in &spawn_cells {
        assert!((3..=6).contains(&x), "spawn column {x} within 3..=6");
        assert_eq!(y, visible_height, "spawn row is the skyline row y == 20");
    }

    // Drive the real engine: a zero-input step spawns I and applies the single
    // immediate gravity drop (row below is free). The observed active piece is
    // therefore exactly one row below the spawn row.
    let mut engine = engine_with_first_piece(PieceType::I);
    step_spawns(&mut engine, PieceType::I);

    let observed = sorted_active_cells(&engine);
    assert_eq!(
        observed,
        vec![(3, 19), (4, 19), (5, 19), (6, 19)],
        "after the immediate one-row drop the I occupies columns 3..=6 at y == 19"
    );
    for &(x, y) in &observed {
        assert!((3..=6).contains(&x), "active column {x} stays within 3..=6");
        // Reconstructed spawn row == skyline row (exactly one drop occurred).
        assert_eq!(y + 1, visible_height, "spawn row (observed + 1) == y == 20");
    }
}

/// 2. §25.1 "O spawns at (5..6,21..22)".
///
/// Guideline columns 5..6 (1-based) == engine x in {4,5}; guideline rows 21..22
/// == engine y in {20,21}. Assert the 2x2 footprint at the expected origin.
#[test]
fn o_spawns_in_columns_5_6_rows_21_22() {
    let config = EngineConfig::default();

    let spawn_origin =
        Piece::from(PieceType::O).spawn_coords(config.board_width, config.visible_height);
    assert_eq!(spawn_origin, (3, 19), "O spawn origin");

    // 2x2 footprint occupies columns {4,5} and rows {20,21} at spawn.
    let spawn_cells = spawn_footprint(PieceType::O, spawn_origin);
    assert_eq!(
        spawn_cells,
        vec![(4, 20), (4, 21), (5, 20), (5, 21)],
        "O spawns as a 2x2 in columns 5,6 / rows 21,22 (0-based {{4,5}} x {{20,21}})"
    );

    // Real engine: spawn + one immediate drop -> 2x2 shifted down one row.
    let mut engine = engine_with_first_piece(PieceType::O);
    step_spawns(&mut engine, PieceType::O);

    let observed = sorted_active_cells(&engine);
    assert_eq!(
        observed,
        vec![(4, 19), (4, 20), (5, 19), (5, 20)],
        "after the immediate drop the O 2x2 occupies columns {{4,5}} x rows {{19,20}}"
    );
    // Spawn rows recovered as observed + 1 == {20,21}.
    let spawn_rows: std::collections::BTreeSet<isize> =
        observed.iter().map(|&(_, y)| y + 1).collect();
    assert_eq!(
        spawn_rows,
        [20isize, 21].into_iter().collect(),
        "reconstructed spawn rows are the two skyline rows 20,21"
    );
}

/// 3. §25.1 "J/L/S/T/Z spawn within columns 4..6 and rows 21..22".
///
/// Parametrized over the five 3-cell pieces. After the immediate one-row drop
/// the active piece occupies engine columns x in 3..=5 and rows y in {19,20}
/// (spawn rows {20,21} shifted down one).
#[test]
fn jlszt_spawn_within_columns_4_to_6_rows_21_22() {
    let config = EngineConfig::default();

    for piece_type in [
        PieceType::J,
        PieceType::L,
        PieceType::S,
        PieceType::T,
        PieceType::Z,
    ] {
        let spawn_origin =
            Piece::from(piece_type).spawn_coords(config.board_width, config.visible_height);
        assert_eq!(spawn_origin, (3, 19), "{piece_type:?} spawn origin");

        // Spawn footprint stays within columns {3,4,5} and rows {20,21}.
        for &(x, y) in &spawn_footprint(piece_type, spawn_origin) {
            assert!(
                (3..=5).contains(&x),
                "{piece_type:?} spawn column {x} within 3..=5"
            );
            assert!(
                (20..=21).contains(&y),
                "{piece_type:?} spawn row {y} within 20..=21"
            );
        }

        // Real engine: after spawn + one immediate drop, columns 3..=5, rows 19,20.
        let mut engine = engine_with_first_piece(piece_type);
        step_spawns(&mut engine, piece_type);

        let observed = sorted_active_cells(&engine);
        assert_eq!(observed.len(), 4, "{piece_type:?} occupies four cells");
        for &(x, y) in &observed {
            assert!(
                (3..=5).contains(&x),
                "{piece_type:?} active column {x} within 3..=5"
            );
            assert!(
                (19..=20).contains(&y),
                "{piece_type:?} active row {y} within 19..=20 after immediate drop"
            );
        }
    }
}

/// 4. §25.1 / §25.10 "Spawn overlap causes Block Out before immediate drop".
///
/// Pre-set a locked cell inside the next piece's spawn footprint via the test
/// seam, then a default step. The engine must emit ONLY `GameOver{BlockOut}`
/// (no spawn, no immediate drop), leave `active` empty and record
/// `game_over == Some(BlockOut)`.
#[test]
fn spawn_overlap_causes_block_out_before_immediate_drop() {
    let config = EngineConfig::default();
    let mut engine = Engine::new(config.clone(), 0);

    // Whatever the first piece is, occupy one of its spawn cells.
    let first = engine.snapshot().next_queue[0];
    let spawn_origin = Piece::from(first).spawn_coords(config.board_width, config.visible_height);
    let blocker = spawn_footprint(first, spawn_origin)[0];
    engine.set_cell(blocker.0, blocker.1, CellKind::Some(PieceType::O));

    let events = engine.step(InputFrame::default());
    assert_eq!(
        events,
        vec![EngineEvent::GameOver {
            reason: GameOverStatus::BlockOut
        }],
        "spawn overlap emits only GameOver{{BlockOut}} before any drop"
    );

    let snapshot = engine.snapshot();
    assert_eq!(snapshot.game_over, Some(GameOverStatus::BlockOut));
    assert!(
        snapshot.active.is_none(),
        "no active piece after a block-out spawn"
    );
}

/// 5. §25.1 "If row below is free, generated piece drops exactly one row".
///
/// On the empty default board the immediate drop fires, so the observed active
/// origin is exactly the spawn origin moved down one row.
#[test]
fn free_row_below_drops_generated_piece_exactly_one_row() {
    let config = EngineConfig::default();
    let mut engine = Engine::new(config.clone(), 0);

    let first = engine.snapshot().next_queue[0];
    let spawn_origin = Piece::from(first).spawn_coords(config.board_width, config.visible_height);

    step_spawns(&mut engine, first);

    let active = engine.snapshot().active.expect("active piece after spawn");
    assert_eq!(active.piece_type, first);
    assert_eq!(
        active.origin,
        (spawn_origin.0, spawn_origin.1 - 1),
        "free row below drops the generated piece exactly one row"
    );
}

/// 6. §25.1 "If row below is blocked, generated piece does not immediately drop".
///
/// Stage a locked cell directly under the spawn footprint (but NOT inside it, so
/// the spawn itself succeeds), seed the engine so the first piece is a known I,
/// then place the active piece at its true spawn origin via the seam and advance
/// a zero-dt frame. The grounded piece must stay put: the origin is unchanged
/// across the step (no immediate drop).
#[test]
fn blocked_row_below_keeps_piece_at_spawn() {
    let config = EngineConfig::default();
    // Force I so the spawn footprint (y == 20, columns 3..=6) and the row below
    // it (y == 19) are known exactly.
    let mut engine = engine_with_first_piece(PieceType::I);
    let spawn_origin =
        Piece::from(PieceType::I).spawn_coords(config.board_width, config.visible_height);
    assert_eq!(spawn_origin, (3, 18));

    // Block the row directly under the spawn footprint: cells at y == 19 under
    // the I's columns 3..=6. (Spawn cells are at y == 20, so spawn still fits.)
    for x in 3..=6 {
        engine.set_cell(x, 19, CellKind::Some(PieceType::O));
    }

    // Place the active I exactly at its spawn origin, bypassing the spawn-time
    // drop, so we isolate "does it drop one row when the row below is blocked?".
    engine.set_active(ActivePiece::new(PieceType::I, spawn_origin));

    let events = engine.step(InputFrame::default());
    assert!(
        events.is_empty(),
        "the zero-dt step over a blocked row must not drop or lock the piece: {events:?}"
    );

    let active = engine.snapshot().active.expect("active piece");
    assert_eq!(
        active.origin, spawn_origin,
        "blocked row below keeps the generated piece at its spawn origin"
    );
}
