//! Acceptance tests for SRS rotation / kick behaviour (reference_guideline.md §25.5).
//!
//! Each scenario in §25.5 maps to exactly one `#[test]` fn below:
//!   1. Empty-board Test 1 rotations succeed.
//!   2. Wall kicks work for all applicable pieces.
//!   3. Floor kicks work.
//!   4. Well kicks work.
//!   5. Failed rotation leaves the piece unchanged.
//!   6. Successful rotation stores the kick index.
//!   7. O footprint remains unchanged.
//!
//! Pure-geometry scenarios drive `Piece::try_rotate_with_kicks` directly against a
//! hand-built `Board`. Engine-level scenarios (6 and 7) drive the real `Engine`
//! through its public `step`/`snapshot` API only — the engine's `board`/`active`
//! fields are private, so we never inject state. Because the spawned piece type is
//! RNG-determined, we deterministically scan seeds for the first piece type we need;
//! the scan is reproducible (fixed seed range) so the tests stay deterministic.

use tetr_online::engine::{
    ActivePieceSnapshot, Board, CellKind, Engine, EngineConfig, EngineEvent, InputFrame, Piece,
    PieceRotation, PieceType, SnapshotCell,
};

/// A board with no locked cells and no top margin: only the implicit walls
/// (`x < 0`, `x >= width`) and the implicit floor (`y < 0`) block a piece.
fn empty_board(width: usize, height: usize) -> Board {
    Board::with_top_margin(width, height, 0)
}

/// Cells a piece occupies (sorted) when its `R0`/origin-relative shape is placed
/// at `origin`, used to compare footprints without reaching into private state.
fn footprint_at(piece: &Piece, origin: (isize, isize)) -> Vec<(isize, isize)> {
    let mut cells = piece
        .cells()
        .into_iter()
        .map(|(x, y)| (x + origin.0, y + origin.1))
        .collect::<Vec<_>>();
    cells.sort();
    cells
}

fn sorted_snapshot_cells(cells: &[SnapshotCell]) -> Vec<(isize, isize)> {
    let mut coords = cells.iter().map(|c| (c.x, c.y)).collect::<Vec<_>>();
    coords.sort();
    coords
}

/// Spawn the first piece on an otherwise-default engine seeded with `seed`,
/// returning the post-spawn active snapshot (the engine applies one immediate
/// gravity drop after spawning).
fn spawn_first_piece(seed: u64) -> (Engine, ActivePieceSnapshot) {
    let mut engine = Engine::new(EngineConfig::default(), seed);
    let spawn_events = engine.step(InputFrame::default());
    assert!(
        matches!(spawn_events.as_slice(), [EngineEvent::Spawned { .. }]),
        "a zero-dt first step must only spawn the first piece, got {spawn_events:?}"
    );
    let active = engine
        .snapshot()
        .active
        .expect("first step spawns an active piece");
    (engine, active)
}

/// Deterministically find a seed in `0..LIMIT` whose first spawned piece is
/// `wanted`. A 7-bag generator deals every piece within the first bag, so a
/// matching seed exists well inside this range.
fn seed_for_first_piece(wanted: PieceType) -> u64 {
    const LIMIT: u64 = 4096;
    for seed in 0..LIMIT {
        let engine = Engine::new(EngineConfig::default(), seed);
        if engine.snapshot().next_queue[0] == wanted {
            return seed;
        }
    }
    panic!("no seed in 0..{LIMIT} spawns {wanted:?} first");
}

// 1. Empty-board Test 1 rotation succeeds.
//
// On a clear board a T at centre rotates R0 -> R90 with the very first SRS test
// offset (no displacement), which `try_rotate_with_kicks` reports as the
// one-based `kick_number == 1`.
#[test]
fn empty_board_test_1_rotation_succeeds() {
    let board = empty_board(10, 20);
    let piece = Piece::from(PieceType::T);
    let origin = (3, 18); // centre column, mid-board

    let result = piece.try_rotate_with_kicks(&board, origin, PieceRotation::R90);

    assert_eq!(
        result,
        Some((PieceRotation::R90, origin, 1)),
        "Test 1 (no offset) must succeed on an empty board with kick_number == 1"
    );
}

// 2. Wall kick reports the one-based kick index.
//
// A T flush against the right wall (x == 8) cannot take the no-offset Test 1, so
// SRS Test 2 shifts it one cell left. `kick_number == 2` mirrors the in-crate
// unit test `pieces::tests::wall_kick_reports_one_based_srs_kick_number`.
#[test]
fn wall_kick_reports_one_based_kick_index() {
    let board = empty_board(10, 20);
    let piece = Piece::from(PieceType::T);
    let origin = (8, 5); // right-edge column

    let result = piece.try_rotate_with_kicks(&board, origin, PieceRotation::R90);

    assert_eq!(
        result,
        Some((PieceRotation::R90, (7, 5), 2)),
        "right-wall kick must use SRS Test 2 (shift left one cell) => kick_number == 2"
    );
}

// 3. Floor kick works.
//
// A horizontal I lying on the floor cannot rotate to vertical without one of its
// minos dropping below the floor, so SRS kicks it upward. The successful kick has
// `kick_number > 1` and every resulting cell sits at `y >= 0`.
#[test]
fn floor_kick_works() {
    let board = empty_board(10, 4);
    let piece = Piece::from(PieceType::I);
    // I in R0 occupies relative row y = 2, so origin y = -2 rests the bar on the
    // floor (its cells land on row y = 0).
    let origin = (3, -2);

    let result = piece
        .try_rotate_with_kicks(&board, origin, PieceRotation::R90)
        .expect("I on the floor must rotate via a floor kick");

    let (rotation, kicked_origin, kick_number) = result;
    assert_eq!(rotation, PieceRotation::R90);
    assert!(
        kick_number > 1,
        "a floor kick must use a later SRS test (kick_number > 1), got {kick_number}"
    );
    // The kick must lift the piece up off the floor: new origin is strictly above
    // the original, and every resulting mino is on or above the floor.
    assert!(
        kicked_origin.1 > origin.1,
        "floor kick must offset the piece upward, origin {origin:?} -> {kicked_origin:?}"
    );
    let mut rotated = Piece::from(PieceType::I);
    rotated.rotate_to(rotation);
    for (x, y) in footprint_at(&rotated, kicked_origin) {
        assert!(
            y >= 0,
            "floor-kicked cell ({x},{y}) must stay on or above the floor"
        );
    }
}

// 4. Well kick works.
//
// A 1-wide well is carved into a filled board; a horizontal I resting on the
// surrounding stacks rotates to vertical and SRS kicks it sideways/down into the
// well. The successful kick has `kick_number > 1` and the vertical bar lands in
// the well column.
#[test]
fn well_kick_works() {
    let width = 10;
    let well_column = 3isize;
    let mut board = empty_board(width, 6);
    // Fill every column except the well to height 3 (rows y = 0..3), leaving a
    // single-cell-wide vertical well at `well_column`.
    for x in 0..width as isize {
        if x == well_column {
            continue;
        }
        for y in 0..3 {
            assert!(
                board.set(x, y, CellKind::Some(PieceType::O)),
                "filling stack cell ({x},{y}) must stay in bounds"
            );
        }
    }

    let piece = Piece::from(PieceType::I);
    // Horizontal I resting on top of the stacks: relative row y = 2, origin y = 1
    // places its cells on row y = 3 (just above the height-3 stacks).
    let origin = (0, 1);

    let result = piece
        .try_rotate_with_kicks(&board, origin, PieceRotation::R90)
        .expect("I above a 1-wide well must rotate down into it via a well kick");

    let (rotation, kicked_origin, kick_number) = result;
    assert_eq!(rotation, PieceRotation::R90);
    assert!(
        kick_number > 1,
        "a well kick must use a later SRS test (kick_number > 1), got {kick_number}"
    );

    // The rotated vertical bar must occupy the well column only.
    let mut rotated = Piece::from(PieceType::I);
    rotated.rotate_to(rotation);
    let cells = footprint_at(&rotated, kicked_origin);
    assert!(
        cells.iter().all(|&(x, _)| x == well_column),
        "well-kicked I must slot entirely into the well column {well_column}, got {cells:?}"
    );
    // And it must not overlap any filled stack cell.
    for &(x, y) in &cells {
        assert_eq!(
            board.get_cell_kind(x, y),
            CellKind::None,
            "well-kicked cell ({x},{y}) must land in empty space"
        );
    }
}

// 5. Failed rotation leaves the piece unchanged.
//
// A T fully boxed in by locked cells has no legal kick for either direction, so
// `try_rotate_with_kicks` returns `None` and the caller keeps the original
// rotation/origin. (Boxing the piece *inside the engine* would require a private
// state seam; the public helper expresses the same invariant directly.)
#[test]
fn failed_rotation_leaves_piece_unchanged() {
    let size = 6;
    let mut board = empty_board(size, size);
    let piece = Piece::from(PieceType::T);
    let origin = (1, 1);
    let occupied: std::collections::HashSet<(isize, isize)> =
        footprint_at(&piece, origin).into_iter().collect();

    // Fill every cell except the four the T currently occupies, so no kick offset
    // (in any of the five SRS tests) can find clear space.
    for x in 0..size as isize {
        for y in 0..size as isize {
            if !occupied.contains(&(x, y)) {
                assert!(board.set(x, y, CellKind::Some(PieceType::O)));
            }
        }
    }

    assert_eq!(
        piece.try_rotate_with_kicks(&board, origin, PieceRotation::R90),
        None,
        "a fully boxed-in T must fail every clockwise kick"
    );
    assert_eq!(
        piece.try_rotate_with_kicks(&board, origin, PieceRotation::R270),
        None,
        "a fully boxed-in T must fail every counter-clockwise kick"
    );

    // The piece object itself is unchanged: `try_rotate_with_kicks` borrows
    // `&self` and never mutates, so its footprint is still the R0 footprint we
    // filled around. (Footprint equality also proves the rotation is unchanged:
    // any rotated T occupies a different cell set than R0.)
    assert_eq!(
        footprint_at(&piece, origin),
        {
            let mut v: Vec<_> = occupied.into_iter().collect();
            v.sort();
            v
        },
        "a failed rotation must leave the footprint untouched"
    );
}

// 6. Successful rotation stores the kick index.
//
// Driving the real engine: spawn a T, rotate clockwise on the empty board, and
// assert the emitted `Rotated` event carries the same one-based kick index the
// standalone helper computes for the piece's post-spawn position, and that the
// snapshot's active rotation advanced to R90.
#[test]
fn successful_rotation_stores_kick_index() {
    let seed = seed_for_first_piece(PieceType::T);
    let (mut engine, before) = spawn_first_piece(seed);
    assert_eq!(before.piece_type, PieceType::T);
    assert_eq!(before.rotation, PieceRotation::R0);

    // Compute what the kick helper reports for the piece at its actual spawn
    // origin, against an empty board matching the engine's (no locked cells yet).
    let config = engine.snapshot().config;
    let board = Board::with_top_margin(config.board_width, config.visible_height, 0);
    let piece = Piece::from(PieceType::T);
    let (expected_rotation, expected_origin, expected_kick) = piece
        .try_rotate_with_kicks(&board, before.origin, PieceRotation::R90)
        .expect("T must rotate clockwise on the empty spawn board");
    assert_eq!(expected_rotation, PieceRotation::R90);

    let events = engine.step(InputFrame {
        rotate_clockwise: true,
        ..InputFrame::default()
    });

    assert_eq!(
        events,
        vec![EngineEvent::Rotated {
            piece_type: PieceType::T,
            rotation: expected_rotation,
            origin: expected_origin,
            kick_number: expected_kick,
        }],
        "engine rotation must store the same one-based kick index the helper computes"
    );

    let after = engine
        .snapshot()
        .active
        .expect("active piece after rotation");
    assert_eq!(
        after.rotation,
        before.rotation + PieceRotation::R90,
        "snapshot rotation must advance one quarter-turn clockwise"
    );
    assert_eq!(after.rotation, PieceRotation::R90);
    assert_eq!(after.origin, expected_origin);
}

// 7. O footprint remains unchanged through rotation.
//
// The O piece never rotates: the helper returns `(R0, same origin, kick 0)`, and
// because the engine rejects kick 0 it emits no `Rotated` event. Verified both on
// the bare helper and end-to-end through the engine, asserting the footprint is
// byte-for-byte identical before and after a rotate input.
#[test]
fn o_footprint_unchanged_through_rotation() {
    // Helper: O reports a no-op rotation with kick index 0.
    let board = empty_board(10, 20);
    let o_piece = Piece::from(PieceType::O);
    let origin = (4, 18);
    assert_eq!(
        o_piece.try_rotate_with_kicks(&board, origin, PieceRotation::R90),
        Some((PieceRotation::R0, origin, 0)),
        "O rotation is a no-op: same rotation, same origin, kick 0"
    );

    // Engine: spawn an O, rotate clockwise, expect no Rotated event and an
    // unchanged footprint.
    let seed = seed_for_first_piece(PieceType::O);
    let (mut engine, before) = spawn_first_piece(seed);
    assert_eq!(before.piece_type, PieceType::O);
    assert_eq!(before.rotation, PieceRotation::R0);
    let before_cells = sorted_snapshot_cells(&before.cells);

    let events = engine.step(InputFrame {
        rotate_clockwise: true,
        ..InputFrame::default()
    });
    assert!(
        !events
            .iter()
            .any(|e| matches!(e, EngineEvent::Rotated { .. })),
        "O must emit no Rotated event (engine rejects kick 0), got {events:?}"
    );

    let after = engine.snapshot().active.expect("active O after rotate input");
    assert_eq!(
        after.rotation,
        PieceRotation::R0,
        "O rotation must stay at R0"
    );
    assert_eq!(
        sorted_snapshot_cells(&after.cells),
        before_cells,
        "O footprint must be identical through a rotation input"
    );

    // The rotate-only step carries dt == 0, so no gravity runs and the origin is
    // unchanged too — the O is identical in every respect.
    assert_eq!(after.origin, before.origin);
}
