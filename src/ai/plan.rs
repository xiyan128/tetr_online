//! Plan-to-input: render a chosen placement into engine [`InputFrame`]s (AI3.4).
//!
//! The planner (AI3.3) picks a [`Placement`] and movegen records the [`Move`] path
//! that reaches it. This module is the last step before the controller: it turns
//! that abstract path into the concrete per-frame button presses the engine reads,
//! honoring the engine's input model exactly.
//!
//! # The engine's per-frame input model (why this is not a 1:1 mapping)
//!
//! [`Engine::step`](crate::engine::Engine::step) consumes **at most one** of each
//! action per frame and in a fixed precedence (see `api.rs::step`): a hold, then a
//! hard drop (which returns immediately), then *one* rotation (CW **xor** CCW), then
//! *one* lateral cell (left **xor** right), then a soft drop. So a path like
//! `[Cw, Left, Left]` cannot be one frame — each lateral cell is its own pulse, and a
//! rotation that shares a frame with a shift would be applied *before* the shift
//! against a different pose. To stay faithful to the path movegen validated, this
//! translator emits **one [`Move`] per [`InputFrame`]**: discrete one-cell lateral
//! pulses (DAS is player-side, so there is no auto-repeat to model here — finding in
//! the M2 plan), one rotation per frame, the hold as its own leading frame, and a
//! single trailing hard-drop frame.
//!
//! # Soft drop is the sonic-drop approximation
//!
//! Movegen's [`Move::SoftDrop`] means "fall straight to the floor" (a sonic drop),
//! but one `InputFrame { soft_drop: true, .. }` only descends a single cell. So a
//! soft drop is expanded into as many soft-drop frames as it takes the piece to
//! come to rest from its current pose — computed by walking a cloned [`ActivePiece`]
//! with the engine's own [`Piece::try_move`], the same primitive movegen used. This
//! keeps lateral *tucks under an overhang* (`SoftDrop` then `Left`/`Right`) faithful:
//! the piece actually reaches the floor before the tuck shift.
//!
//! # Frames carry `dt_seconds == 0`
//!
//! Every maneuvering frame uses `dt_seconds == 0.0`, which makes the engine's
//! `advance_time` a no-op (it early-returns on zero dt): gravity and the lock timer
//! never advance *while* the AI is positioning the piece, so the maneuver lands the
//! piece at exactly the column and rotation movegen intended regardless of how many
//! frames it spans. The trailing hard-drop frame then locks it. Real wall-clock
//! pacing (think-time, acting cadence) is the controller's job (AI3.5), not this
//! pure translator's.
//!
//! # Determinism
//!
//! Pure Rust, no Bevy, no RNG, no clock — like [`crate::engine`]. The frame list is a
//! deterministic function of `(board, start pose, path)`. Any finesse *misexecution*
//! a difficulty setting wants to inject lives in the controller's seeded RNG, never
//! here.

use crate::ai::movegen::{spawn_piece, Move, Placement};
use crate::engine::{ActivePiece, Board, InputFrame, MoveDirection, PieceAction};

/// Render `placement`'s movement path into the [`InputFrame`] sequence that drives
/// the active piece from `start` to the placement and locks it with a hard drop.
///
/// `start` is the pose the path was recorded from (the active piece's spawn pose,
/// as movegen normalizes it); `board` is the board the maneuver happens on, needed
/// only to expand each [`Move::SoftDrop`] into the right number of one-cell descents.
///
/// The returned `Vec` is: an optional leading hold frame, then one frame per path
/// move (lateral/rotation as a single pulse, a soft drop expanded to a sonic drop),
/// then exactly one trailing hard-drop frame. Feeding these to a fresh seeded
/// [`Engine`](crate::engine::Engine) one per `step` reproduces the placement (see the
/// round-trip test).
pub fn placement_to_inputs(
    board: &Board,
    start: &ActivePiece,
    placement: &Placement,
) -> Vec<InputFrame> {
    let mut frames = Vec::new();
    // Walk a clone of the start pose alongside the path so a soft drop knows how far
    // the piece falls. Mirror movegen's normalization: a clean piece at the start
    // origin/rotation, no inherited lock-down history.
    let mut piece = ActivePiece::new(start.piece_type(), start.origin());

    for mv in &placement.path {
        match mv {
            Move::Hold => {
                // A hold swap makes the held/next piece active at its own spawn
                // pose, so the shadow piece must switch with it. Keeping the
                // pre-hold piece here would measure the wrong shape in the soft-drop
                // expansion (and lateral tracking) below, so a held placement that
                // tucks under an overhang would desync from the placement the
                // planner chose. `placement.piece` is the resting pose of the
                // swapped-in piece, so its type is the post-hold active piece.
                piece = spawn_piece(placement.piece_type(), board.width(), board.height());
                frames.push(hold_frame());
            }
            Move::Left => {
                step_lateral(board, &mut piece, MoveDirection::Left);
                frames.push(pulse(|f| f.left = true));
            }
            Move::Right => {
                step_lateral(board, &mut piece, MoveDirection::Right);
                frames.push(pulse(|f| f.right = true));
            }
            Move::Cw => {
                rotate(&mut piece, board, RotationDir::Cw);
                frames.push(pulse(|f| f.rotate_clockwise = true));
            }
            Move::Ccw => {
                rotate(&mut piece, board, RotationDir::Ccw);
                frames.push(pulse(|f| f.rotate_counterclockwise = true));
            }
            Move::SoftDrop => {
                // Sonic drop: one soft-drop pulse per cell the piece can still fall.
                let cells = drop_to_floor(board, &mut piece);
                for _ in 0..cells {
                    frames.push(pulse(|f| f.soft_drop = true));
                }
            }
        }
    }

    frames.push(pulse(|f| f.hard_drop = true));
    frames
}

/// A neutral, zero-`dt` frame with one field set by `set` — the per-frame pulse model
/// (exactly one action, no gravity advance).
fn pulse(set: impl FnOnce(&mut InputFrame)) -> InputFrame {
    let mut frame = InputFrame {
        dt_seconds: 0.0,
        ..InputFrame::default()
    };
    set(&mut frame);
    frame
}

/// The leading hold frame. Hold is its own frame because the engine applies a hold
/// *before* (and a hard drop short-circuits) any movement in the same `step`.
fn hold_frame() -> InputFrame {
    pulse(|f| f.hold = true)
}

/// Advance the shadow `piece` one lateral cell if the engine would allow it, so the
/// running pose stays in sync for a later soft-drop expansion. A blocked shift is a
/// no-op here just as it is in the engine (the emitted pulse is then a no-op too —
/// movegen never produces such a path, but staying in lockstep is cheap insurance).
fn step_lateral(board: &Board, piece: &mut ActivePiece, dir: MoveDirection) {
    if let Some(origin) = piece.piece().try_move(board, piece.origin(), dir) {
        piece.move_to(origin, PieceAction::Move);
    }
}

/// Direction of an SRS rotation, mirroring [`Move::Cw`] / [`Move::Ccw`].
enum RotationDir {
    Cw,
    Ccw,
}

/// Advance the shadow `piece` by one SRS rotation (kicks delegated to the engine),
/// matching movegen's `rotate` so the running pose tracks the path. A no-op kick
/// (O piece / kick number 0) leaves the pose unchanged, like the engine.
fn rotate(piece: &mut ActivePiece, board: &Board, dir: RotationDir) {
    use crate::engine::{PieceRotation, RotationDirection};
    let (engine_dir, target) = match dir {
        RotationDir::Cw => (
            RotationDirection::Clockwise,
            piece.rotation() + PieceRotation::R90,
        ),
        RotationDir::Ccw => (
            RotationDirection::Counterclockwise,
            piece.rotation() + PieceRotation::R270,
        ),
    };
    if let Some((rotation, origin, kick_number)) =
        piece
            .piece()
            .try_rotate_with_kicks(board, piece.origin(), target)
    {
        if kick_number != 0 {
            piece.rotate_to(rotation, origin, engine_dir, kick_number, false);
        }
    }
}

/// Drop the shadow `piece` straight to the floor, returning how many cells it fell
/// (the number of soft-drop pulses a sonic drop expands to). Uses the engine's
/// `try_move(.., Down)`, the same primitive movegen's `soft_drop` uses.
fn drop_to_floor(board: &Board, piece: &mut ActivePiece) -> usize {
    let mut cells = 0;
    while let Some(origin) = piece
        .piece()
        .try_move(board, piece.origin(), MoveDirection::Down)
    {
        piece.move_to(origin, PieceAction::SoftDrop);
        cells += 1;
    }
    cells
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::movegen::{generate, spawn_piece};
    use crate::engine::{Engine, EngineConfig, PieceRotation, PieceType};

    /// Absolute occupied cells of a piece at its pose, sorted for comparison.
    fn cells_of(piece: &ActivePiece) -> Vec<(isize, isize)> {
        let (ox, oy) = piece.origin();
        let mut cells: Vec<(isize, isize)> = piece
            .piece()
            .cells()
            .iter()
            .map(|(x, y)| (x + ox, y + oy))
            .collect();
        cells.sort();
        cells
    }

    /// Build a placement that holds an explicit path, for the unit (non-round-trip)
    /// assertions — bypasses movegen so we can pin the exact frame translation.
    fn placement_with_path(piece: ActivePiece, path: Vec<Move>) -> Placement {
        Placement {
            piece,
            path,
            used_hold: false,
        }
    }

    #[test]
    fn n_left_moves_yield_n_left_pulses_then_one_hard_drop() {
        // A pure-lateral path of N lefts must render to N single-left pulses followed
        // by exactly one hard-drop frame, nothing else set.
        let board = Board::new(10, 20);
        let start = spawn_piece(PieceType::T, 10, 20);
        let n = 3;
        let path = vec![Move::Left; n];
        let frames = placement_to_inputs(&board, &start, &placement_with_path(start.clone(), path));

        assert_eq!(frames.len(), n + 1, "N lefts + 1 hard drop");
        for f in &frames[..n] {
            assert!(
                f.left && !f.right && !f.hard_drop,
                "each is a lone left pulse"
            );
            assert_eq!(f.dt_seconds, 0.0, "maneuver frames advance no time");
        }
        let last = frames.last().unwrap();
        assert!(last.hard_drop && !last.left, "final frame is the hard drop");
    }

    #[test]
    fn rotations_and_hold_render_one_per_frame_in_order() {
        // [Hold, Cw, Ccw] -> a hold frame, a CW frame, a CCW frame, then hard drop.
        let board = Board::new(10, 20);
        let start = spawn_piece(PieceType::T, 10, 20);
        let path = vec![Move::Hold, Move::Cw, Move::Ccw];
        let frames = placement_to_inputs(&board, &start, &placement_with_path(start.clone(), path));

        assert_eq!(frames.len(), 4);
        assert!(frames[0].hold && !frames[0].hard_drop);
        assert!(frames[1].rotate_clockwise && !frames[1].rotate_counterclockwise);
        assert!(frames[2].rotate_counterclockwise && !frames[2].rotate_clockwise);
        assert!(frames[3].hard_drop);
        // Exactly one action per frame (no frame sets two buttons).
        for f in &frames {
            let set = [
                f.left,
                f.right,
                f.soft_drop,
                f.hard_drop,
                f.rotate_clockwise,
                f.rotate_counterclockwise,
                f.hold,
            ]
            .iter()
            .filter(|b| **b)
            .count();
            assert_eq!(set, 1, "exactly one button per frame");
        }
    }

    #[test]
    fn soft_drop_expands_to_one_pulse_per_cell_fallen() {
        // On an empty 10x20 board an O at spawn rests several cells below; a path of
        // a single SoftDrop must expand to that many soft-drop pulses (a sonic drop),
        // not one.
        let board = Board::new(10, 20);
        let start = spawn_piece(PieceType::O, 10, 20);
        // Independently measure the fall distance with the same primitive.
        let mut shadow = ActivePiece::new(start.piece_type(), start.origin());
        let expected = drop_to_floor(&board, &mut shadow);
        assert!(expected > 1, "spawn should be well above the floor");

        let frames = placement_to_inputs(
            &board,
            &start,
            &placement_with_path(start.clone(), vec![Move::SoftDrop]),
        );
        let soft_pulses = frames.iter().filter(|f| f.soft_drop).count();
        assert_eq!(soft_pulses, expected, "one soft-drop pulse per cell fallen");
        assert!(frames.last().unwrap().hard_drop);
    }

    /// Drive `frames` into `engine` one per `step`, stopping right before the final
    /// hard-drop frame, and return the active piece's pose at that point (origin +
    /// rotation) — i.e. where the planner intended the piece to be before it locks.
    ///
    /// Note the engine drops a freshly spawned piece one cell on spawn, so the y of
    /// the pose mid-maneuver may sit one below movegen's recorded origin until a
    /// soft/hard drop normalizes it to the floor; the caller compares the *column*
    /// and rotation here, and proves the full pose via the post-lock board cells.
    fn run_until_hard_drop(
        engine: &mut Engine,
        frames: &[InputFrame],
    ) -> ((isize, isize), PieceRotation) {
        let split = frames.len() - 1; // everything before the trailing hard drop
        for frame in &frames[..split] {
            engine.step(frame.clone());
        }
        let snap = engine.snapshot();
        let active = snap.active.expect("piece still active before hard drop");
        (active.origin, active.rotation)
    }

    #[test]
    fn round_trip_reaches_intended_column_and_rotation_then_locks_there() {
        // The end-to-end proof: for the piece the engine actually spawns, take EVERY
        // movegen placement, translate it to frames, feed them into a fresh seeded
        // engine one per `step`, and assert
        //   (a) the active piece reaches the placement's column + rotation before the
        //       hard drop (the task's literal "reaches the intended column/rotation"),
        //   (b) the trailing hard-drop frame locks exactly the cells the planner
        //       intended onto the board (proves the full resting pose).
        //
        // We test whatever piece the seed spawns first (re-deriving placements for
        // *that* piece) rather than a hard-coded type, so the test can never go
        // vacuous if the generator/seed changes — and we assert a placement was
        // actually checked.
        let config = EngineConfig::default();
        let w = config.board_width;
        let h = config.visible_height;
        let board = Board::with_top_margin(w, h, config.buffer_height);
        let seed = 7;

        let spawned = Engine::new(config.clone(), seed)
            .snapshot()
            .next_queue
            .first()
            .copied()
            .expect("engine always has a queued piece");

        let start = spawn_piece(spawned, w, h);
        let placements = generate(&board, &start);
        assert!(
            !placements.is_empty(),
            "movegen should find placements for the spawned {spawned:?}"
        );

        let mut checked = 0;
        for placement in &placements {
            let frames = placement_to_inputs(&board, &start, placement);

            let mut engine = Engine::new(config.clone(), seed);
            let (origin, rotation) = run_until_hard_drop(&mut engine, &frames);
            assert_eq!(
                rotation,
                placement.rotation(),
                "{spawned:?} reached wrong rotation; path = {:?}",
                placement.path
            );
            // Column must match exactly; y is normalized by the drop (see the
            // helper's note), and the full pose is proven by the board cells below.
            assert_eq!(
                origin.0,
                placement.origin().0,
                "{spawned:?} reached wrong column; path = {:?}",
                placement.path
            );

            // The hard-drop frame locks the piece. Its resulting board cells must
            // equal the placement's resting cells (same column, dropped to floor).
            let expected_cells = cells_of(&placement.piece);
            engine.step(frames.last().unwrap().clone());
            let mut board_cells: Vec<(isize, isize)> = engine
                .snapshot()
                .board_cells
                .iter()
                .map(|c| (c.x, c.y))
                .collect();
            board_cells.sort();
            assert_eq!(
                board_cells, expected_cells,
                "{spawned:?} hard drop locked the wrong cells; path = {:?}",
                placement.path
            );
            checked += 1;
        }

        assert!(
            checked > 0,
            "the round trip must exercise at least one placement"
        );
    }

    #[test]
    fn held_tuck_placement_round_trips_through_the_engine() {
        // The hold-path analogue of the no-hold round trip: a HELD placement whose
        // maneuver tucks under an overhang (soft-drop into the open columns, then a
        // shift onto the floor beneath the shelf). It executes faithfully on the real
        // engine ONLY if the translator re-bases its shadow piece onto the swapped-in
        // piece at Move::Hold; the bug it guards drops the pre-hold shape and lands
        // the held piece in the wrong cells. The no-hold round trip can't catch this
        // (it never exercises a piece swap), so this is the regression guard for the
        // hold path's plan-to-input fidelity.
        use crate::ai::movegen::generate_with_hold;
        use crate::engine::CellKind;

        let config = EngineConfig::default();
        let w = config.board_width;
        let h = config.visible_height;
        let buffer = config.buffer_height;
        let seed = 7;
        let shelf_y = 4;

        // Board (for movegen) and engine (for execution) carry the SAME overhang: a
        // shelf over the left columns with an open well on the right (4 wide, so any
        // piece — not just a vertical I — can drop through it), so reaching the floor
        // under the shelf forces a soft-drop in the well then a shift under the shelf.
        let mut board = Board::with_top_margin(w, h, buffer);
        let mut engine = Engine::new(config.clone(), seed);
        for x in 0..(w as isize - 4) {
            board.set(x, shelf_y, CellKind::Some(PieceType::I));
            engine.set_cell(x, shelf_y, CellKind::Some(PieceType::I));
        }

        engine.step(InputFrame::default()); // spawn the active piece
        let snapshot = engine.snapshot();
        let active = snapshot.active.as_ref().expect("active piece spawned");
        let start = spawn_piece(active.piece_type, w, h);
        // Hold is empty after the first spawn, so a hold swaps in the next-queue piece
        // — exactly what generate_with_hold enumerates with hold = None.
        let queue_front = snapshot
            .next_queue
            .first()
            .copied()
            .expect("a queued piece");

        let placements = generate_with_hold(&board, &start, None, Some(queue_front), |pt| {
            spawn_piece(pt, w, h)
        });
        // A held placement that tucks: holds, soft-drops, then shifts AFTER the
        // soft-drop (the tuck signature).
        let tuck = placements
            .iter()
            .find(|p| {
                if !p.used_hold {
                    return false;
                }
                let drop_idx = p.path.iter().position(|m| *m == Move::SoftDrop);
                let last_shift = p
                    .path
                    .iter()
                    .rposition(|m| matches!(m, Move::Left | Move::Right));
                matches!((drop_idx, last_shift), (Some(d), Some(s)) if s > d)
            })
            .expect("a held tuck placement should be reachable under the shelf");

        let frames = placement_to_inputs(&board, &start, tuck);

        // The cells already on the board (the shelf); the maneuver must add exactly
        // the tuck's cells (no line clear — the shelf row never completes).
        let before: std::collections::HashSet<(isize, isize)> = engine
            .snapshot()
            .board_cells
            .iter()
            .map(|c| (c.x, c.y))
            .collect();
        for frame in &frames {
            engine.step(frame.clone());
        }
        let mut locked: Vec<(isize, isize)> = engine
            .snapshot()
            .board_cells
            .iter()
            .map(|c| (c.x, c.y))
            .filter(|c| !before.contains(c))
            .collect();
        locked.sort();

        assert_eq!(
            locked,
            cells_of(&tuck.piece),
            "held tuck diverged from the planner's placement; path = {:?}",
            tuck.path
        );
    }

    #[test]
    fn held_soft_drop_uses_the_swapped_in_pieces_shape_not_the_active_pieces() {
        // The crisp contract: after Move::Hold a soft-drop expands by the SWAPPED-IN
        // piece's fall (the engine maneuvers that piece), not the pre-hold active
        // piece's. We give the two pieces different footprints over a single pillar so
        // their falls differ — directly catching a regression where the shadow keeps
        // tracking the active piece. (The end-to-end round trip above can't isolate
        // this: a trailing hard-drop normalizes an over-counted final soft-drop.)
        use crate::engine::CellKind;

        let mut board = Board::new(10, 20);
        // A pillar at column 6 only. The swapped-in I (cols 3-6) catches it and rests
        // high; the active O (cols 4-5) misses it and would fall to the floor.
        for y in 0..=10 {
            board.set(6, y, CellKind::Some(PieceType::I));
        }

        let active_o = spawn_piece(PieceType::O, 10, 20); // pre-hold (active) piece
        let held_i = spawn_piece(PieceType::I, 10, 20); // post-hold (swapped-in) piece

        // Each piece's true fall on this board: the I onto the pillar, the O past it.
        let held_fall = drop_to_floor(&board, &mut ActivePiece::new(PieceType::I, held_i.origin()));
        let active_fall = drop_to_floor(
            &board,
            &mut ActivePiece::new(PieceType::O, active_o.origin()),
        );
        assert_ne!(
            held_fall, active_fall,
            "premise: the two footprints must fall different distances"
        );

        let placement = Placement {
            piece: held_i,
            path: vec![Move::Hold, Move::SoftDrop],
            used_hold: true,
        };
        let soft_pulses = placement_to_inputs(&board, &active_o, &placement)
            .iter()
            .filter(|f| f.soft_drop)
            .count();

        assert_eq!(
            soft_pulses, held_fall,
            "soft-drop must expand by the swapped-in I's fall ({held_fall}), not the active O's ({active_fall})"
        );
    }
}
