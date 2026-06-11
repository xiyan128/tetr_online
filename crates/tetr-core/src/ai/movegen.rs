//! SRS-aware placement movement generation.
//!
//! A placement search needs the *set of final resting poses* the active piece can
//! reach from spawn, plus the button sequence to reach each one. This module
//! enumerates them with a breadth-first search over `(x, y, rotation)` states,
//! mirroring Cold Clear's movegen (research finding \[8\]): a frontier + a
//! [`FxHashSet`] visited-set, soft-drop *approximated* rather than enumerating every
//! gravity cell, and — crucially — **all SRS wall kicks delegated to the engine's
//! own [`Piece::try_rotate_with_kicks`] / [`Piece::try_move`]**. The search never
//! re-encodes a kick table, so it can never disagree with the real rules (and an
//! AI can never exploit a kick-table divergence bug).
//!
//! # What a placement is
//!
//! A [`Placement`] is a *resting* pose: a pose with solid ground (a wall or a
//! filled cell) directly beneath it, so the piece would lock there on a hard drop.
//! Every reachable resting pose is emitted, including ones reached by rotating
//! *into* a slot as the final action — that is what lets the evaluator see tucks
//! and T-spins (the recorded [`ActivePiece`] keeps its `last_successful_action ==
//! Rotate`, which is exactly what [`classify_t_spin`]
//! checks).
//!
//! # Soft-drop approximation
//!
//! Real soft-drop is a per-tick gravity acceleration; enumerating each intermediate
//! cell would explode the state space for no decision-relevant gain. Following Cold
//! Clear, a [`Move::SoftDrop`] in a path means "fall straight down to the floor"
//! (a sonic drop). Lateral *tucks* under an overhang are still found: the BFS can
//! move left/right at any reached height, then soft-drop again. So a tuck that
//! needs "soft-drop, shift, soft-drop" is reachable, while the path stays short.
//!
//! # Hold
//!
//! [`generate_with_hold`] also enumerates placements for the piece a hold swap would
//! bring into play (the current hold piece, or — if hold is empty — the next piece
//! from the queue), tagging each [`Placement`] with [`Placement::used_hold`] and
//! prefixing its path with [`Move::Hold`]. A planner can therefore choose across
//! "place the current piece" *and* "hold, then place the other piece" in one set.
//!
//! # Determinism
//!
//! Pure Rust, no Bevy, no RNG, no clock. The BFS visits states in a fixed order and
//! the output [`Vec`] is sorted into a canonical order, so the same board + piece
//! always yields the same placement list in the same order — a search built on top
//! stays reproducible.

use std::collections::VecDeque;

use rustc_hash::FxHashSet;
use smallvec::SmallVec;

use crate::engine::{
    classify_t_spin, is_t_slot, ActivePiece, MoveDirection, Occupancy, Piece, PieceAction,
    PieceRotation, PieceType, RotationDirection, TSpinKind,
};

/// One button press in the path to a placement.
///
/// [`Move::SoftDrop`] is the soft-drop *approximation* (fall straight to the
/// floor), not a single gravity cell — see the [module docs](self).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Move {
    /// Shift one cell left.
    Left,
    /// Shift one cell right.
    Right,
    /// Rotate clockwise (SRS, kicks delegated to the engine).
    Cw,
    /// Rotate counter-clockwise (SRS, kicks delegated to the engine).
    Ccw,
    /// Soft-drop: fall straight down until resting (sonic drop approximation).
    SoftDrop,
    /// Swap with the hold slot. Only ever the first move of a path, emitted by
    /// [`generate_with_hold`].
    Hold,
}

/// A reachable final resting pose for the active piece.
///
/// Carries the landed [`ActivePiece`] (so the caller can lock it with
/// the search's `BitBoard::lock_piece` and classify a T-spin with
/// [`classify_t_spin`]) and the [`Move`] path that
/// reaches it from spawn.
#[derive(Clone, Debug)]
pub struct Placement {
    /// The piece at its resting pose, with the lock-down / rotation bookkeeping the
    /// engine primitives expect (notably `last_successful_action`, which T-spin
    /// classification reads).
    pub piece: ActivePiece,
    /// The button sequence from the start pose to this resting pose. A `SmallVec` so the
    /// BFS — which clones+extends a path per reachable pose — stays stack-allocated for
    /// the short paths (≤16 moves) that dominate, avoiding per-pose heap churn.
    pub path: SmallVec<[Move; 16]>,
    /// Whether reaching this placement required a hold swap (the piece here is the
    /// *held/next* piece, not the one that was active). Always `false` from
    /// [`generate`]; possibly `true` from [`generate_with_hold`].
    pub used_hold: bool,
}

impl Placement {
    /// The resting pose's rotation.
    pub fn rotation(&self) -> PieceRotation {
        self.piece.rotation()
    }

    /// The resting pose's origin.
    pub fn origin(&self) -> (isize, isize) {
        self.piece.origin()
    }

    /// The piece type at this placement (the held/next piece when [`used_hold`]).
    ///
    /// [`used_hold`]: Placement::used_hold
    pub fn piece_type(&self) -> PieceType {
        self.piece.piece_type()
    }
}

/// The geometric part of a search node's identity: origin + rotation.
///
/// On its own a pose under-identifies a node — T-spin classification reads the
/// piece's action history, not just where it sits — so the BFS keys its visited
/// set on `(PoseKey, spin_rank)` and dedups *emissions* on
/// `(PoseKey, classification)`; see [`spin_rank`] / [`classification_key`].
/// Rotation is stored as its `u8` discriminant so the key is totally ordered
/// (`PieceRotation` is `Eq`/`Hash` but not `Ord`).
type PoseKey = (isize, isize, u8);

fn pose_key(piece: &ActivePiece) -> PoseKey {
    let (x, y) = piece.origin();
    (x, y, piece.rotation() as u8)
}

/// Enumerate every reachable final placement for `start` on `board`.
///
/// Breadth-first over `(x, y, rotation)` from `start`'s current pose, delegating
/// SRS to the engine. The returned placements are *resting* poses (ground beneath
/// them), each with the shortest [`Move`] path found to reach it. The list is in a
/// deterministic, canonical order.
pub fn generate<B: Occupancy>(board: &B, start: &ActivePiece) -> Vec<Placement> {
    let mut placements = enumerate(board, start, false);
    sort_placements(&mut placements);
    placements
}

/// Like [`generate`], but also includes the placements reachable *after* a hold
/// swap, for hold-aware search.
///
/// The swap brings in `hold` if the hold slot is occupied, otherwise the first
/// piece of `queue` (mirroring the engine's hold rule: an empty hold pulls the
/// next piece). Those placements are tagged [`Placement::used_hold`] and their
/// paths are prefixed with [`Move::Hold`]. If no piece can be swapped in (hold
/// empty *and* queue empty), only the no-hold placements are returned.
///
/// `spawn_for` supplies the spawn pose of a swapped-in piece — pass a closure
/// wrapping the board geometry (e.g. from `EngineConfig`); see the unit tests for
/// the idiom. This keeps `movegen` free of snapshot/config plumbing.
pub fn generate_with_hold<B: Occupancy>(
    board: &B,
    start: &ActivePiece,
    hold: Option<PieceType>,
    queue_front: Option<PieceType>,
    spawn_for: impl Fn(PieceType) -> ActivePiece,
) -> Vec<Placement> {
    let mut placements = enumerate(board, start, false);

    // The piece a hold swap would make active: the current hold piece, or the next
    // queued piece when the hold slot is empty. Movegen always offers the swap as a
    // candidate; the once-per-piece hold restriction is enforced upstream by the
    // controller (the engine tracks it via `hold_used`), so the planner is free to
    // pick a held placement only when a hold is actually available.
    if let Some(swapped_in) = hold.or(queue_front) {
        let swapped_start = spawn_for(swapped_in);
        let mut held = enumerate(board, &swapped_start, true);
        placements.append(&mut held);
    }

    sort_placements(&mut placements);
    placements
}

/// Core BFS. `used_hold` is stamped onto every emitted placement and, when set,
/// prefixes each path with [`Move::Hold`].
fn enumerate<B: Occupancy>(board: &B, start: &ActivePiece, used_hold: bool) -> Vec<Placement> {
    // Normalize the start pose: the search re-derives reachable poses from scratch,
    // so it begins from a clean piece at the start origin AND rotation, with
    // spawn-fresh history ([`ActivePiece::at_pose`]) — no inherited lock-down or
    // kick state that could mis-flag a T-spin on the *start* pose itself.
    let start = ActivePiece::at_pose(start.piece_type(), start.origin(), start.rotation());

    // A node's identity is its pose PLUS its spin rank: two paths to the same
    // `(x, y, rotation)` are interchangeable for geometry, but NOT for scoring —
    // T-spin classification reads the action history, so a shift-final arrival
    // must not shadow a rotate-final (spin) arrival at the same pose.
    let mut visited: FxHashSet<(PoseKey, u8)> = FxHashSet::default();
    // Emission is deduplicated separately, per (pose, classification): ranks are
    // search bookkeeping, but two resting nodes whose lock would *classify*
    // identically are interchangeable placements — keep the first (BFS-shortest).
    let mut emitted: FxHashSet<(PoseKey, u8)> = FxHashSet::default();
    let mut frontier: VecDeque<(ActivePiece, SmallVec<[Move; 16]>)> = VecDeque::new();
    let mut placements: Vec<Placement> = Vec::new();

    visited.insert((pose_key(&start), spin_rank(&start)));
    frontier.push_back((start, SmallVec::new()));

    while let Some((piece, path)) = frontier.pop_front() {
        // If this pose rests on ground, it is a candidate final placement.
        if is_resting(board, &piece)
            && emitted.insert((pose_key(&piece), classification_key(&piece, board)))
        {
            let mut full_path = path.clone();
            if used_hold {
                full_path.insert(0, Move::Hold);
            }
            placements.push(Placement {
                piece: piece.clone(),
                path: full_path,
                used_hold,
            });
        }

        // Expand neighbours: lateral shifts, both rotations, and a soft-drop.
        for mv in [Move::Left, Move::Right, Move::Cw, Move::Ccw, Move::SoftDrop] {
            if let Some(next) = apply_move(board, &piece, mv) {
                if visited.insert((pose_key(&next), spin_rank(&next))) {
                    let mut next_path = path.clone();
                    next_path.push(mv);
                    frontier.push_back((next, next_path));
                }
            }
        }
    }

    placements
}

/// The T-spin-relevant component of a search node's identity beyond its pose.
///
/// `0`: a lock here never classifies (shift/drop-final with no sticky flag).
/// `1`: rotate-final (kick 1-4) — classification enabled, Mini stays Mini.
/// `2`: kick-5 rotate-final or the sticky §12.4 flag — Mini promotes to Full and
/// the state survives later non-rotation moves.
///
/// Non-T pieces never classify, so they keep a single rank — the visited set (and
/// thus the BFS state space) only grows for the T piece.
fn spin_rank(piece: &ActivePiece) -> u8 {
    if piece.piece_type() != PieceType::T {
        return 0;
    }
    if piece.used_kick_5_into_t_slot() {
        return 2;
    }
    if piece.last_successful_action() == PieceAction::Rotate {
        if piece.last_rotation_kick_number() == Some(5) {
            2
        } else {
            1
        }
    } else {
        0
    }
}

/// What locking `piece` at its current pose would classify as, as a small
/// emission-dedup key: `0` no spin, `1` Mini, `2` Full.
fn classification_key<B: Occupancy>(piece: &ActivePiece, board: &B) -> u8 {
    match classify_t_spin(piece, board) {
        None => 0,
        Some(TSpinKind::Mini) => 1,
        Some(TSpinKind::Full) => 2,
    }
}

/// Apply one [`Move`] to `piece` on `board`, delegating SRS to the engine.
///
/// Returns the resulting piece, or `None` if the move is blocked (or a no-op, e.g.
/// rotating an O piece, or a soft-drop that does not change the resting row).
fn apply_move<B: Occupancy>(board: &B, piece: &ActivePiece, mv: Move) -> Option<ActivePiece> {
    match mv {
        Move::Left => shift(board, piece, MoveDirection::Left),
        Move::Right => shift(board, piece, MoveDirection::Right),
        Move::Cw => rotate(board, piece, RotationDirection::Clockwise),
        Move::Ccw => rotate(board, piece, RotationDirection::Counterclockwise),
        Move::SoftDrop => soft_drop(board, piece),
        // Hold is never enqueued as a BFS neighbour; it is prepended by `enumerate`.
        Move::Hold => None,
    }
}

/// One lateral cell, via the engine's `try_move`.
pub(crate) fn shift<B: Occupancy>(
    board: &B,
    piece: &ActivePiece,
    dir: MoveDirection,
) -> Option<ActivePiece> {
    let origin = piece.piece().try_move(board, piece.origin(), dir)?;
    let mut moved = piece.clone();
    moved.move_to(origin, PieceAction::Move);
    Some(moved)
}

/// One SRS rotation, via the engine's `try_rotate_with_kicks` (kicks included).
///
/// Returns `None` for a no-op (kick number `0`, the O piece) or a pose that does
/// not actually change `(origin, rotation)`, so the BFS does not loop.
pub(crate) fn rotate<B: Occupancy>(
    board: &B,
    piece: &ActivePiece,
    dir: RotationDirection,
) -> Option<ActivePiece> {
    let target = match dir {
        RotationDirection::Clockwise => piece.rotation() + PieceRotation::R90,
        RotationDirection::Counterclockwise => piece.rotation() + PieceRotation::R270,
    };
    let (rotation, origin, kick_number) =
        piece
            .piece()
            .try_rotate_with_kicks(board, piece.origin(), target)?;
    // kick_number 0 == the O piece's no-op rotation (mirrors api.rs::rotate).
    if kick_number == 0 {
        return None;
    }
    // Mirror the engine's §7.5 point-5 override exactly (api.rs::rotate_active_piece):
    // if SRS test 5 placed a T into a T-slot, set the sticky flag so the spin
    // classifies Full and SURVIVES later non-rotation moves (§12.4) — which is what
    // the engine will do when this path is replayed. Without it, movegen would
    // undervalue every kick-5 spin that shifts before locking.
    let entered_t_slot_with_kick_5 = kick_number == 5 && piece.piece_type() == PieceType::T && {
        let mut probe = piece.clone();
        probe.rotate_to(rotation, origin, dir, kick_number, false);
        is_t_slot(&probe, board)
    };
    let mut rotated = piece.clone();
    rotated.rotate_to(
        rotation,
        origin,
        dir,
        kick_number,
        entered_t_slot_with_kick_5,
    );
    Some(rotated)
}

/// Soft-drop approximation: fall straight to the floor.
///
/// Returns `None` if the piece is already resting (no downward movement), so the
/// BFS treats it as a no-op rather than a self-loop.
fn soft_drop<B: Occupancy>(board: &B, piece: &ActivePiece) -> Option<ActivePiece> {
    let mut dropped = piece.clone();
    let mut moved = false;
    while let Some(origin) = dropped
        .piece()
        .try_move(board, dropped.origin(), MoveDirection::Down)
    {
        dropped.move_to(origin, PieceAction::SoftDrop);
        moved = true;
    }
    moved.then_some(dropped)
}

/// Whether `piece` rests on ground: it cannot move one cell down.
fn is_resting<B: Occupancy>(board: &B, piece: &ActivePiece) -> bool {
    piece
        .piece()
        .try_move(board, piece.origin(), MoveDirection::Down)
        .is_none()
}

/// Sort placements into a canonical, deterministic order: by rotation, then origin,
/// then `used_hold`. Stable so the (shortest) path found first is preserved.
fn sort_placements(placements: &mut [Placement]) {
    placements.sort_by(|a, b| {
        let ka = (a.rotation() as u8, a.origin().0, a.origin().1, a.used_hold);
        let kb = (b.rotation() as u8, b.origin().0, b.origin().1, b.used_hold);
        ka.cmp(&kb)
    });
}

/// The spawn pose a freshly dealt `piece_type` takes for board geometry
/// `(width, visible_height)` — the engine's spawn coordinates. A convenience for
/// callers building the `spawn_for` closure of [`generate_with_hold`].
pub fn spawn_piece(piece_type: PieceType, width: usize, visible_height: usize) -> ActivePiece {
    let origin = Piece::from(piece_type).spawn_coords(width, visible_height);
    ActivePiece::new(piece_type, origin)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{Board, CellKind};

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

    /// Replay a placement's path from `start` on `board` and assert it ends exactly
    /// at the placement's pose — proves the recorded path is faithful.
    fn replay(board: &Board, start: &ActivePiece, placement: &Placement) -> ActivePiece {
        let mut piece = ActivePiece::new(start.piece_type(), start.origin());
        for mv in &placement.path {
            piece = apply_move(board, &piece, *mv)
                .unwrap_or_else(|| panic!("path move {mv:?} was blocked on replay"));
        }
        piece
    }

    #[test]
    fn flat_board_reaches_every_column_for_o() {
        // O piece on a flat empty board: one resting pose per horizontal position.
        let board = Board::new(10, 20);
        let start = spawn_piece(PieceType::O, 10, 20);
        let placements = generate(&board, &start);

        // O occupies a 2-wide footprint, so 9 distinct horizontal landing spots.
        let resting_origins: FxHashSet<isize> = placements.iter().map(|p| p.origin().0).collect();
        assert_eq!(resting_origins.len(), 9, "O should reach all 9 columns");

        // Every placement actually rests on the floor (lowest cell at y == 0).
        for p in &placements {
            let min_y = cells_of(&p.piece).iter().map(|(_, y)| *y).min().unwrap();
            assert_eq!(min_y, 0, "placement must rest on the floor");
        }
    }

    #[test]
    fn recorded_paths_are_faithful() {
        // Every placement's path, replayed from spawn, must reach its pose.
        let board = Board::new(10, 20);
        for piece_type in PieceType::all() {
            let start = spawn_piece(piece_type, 10, 20);
            for placement in generate(&board, &start) {
                let landed = replay(&board, &start, &placement);
                assert_eq!(
                    pose_key(&landed),
                    pose_key(&placement.piece),
                    "{piece_type:?} path {:?} did not reach its pose",
                    placement.path
                );
            }
        }
    }

    #[test]
    fn no_duplicate_pose_classifications() {
        // Emission is deduplicated per (pose, T-spin classification): a pose may
        // legitimately appear twice when a spin and a non-spin arrival both exist
        // (they score differently), but never twice with the SAME classification.
        let board = Board::new(10, 20);
        for piece_type in PieceType::all() {
            let start = spawn_piece(piece_type, 10, 20);
            let placements = generate(&board, &start);
            let mut keys: Vec<(PoseKey, u8)> = placements
                .iter()
                .map(|p| (pose_key(&p.piece), classification_key(&p.piece, &board)))
                .collect();
            let n = keys.len();
            keys.sort();
            keys.dedup();
            assert_eq!(
                keys.len(),
                n,
                "{piece_type:?} produced duplicate (pose, class) placements"
            );
        }
    }

    #[test]
    fn determinism_same_inputs_same_order() {
        let board = Board::new(10, 20);
        let start = spawn_piece(PieceType::T, 10, 20);
        let a = generate(&board, &start);
        let b = generate(&board, &start);
        assert_eq!(a.len(), b.len());
        for (pa, pb) in a.iter().zip(&b) {
            assert_eq!(pose_key(&pa.piece), pose_key(&pb.piece));
            assert_eq!(pa.path, pb.path);
        }
    }

    #[test]
    fn lateral_tuck_under_a_shelf_needs_softdrop_then_shift() {
        // A placement reachable ONLY by soft-dropping first and then sliding
        // sideways *under an overhanging shelf* — a genuine lateral tuck. A piece
        // already over the shelf up high would rest ON it; to reach the floor
        // beneath, it must drop where the shelf isn't, then slide under. So the
        // tuck path is "...SoftDrop... then a lateral shift".
        //
        //   col:  0 1 2 3 4
        //    y4:        X X X     shelf over cols 2-4
        //    y0:  . . . . .       open floor
        let mut board = Board::new(5, 20);
        for x in 2..5 {
            board.set(x, 4, CellKind::Some(PieceType::I)); // overhanging shelf
        }

        // A vertical I tucks cleanly: rotate upright, drop on the left, slide right
        // under the shelf onto the floor.
        let start = spawn_piece(PieceType::I, 5, 20);
        let placements = generate(&board, &start);

        // Some reachable placement shifts laterally AFTER a soft-drop (the tuck
        // signature), and that pose rests on the floor under the shelf.
        let tuck = placements.iter().find(|p| {
            let drop_idx = p.path.iter().position(|m| *m == Move::SoftDrop);
            let last_shift = p
                .path
                .iter()
                .rposition(|m| matches!(m, Move::Left | Move::Right));
            matches!((drop_idx, last_shift), (Some(d), Some(s)) if s > d)
        });
        assert!(
            tuck.is_some(),
            "a lateral tuck (shift AFTER soft-drop) should be reachable under the shelf; \
             paths = {:?}",
            placements.iter().map(|p| &p.path).collect::<Vec<_>>()
        );

        // And the tuck reaches under the shelf (a cell at y0 below the shelf cols).
        let tuck = tuck.unwrap();
        assert!(
            cells_of(&tuck.piece)
                .iter()
                .any(|(x, y)| *y == 0 && (2..5).contains(x)),
            "the tuck should land a cell on the floor under the shelf; cells = {:?}",
            cells_of(&tuck.piece)
        );
    }

    #[test]
    fn srs_illegal_poses_are_absent() {
        // On a flat board an I piece cannot rest "inside" the floor or overlapping
        // walls. Assert no placement has a cell below y==0 or outside [0, width).
        let board = Board::new(10, 20);
        for piece_type in PieceType::all() {
            let start = spawn_piece(piece_type, 10, 20);
            for placement in generate(&board, &start) {
                for (x, y) in cells_of(&placement.piece) {
                    assert!(
                        (0..10).contains(&x) && y >= 0,
                        "{piece_type:?} placement has an out-of-bounds cell ({x},{y})"
                    );
                    // And the cell is not inside a filled block (board is empty).
                    assert_eq!(board.get_cell_kind(x, y), CellKind::None);
                }
            }
        }
    }

    #[test]
    fn t_spin_slot_is_reachable_via_softdrop_and_kick() {
        // Craft a T-spin overhang and assert movegen finds a placement that the
        // engine classifies as a T-spin, reached by soft-dropping in then rotating
        // (with a wall kick) INTO the slot — the "tuck/spin needing soft-drop + a
        // kick" case the task calls out.
        //
        // T lands at R180 (bump down) at origin (1,0): bar at (1,1),(2,1),(3,1),
        // bump at (2,0). Shoulders under the arms at (1,0),(3,0); an overhang lip
        // at (1,2) caps a corner so the slot is only enterable by rotating under
        // it. (Geometry verified empirically — see git history's probe.)
        //   col:  0 1 2 3 4
        //    y2:    X            overhang lip
        //    y0:    X . X        shoulders; the bump-slot (2,0) is open
        use crate::engine::classify_t_spin;
        let mut board = Board::new(5, 20);
        board.set(1, 0, CellKind::Some(PieceType::I));
        board.set(3, 0, CellKind::Some(PieceType::I));
        board.set(1, 2, CellKind::Some(PieceType::I)); // overhang lip

        let start = spawn_piece(PieceType::T, 5, 20);
        let placements = generate(&board, &start);

        // At least one reachable placement classifies as a T-spin.
        let spins: Vec<&Placement> = placements
            .iter()
            .filter(|p| classify_t_spin(&p.piece, &board).is_some())
            .collect();
        assert!(
            !spins.is_empty(),
            "a T-spin placement should be reachable in the crafted slot"
        );

        // The spin is reached by a *final rotation* (you rotate into the slot) and
        // its path required a soft-drop first. At least one spin used a non-trivial
        // wall kick (kick number > 1) — i.e. the kick table genuinely mattered, and
        // movegen got it for free by delegating to the engine.
        let tuck_spin = spins.iter().find(|p| {
            matches!(p.path.last(), Some(Move::Cw) | Some(Move::Ccw))
                && p.path.contains(&Move::SoftDrop)
        });
        assert!(
            tuck_spin.is_some(),
            "a T-spin reached by soft-drop then a final rotation should exist; spins = {:?}",
            spins.iter().map(|p| &p.path).collect::<Vec<_>>()
        );
        assert!(
            spins
                .iter()
                .any(|p| p.piece.last_rotation_kick_number().is_some_and(|k| k > 1)),
            "at least one reachable spin should use a non-trivial SRS wall kick"
        );
    }

    #[test]
    fn hold_adds_the_other_piece_placements() {
        // With an O active and a held T, generate_with_hold offers BOTH pieces'
        // placements; the held ones are tagged and Hold-prefixed.
        let board = Board::new(10, 20);
        let start = spawn_piece(PieceType::O, 10, 20);
        let placements = generate_with_hold(&board, &start, Some(PieceType::T), None, |pt| {
            spawn_piece(pt, 10, 20)
        });

        let has_o = placements
            .iter()
            .any(|p| !p.used_hold && p.piece_type() == PieceType::O);
        let has_t = placements
            .iter()
            .any(|p| p.used_hold && p.piece_type() == PieceType::T);
        assert!(has_o, "current (O) placements present");
        assert!(has_t, "held (T) placements present");

        // Every held placement's path starts with Hold.
        for p in placements.iter().filter(|p| p.used_hold) {
            assert_eq!(p.path.first(), Some(&Move::Hold));
        }
    }

    #[test]
    fn hold_empty_pulls_next_piece() {
        // Hold empty: the swap brings in the queue front.
        let board = Board::new(10, 20);
        let start = spawn_piece(PieceType::O, 10, 20);
        let placements = generate_with_hold(&board, &start, None, Some(PieceType::I), |pt| {
            spawn_piece(pt, 10, 20)
        });
        assert!(
            placements
                .iter()
                .any(|p| p.used_hold && p.piece_type() == PieceType::I),
            "an empty hold should offer the next (I) piece"
        );
    }

    #[test]
    fn shift_final_arrival_does_not_shadow_a_spin_arrival() {
        // Regression for the spin-shadowing bug: with the visited-set keyed on pose
        // alone, whichever path reached a pose FIRST (BFS-shortest) owned it — and a
        // shift-final arrival classifies as no spin, hiding a reachable rotate-final
        // spin at the same pose from the planner entirely.
        //
        //   col:  0 1 2 3
        //    y2:    X          lip
        //    y0:    X . X      shoulders; slot center (2, 1)
        //
        // The T can slide along the floor into origin (1, 0) at R0 (3 moves,
        // shift-final => None) or rotate into the same pose (4 moves, rotate-final
        // => Mini). Both must be emitted; before the fix only the shorter, spinless
        // one survived.
        let mut board = Board::new(10, 20);
        for (x, y) in [(1, 0), (3, 0), (1, 2)] {
            board.set(x, y, CellKind::Some(PieceType::I));
        }
        let start = spawn_piece(PieceType::T, 10, 20);
        let placements = generate(&board, &start);

        let at_pose: Vec<&Placement> = placements
            .iter()
            .filter(|p| pose_key(&p.piece) == (1, 0, PieceRotation::R0 as u8))
            .collect();
        let classes: Vec<u8> = at_pose
            .iter()
            .map(|p| classification_key(&p.piece, &board))
            .collect();
        assert!(
            classes.contains(&0) && classes.iter().any(|&c| c > 0),
            "pose (1,0,R0) must be emitted both as a plain placement and as a spin; got classes {classes:?} \
             from paths {:?}",
            at_pose.iter().map(|p| &p.path).collect::<Vec<_>>()
        );
        // The spin variant really is rotate-final and the plain one shift-final.
        let spin = at_pose
            .iter()
            .find(|p| classification_key(&p.piece, &board) > 0)
            .unwrap();
        assert!(matches!(spin.path.last(), Some(Move::Cw | Move::Ccw)));
        let plain = at_pose
            .iter()
            .find(|p| classification_key(&p.piece, &board) == 0)
            .unwrap();
        assert!(matches!(plain.path.last(), Some(Move::Left | Move::Right)));
    }

    #[test]
    fn rotation_sets_the_kick5_sticky_flag_like_the_engine() {
        // movegen's rotate() must mirror api.rs::rotate_active_piece's §7.5 point-5
        // override: an SRS test-5 rotation of a T into a T-slot sets the sticky flag
        // (so the spin classifies Full and survives later non-rotation moves). The
        // fixture is the engine's own regression board
        // (api.rs::engine_sets_point_5_override_when_kick_5_rotates_into_a_t_slot).
        let mut board = Board::new(10, 20);
        for (x, y) in [(5, 5), (4, 7), (3, 5), (3, 3)] {
            board.set(x, y, CellKind::Some(PieceType::I));
        }
        let piece = ActivePiece::at_pose(PieceType::T, (4, 5), PieceRotation::R0);

        let rotated = apply_move(&board, &piece, Move::Cw).expect("the kick-5 rotation resolves");
        assert_eq!(
            rotated.last_rotation_kick_number(),
            Some(5),
            "resolves via SRS test 5"
        );
        assert!(
            rotated.used_kick_5_into_t_slot(),
            "kick-5 into a T-slot must set the sticky flag, exactly as the engine does"
        );
        assert_eq!(
            classify_t_spin(&rotated, &board),
            Some(TSpinKind::Full),
            "the kick-5 slot classifies Full"
        );

        // Negative control: the same rotation on an empty board resolves without a
        // kick-5 slot and must NOT set the flag.
        let empty = Board::new(10, 20);
        let rotated = apply_move(&empty, &piece, Move::Cw).expect("rotation resolves");
        assert!(!rotated.used_kick_5_into_t_slot());
    }

    #[test]
    fn placement_classification_matches_the_engine_award() {
        // The gold parity contract: for every placement movegen emits, replaying its
        // path through a REAL engine (via plan-to-input) must award exactly the
        // T-spin classification `classify_t_spin` predicts for the placement piece.
        // This pins the whole chain — path recording, spin-state tracking, the
        // sticky flag, plan rendering, and the engine's own classifier — together.
        use crate::ai::plan::placement_to_inputs;
        use crate::engine::{Engine, EngineConfig, EngineEvent, EngineScoreAction};

        let fixtures: [&[(isize, isize)]; 4] = [
            &[],                                               // empty board: no spins anywhere
            &[(1, 0), (3, 0), (1, 2)],                         // the slide-or-spin slot above
            &[(5, 5), (4, 7), (3, 5), (3, 3)],                 // the engine's kick-5 fixture
            &[(1, 0), (3, 0), (1, 2), (5, 0), (7, 0), (5, 2)], // two slots
        ];

        for (i, cells) in fixtures.iter().enumerate() {
            let mut board = Board::new(10, 20);
            for &(x, y) in *cells {
                board.set(x, y, CellKind::Some(PieceType::I));
            }
            for piece_type in [PieceType::T, PieceType::L] {
                let start = spawn_piece(piece_type, 10, 20);
                for placement in generate(&board, &start) {
                    let predicted = crate::engine::classify_t_spin(&placement.piece, &board);

                    // Replay on a fresh engine carrying the same board + start pose.
                    let mut engine = Engine::new(EngineConfig::default(), 0);
                    for &(x, y) in *cells {
                        assert!(engine.set_cell(x, y, CellKind::Some(PieceType::I)));
                    }
                    engine.set_active(start.clone());
                    let mut awarded = None;
                    for frame in placement_to_inputs(&board, &start, &placement) {
                        for event in engine.step(frame) {
                            if let EngineEvent::ScoreAwarded {
                                action: EngineScoreAction::TSpin { kind, .. },
                                ..
                            } = event
                            {
                                awarded = Some(kind);
                            }
                        }
                    }
                    assert_eq!(
                        awarded,
                        predicted,
                        "fixture {i}, {piece_type:?} placement at {:?} rot {:?} path {:?}: \
                         engine awarded {awarded:?} but movegen predicted {predicted:?}",
                        placement.origin(),
                        placement.rotation(),
                        placement.path,
                    );
                }
            }
        }
    }
}
