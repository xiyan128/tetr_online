//! Cheap, cloneable search state for the placement-search AI (AI3.1).
//!
//! [`Engine`](crate::engine::Engine) is intentionally **not** `Clone` — it owns
//! the seeded piece generator, the score machine, and per-tick timing that a
//! search must not fork or advance. So the AI carries its own lightweight,
//! cloneable mirror of just the state a placement search needs: the board, the
//! active piece, the hold slot, the revealed Next queue, the reconstructed 7-bag
//! remainder, and the Back-to-Back flag. A [`SearchState`] is built from an
//! [`EngineSnapshot`] via [`SearchState::from_snapshot`] and advanced one
//! placement at a time by [`SearchState::commit`], which locks the active piece
//! through the engine's own [`lock_and_clear`] primitive so the simulated board
//! can never disagree with the real rules.
//!
//! # Bag reconstruction
//!
//! The snapshot exposes the revealed Next queue but **not** the generator's
//! internal bag, so a hold-aware lookahead must reconstruct it. We mirror Cold
//! Clear's `bag: EnumSet<Piece>` with a small [`BagState`] bitset recording which
//! of the seven tetrominoes have *not yet* been dealt out of the current bag.
//! Walking the already-dealt pieces (the active piece, then the revealed queue)
//! front-to-back and refilling whenever the bag empties yields the exact set the
//! next unknown piece will be drawn from — which is what `commit` needs once it
//! runs past the revealed queue.

use std::collections::VecDeque;

use crate::engine::{
    lock_and_clear, ActivePiece, Board, CellKind, EngineSnapshot, LockOutcome, Piece,
    PieceRotation, RotationDirection,
};

/// The remainder of the current 7-bag: which tetrominoes have **not** yet been
/// dealt out of it.
///
/// Mirrors Cold Clear's `bag: EnumSet<Piece>`. Stored as a 7-bit mask (one bit
/// per [`PieceType`](crate::engine::PieceType), indexed by
/// [`PieceType::all`](crate::engine::PieceType::all) order). A bag that is empty
/// is *full* again on the next draw — the seven-bag invariant — so [`BagState`]
/// refills itself transparently in [`BagState::deal`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BagState {
    /// Bit `i` set ⇔ `PieceType::all()[i]` is still available in this bag.
    remaining: u8,
}

impl BagState {
    /// All seven pieces present (the start-of-bag state).
    const FULL_MASK: u8 = (1 << crate::engine::PieceType::LEN) - 1;

    /// A fresh, full bag (all seven pieces available).
    pub fn full() -> Self {
        Self {
            remaining: Self::FULL_MASK,
        }
    }

    fn bit(piece_type: crate::engine::PieceType) -> u8 {
        1 << Self::index_of(piece_type)
    }

    fn index_of(piece_type: crate::engine::PieceType) -> usize {
        crate::engine::PieceType::all()
            .iter()
            .position(|p| *p == piece_type)
            .expect("PieceType::all() contains every piece")
    }

    /// Whether `piece_type` is still available to be dealt from this bag.
    pub fn contains(self, piece_type: crate::engine::PieceType) -> bool {
        self.remaining & Self::bit(piece_type) != 0
    }

    /// How many pieces are still in the bag (`1..=7`; an empty mask reports `7`
    /// because the bag is refilled before the next draw, so this is never `0`).
    pub fn remaining_count(self) -> usize {
        if self.remaining == 0 {
            crate::engine::PieceType::LEN
        } else {
            self.remaining.count_ones() as usize
        }
    }

    /// Account for `piece_type` having been dealt out of the bag.
    ///
    /// Refills the bag first if it was empty, preserving the seven-bag invariant
    /// (each bag deals every piece exactly once before the next begins). This is
    /// the bookkeeping side of a deal; the *choice* of which piece to deal when
    /// it is unknown is the search's job.
    pub fn deal(&mut self, piece_type: crate::engine::PieceType) {
        if self.remaining == 0 {
            self.remaining = Self::FULL_MASK;
        }
        self.remaining &= !Self::bit(piece_type);
    }

    /// Reconstruct the bag remainder after `dealt` pieces have been handed out,
    /// in deal order, starting from a fresh bag.
    pub fn from_dealt(dealt: impl IntoIterator<Item = crate::engine::PieceType>) -> Self {
        let mut bag = Self::full();
        for piece_type in dealt {
            bag.deal(piece_type);
        }
        bag
    }
}

/// A cheap, cloneable snapshot of the state a placement search reads and forks.
///
/// Built from an [`EngineSnapshot`] with [`SearchState::from_snapshot`] and
/// advanced one placement at a time with [`SearchState::commit`]. Cloning is
/// shallow board+piece data (no engine, no RNG, no timers), so a search can fork
/// it freely.
#[derive(Clone)]
pub struct SearchState {
    /// The playfield, including the hidden spawn buffer (so locking near the top
    /// behaves identically to the engine).
    pub board: Board,
    /// The piece currently in play, at its current pose.
    pub active: ActivePiece,
    /// The hold slot, if occupied. Holding is not yet modelled by `commit`; it is
    /// carried so the evaluator and a hold-aware search can read it.
    pub hold: Option<crate::engine::PieceType>,
    /// The revealed Next queue, front = next to spawn.
    pub queue: VecDeque<crate::engine::PieceType>,
    /// The reconstructed 7-bag remainder the next *unknown* piece is drawn from.
    pub bag: BagState,
    /// Whether a Back-to-Back chain is currently active.
    pub b2b: bool,
    /// Board geometry captured from the snapshot config, needed to rebuild the
    /// board and to spawn freshly dealt pieces at the correct origin.
    board_width: usize,
    visible_height: usize,
}

impl SearchState {
    /// Build a search state from an engine snapshot.
    ///
    /// The board is rebuilt (margin included) from the snapshot's config and
    /// occupied cells; the active piece is reconstructed at its reported pose; and
    /// the 7-bag remainder is reconstructed from the already-dealt pieces (the
    /// active piece followed by the revealed queue). Returns `None` only when the
    /// snapshot has no active piece (e.g. before the first spawn or after game
    /// over), since a search has nothing to plan from in that case.
    pub fn from_snapshot(snapshot: &EngineSnapshot) -> Option<Self> {
        let active_snapshot = snapshot.active.as_ref()?;

        let config = &snapshot.config;
        let board = rebuild_board(snapshot);
        let active = rebuild_active(active_snapshot);

        let queue: VecDeque<crate::engine::PieceType> =
            snapshot.next_queue.iter().copied().collect();

        // The active piece was the most recently dealt piece, then the queue (in
        // order). Hold is *not* part of bag accounting: a held piece left the
        // dealt stream when it was put aside, and the bag already advanced past it
        // at deal time.
        let dealt = std::iter::once(active.piece_type()).chain(queue.iter().copied());
        let bag = BagState::from_dealt(dealt);

        Some(Self {
            board,
            active,
            hold: snapshot.hold,
            queue,
            bag,
            b2b: snapshot.back_to_back_active,
            board_width: config.board_width,
            visible_height: config.visible_height,
        })
    }

    /// Lock the active piece at its current pose and advance to the next piece.
    ///
    /// Locks through the engine's own [`lock_and_clear`] (so the cleared rows and
    /// resulting board match the real rules exactly), then deals the next piece:
    /// the front of the revealed queue becomes the new active piece (spawned at
    /// its guideline origin), and the bag is advanced for it. When the revealed
    /// queue runs dry, the search is expected to supply a speculative next piece
    /// via [`SearchState::commit_with_next`]; calling `commit` with an empty queue
    /// leaves `active` unchanged and only mutates the board, which a caller can
    /// detect by inspecting whether the queue was empty beforehand.
    ///
    /// Returns the [`LockOutcome`] from the lock for the evaluator's reward half.
    pub fn commit(&mut self) -> LockOutcome {
        let outcome = lock_and_clear(&self.active, &mut self.board);
        if let Some(next) = self.queue.pop_front() {
            self.spawn(next);
        }
        outcome
    }

    /// Like [`SearchState::commit`], but deals `next` as the new active piece
    /// instead of pulling from the revealed queue.
    ///
    /// This is the speculative-lookahead path: once a search exhausts the revealed
    /// Next queue it enumerates the bag's remaining pieces ([`SearchState::bag`])
    /// and commits each via this method to explore "what if the next piece is X".
    pub fn commit_with_next(&mut self, next: crate::engine::PieceType) -> LockOutcome {
        let outcome = lock_and_clear(&self.active, &mut self.board);
        self.spawn(next);
        outcome
    }

    /// Replace the active piece with a freshly dealt `piece_type` at its spawn
    /// origin and advance the bag for it.
    fn spawn(&mut self, piece_type: crate::engine::PieceType) {
        let origin = Piece::from(piece_type).spawn_coords(self.board_width, self.visible_height);
        self.active = ActivePiece::new(piece_type, origin);
        self.bag.deal(piece_type);
    }

    /// Build a search state directly from parts, for crafted-board unit tests in the
    /// AI crate that need a specific position without spinning up an [`Engine`].
    ///
    /// The board geometry is taken from `board`, the bag starts full, and B2B is
    /// off; pass the `hold` slot and revealed `queue` explicitly. Test-only — the
    /// production path is [`SearchState::from_snapshot`].
    #[cfg(test)]
    pub(crate) fn for_test(
        board: Board,
        active: ActivePiece,
        hold: Option<crate::engine::PieceType>,
        queue: VecDeque<crate::engine::PieceType>,
    ) -> Self {
        let board_width = board.width();
        let visible_height = board.height();
        Self {
            board,
            active,
            hold,
            queue,
            bag: BagState::full(),
            b2b: false,
            board_width,
            visible_height,
        }
    }
}

/// Rebuild the playfield (margin included) from the snapshot's occupied cells.
fn rebuild_board(snapshot: &EngineSnapshot) -> Board {
    let config = &snapshot.config;
    let mut board = Board::with_top_margin(
        config.board_width,
        config.visible_height,
        config.buffer_height,
    );
    for cell in &snapshot.board_cells {
        board.set(cell.x, cell.y, CellKind::Some(cell.piece_type));
    }
    board
}

/// Reconstruct an [`ActivePiece`] at the pose reported by the snapshot.
///
/// [`ActivePiece::new`] always spawns at rotation `R0`; we apply the snapshot's
/// rotation with [`ActivePiece::rotate_to`] so the search starts from the real
/// pose. The kick / direction bookkeeping passed here is synthetic (the search
/// re-derives reachable poses from scratch and ignores lock-down history), so the
/// only field that matters is the resulting rotation.
fn rebuild_active(snapshot: &crate::engine::ActivePieceSnapshot) -> ActivePiece {
    let mut active = ActivePiece::new(snapshot.piece_type, snapshot.origin);
    if snapshot.rotation != PieceRotation::R0 {
        active.rotate_to(
            snapshot.rotation,
            snapshot.origin,
            RotationDirection::Clockwise,
            1,
            false,
        );
    }
    active
}

/// The neutral spawn origin a freshly dealt piece would take, mirroring the
/// engine's spawn coordinates. Exposed for tests that build placements by hand.
#[cfg(test)]
fn spawn_origin(
    piece_type: crate::engine::PieceType,
    width: usize,
    visible_height: usize,
) -> (isize, isize) {
    Piece::from(piece_type).spawn_coords(width, visible_height)
}

/// Drop `active` straight down on `board` until it rests, returning the landed
/// piece. A movement-free helper used by the tests to build realistic
/// placements; the real movegen (AI3.3) will supply the lateral path.
#[cfg(test)]
fn hard_drop(active: &ActivePiece, board: &Board) -> ActivePiece {
    let mut landed = active.clone();
    while let Some(origin) =
        landed
            .piece()
            .try_move(board, landed.origin(), crate::engine::MoveDirection::Down)
    {
        landed.move_to(origin, crate::engine::PieceAction::HardDrop);
    }
    landed
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{Engine, EngineConfig, InputFrame, PieceType};

    /// A snapshot from a fresh engine that has spawned its first piece.
    fn spawned_snapshot(seed: u64) -> EngineSnapshot {
        let mut engine = Engine::new(EngineConfig::default(), seed);
        engine.step(InputFrame::default()); // spawn the first piece
        engine.snapshot()
    }

    #[test]
    fn from_snapshot_maps_key_fields() {
        let snapshot = spawned_snapshot(42);
        let state = SearchState::from_snapshot(&snapshot).expect("active piece present");

        let active = snapshot.active.as_ref().unwrap();
        assert_eq!(state.active.piece_type(), active.piece_type);
        assert_eq!(state.active.rotation(), active.rotation);
        assert_eq!(state.active.origin(), active.origin);
        assert_eq!(state.hold, snapshot.hold);
        assert_eq!(
            state.queue.iter().copied().collect::<Vec<_>>(),
            snapshot.next_queue
        );
        assert_eq!(state.b2b, snapshot.back_to_back_active);
        // The board reconstruction reproduces exactly the occupied cells.
        for cell in &snapshot.board_cells {
            assert_eq!(
                state.board.get_cell_kind(cell.x, cell.y),
                CellKind::Some(cell.piece_type)
            );
        }
    }

    #[test]
    fn from_snapshot_returns_none_without_active_piece() {
        // A brand-new engine has a queue but no active piece yet.
        let engine = Engine::new(EngineConfig::default(), 7);
        let snapshot = engine.snapshot();
        assert!(snapshot.active.is_none());
        assert!(SearchState::from_snapshot(&snapshot).is_none());
    }

    #[test]
    fn from_snapshot_preserves_rotation() {
        let mut engine = Engine::new(EngineConfig::default(), 1);
        engine.step(InputFrame::default());
        // Rotate the active piece so the snapshot is not at R0 (skip if the piece
        // is an O, which never leaves R0).
        let pre = engine.snapshot().active.unwrap();
        if pre.piece_type != PieceType::O {
            engine.step(InputFrame {
                rotate_clockwise: true,
                ..InputFrame::default()
            });
            let snapshot = engine.snapshot();
            let active = snapshot.active.as_ref().unwrap();
            assert_ne!(active.rotation, PieceRotation::R0);

            let state = SearchState::from_snapshot(&snapshot).unwrap();
            assert_eq!(state.active.rotation(), active.rotation);
            assert_eq!(state.active.origin(), active.origin);
        }
    }

    #[test]
    fn commit_locks_active_to_board_and_advances_queue() {
        let snapshot = spawned_snapshot(99);
        let mut state = SearchState::from_snapshot(&snapshot).unwrap();

        let next_up = state.queue.front().copied().unwrap();
        let queue_len_before = state.queue.len();
        // Hard-drop the active piece to a realistic resting pose, then commit it.
        state.active = hard_drop(&state.active, &state.board);
        let landed_cells = state.active.piece().cells();
        let landed_origin = state.active.origin();
        let placed_type = state.active.piece_type();

        let outcome = state.commit();

        // The locked cells are now on the board (empty board → no line clear).
        assert!(outcome.cleared_rows.is_empty());
        for (cx, cy) in landed_cells {
            assert_eq!(
                state
                    .board
                    .get_cell_kind(cx + landed_origin.0, cy + landed_origin.1),
                CellKind::Some(placed_type)
            );
        }
        // The queue advanced: the old front is now the active piece.
        assert_eq!(state.active.piece_type(), next_up);
        assert_eq!(state.queue.len(), queue_len_before - 1);
        // The new active piece spawned at its guideline origin.
        assert_eq!(
            state.active.origin(),
            spawn_origin(
                next_up,
                snapshot.config.board_width,
                snapshot.config.visible_height
            )
        );
    }

    #[test]
    fn commit_clears_a_full_row() {
        // Build a 4-wide engine, fill row 0 except one column, and drop an I into
        // the gap to force a single-line clear through `commit`.
        let config = EngineConfig {
            board_width: 4,
            ..EngineConfig::default()
        };
        let mut engine = Engine::new(config, 0);
        for x in 1..4 {
            engine.set_cell(x, 0, CellKind::Some(PieceType::O));
        }
        engine.step(InputFrame::default()); // spawn so the snapshot has an active piece
        let snapshot = engine.snapshot();
        let mut state = SearchState::from_snapshot(&snapshot).unwrap();

        // Place a horizontal I across row 0 (origin (0, -2) puts its cells on y=0).
        state.active = ActivePiece::new(PieceType::I, (0, -2));
        let outcome = state.commit();

        assert_eq!(outcome.cleared_rows, vec![0]);
        // Row 0 cleared → the board is empty again.
        assert!(state.board.cells().is_empty());
    }

    #[test]
    fn commit_with_next_deals_the_supplied_piece() {
        let snapshot = spawned_snapshot(5);
        let mut state = SearchState::from_snapshot(&snapshot).unwrap();
        let queue_before: Vec<_> = state.queue.iter().copied().collect();

        state.active = hard_drop(&state.active, &state.board);
        state.commit_with_next(PieceType::T);

        // The speculative piece is now active; the revealed queue is untouched.
        assert_eq!(state.active.piece_type(), PieceType::T);
        assert_eq!(
            state.queue.iter().copied().collect::<Vec<_>>(),
            queue_before
        );
    }

    #[test]
    fn bag_reconstructs_from_revealed_queue() {
        let snapshot = spawned_snapshot(123);
        let state = SearchState::from_snapshot(&snapshot).unwrap();

        // Re-derive the expected bag from active + queue and compare.
        let dealt = std::iter::once(snapshot.active.as_ref().unwrap().piece_type)
            .chain(snapshot.next_queue.iter().copied());
        let expected = BagState::from_dealt(dealt);
        assert_eq!(state.bag, expected);

        // Pieces already dealt this bag are absent; the seven-bag invariant means
        // the union of dealt + remaining over a whole bag is all seven types.
        let dealt_this_bag: Vec<_> = std::iter::once(snapshot.active.as_ref().unwrap().piece_type)
            .chain(snapshot.next_queue.iter().copied())
            .take(7)
            .collect();
        for pt in &dealt_this_bag {
            assert!(!state.bag.contains(*pt));
        }
    }

    #[test]
    fn bag_deal_refills_when_empty_and_tracks_membership() {
        let mut bag = BagState::full();
        assert_eq!(bag.remaining_count(), 7);
        for pt in PieceType::all() {
            assert!(bag.contains(pt));
            bag.deal(pt);
        }
        // A full bag dealt out is empty; the next deal refills it.
        assert_eq!(bag.remaining_count(), 7); // reports a fresh full bag
        bag.deal(PieceType::I);
        assert!(!bag.contains(PieceType::I));
        assert_eq!(bag.remaining_count(), 6);
    }

    #[test]
    fn determinism_same_snapshot_yields_identical_state() {
        let snapshot = spawned_snapshot(2024);
        let a = SearchState::from_snapshot(&snapshot).unwrap();
        let b = SearchState::from_snapshot(&snapshot).unwrap();

        assert_eq!(a.active, b.active);
        assert_eq!(a.hold, b.hold);
        assert_eq!(a.queue, b.queue);
        assert_eq!(a.bag, b.bag);
        assert_eq!(a.b2b, b.b2b);
        assert_eq!(a.board.cell_coords(), b.board.cell_coords());

        // Committing the same sequence of placements stays in lockstep.
        let mut a = a;
        let mut b = b;
        for _ in 0..3 {
            a.active = hard_drop(&a.active, &a.board);
            b.active = hard_drop(&b.active, &b.board);
            assert_eq!(a.commit(), b.commit());
            assert_eq!(a.active, b.active);
            assert_eq!(a.board.cell_coords(), b.board.cell_coords());
        }
    }
}
