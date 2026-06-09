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
//! via `BitBoard::lock_piece` (the bitboard mirror of `lock_and_clear`) so the board
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

use smallvec::SmallVec;

use crate::ai::movegen::Placement;
use crate::engine::{
    classify_t_spin, ActivePiece, BitBoard, Board, CellKind, EngineSnapshot, LockOutcome, Piece,
    PieceRotation, RotationDirection, TSpinKind,
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
    /// The playfield occupancy (incl. the hidden spawn buffer). A `Copy`, zero-alloc
    /// [`BitBoard`]: the search forks it per candidate placement, so bit-AND collision,
    /// bit-op locking, and identity `columns()` are the heart of the performance strike.
    pub board: BitBoard,
    /// The piece currently in play, at its current pose.
    pub active: ActivePiece,
    /// The hold slot, if occupied. Carried so the evaluator and a hold-aware
    /// search can read it. A multi-ply search must call [`commit_placement`] (not
    /// [`commit`]) when a [`Placement`] has `used_hold == true`, so the swap is
    /// reflected here.
    ///
    /// [`commit_placement`]: SearchState::commit_placement
    /// [`commit`]: SearchState::commit
    pub hold: Option<crate::engine::PieceType>,
    /// The revealed Next queue, front = next to spawn.
    pub queue: SmallVec<[crate::engine::PieceType; 16]>,
    /// The reconstructed 7-bag remainder the next *unknown* piece is drawn from.
    pub bag: BagState,
    /// Whether a Back-to-Back chain is currently active.
    pub b2b: bool,
    /// Length of the current combo chain: consecutive line-clearing placements so
    /// far (`0` after a clear-less lock or at the start). The index a search uses to
    /// value combo attack for the *next* clear. Tracked along the path like [`b2b`];
    /// a search reads the pre-placement value to score a clear's combo bonus.
    pub combo: u32,
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
        let board = BitBoard::from_board(&rebuild_board(snapshot));
        let active = rebuild_active(active_snapshot);

        let queue: SmallVec<[crate::engine::PieceType; 16]> =
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
            combo: snapshot.combo, // resume the real in-game combo, so the search can value continuing it
            board_width: config.board_width,
            visible_height: config.visible_height,
        })
    }

    /// Lock the active piece at its current pose and advance to the next piece.
    ///
    /// Locks via `BitBoard::lock_piece` (so the cleared rows and
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
        let piece = self.active.clone();
        let t_spin = classify_t_spin(&piece, &self.board);
        let outcome = self.board.lock_piece(&piece);
        self.update_b2b(&outcome, t_spin);
        self.update_combo(&outcome);
        if let Some(next) = self.deal_from_queue() {
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
        let piece = self.active.clone();
        let t_spin = classify_t_spin(&piece, &self.board);
        let outcome = self.board.lock_piece(&piece);
        self.update_b2b(&outcome, t_spin);
        self.update_combo(&outcome);
        self.spawn(next);
        outcome
    }

    /// Advance through a [`Placement`] from [`generate_with_hold`], honouring a hold
    /// swap and tracking Back-to-Back — the hold-aware analogue of [`commit`].
    ///
    /// [`commit`] and [`commit_with_next`] lock `self.active` and cannot model a
    /// swap, but [`generate_with_hold`] emits `used_hold` placements whose
    /// [`Placement::piece`] is the *swapped-in* piece at its resting pose. A
    /// multi-ply search transitions through those placements with this method.
    ///
    /// When `placement.used_hold`, the currently active piece is displaced into the
    /// hold slot. If hold was empty, that swap is funded by pulling the queue front
    /// (mirroring the engine's empty-hold rule, [`generate_with_hold`]'s
    /// `hold.or(queue_front)`); the swapped-in piece is the one already at its
    /// resting pose in `placement`, so it is **not** re-spawned or re-dealt.
    ///
    /// Regardless of the swap, the placement's piece is classified against the
    /// pre-lock board (engine order, matching [`Engine`](crate::engine::Engine)'s
    /// own lock path) and locked via `BitBoard::lock_piece`; the Back-to-Back flag
    /// is then transitioned and the *next queued* piece is spawned (the only
    /// [`BagState::deal`] this method performs).
    ///
    /// Returns the [`LockOutcome`] from the lock for the evaluator's reward half.
    ///
    /// [`generate_with_hold`]: crate::ai::movegen::generate_with_hold
    /// [`commit`]: SearchState::commit
    /// [`commit_with_next`]: SearchState::commit_with_next
    pub fn commit_placement(&mut self, placement: &Placement) -> LockOutcome {
        let outcome = self.apply_placement(placement);
        if let Some(next) = self.deal_from_queue() {
            self.spawn(next); // the ONLY deal — next ply's active
        }
        outcome
    }

    /// Like [`commit_placement`], but deals `next` as the new active piece instead of
    /// pulling from the revealed queue — the **hold-aware** analogue of
    /// [`commit_with_next`] (which locks `self.active` and cannot model a swap).
    ///
    /// This is the speculative-lookahead transition past the visible queue: it performs
    /// the same `used_hold` swap (including the empty-hold queue-funding rule) and the
    /// same pre-lock T-spin classify as [`commit_placement`] — they share
    /// [`apply_placement`](Self::apply_placement) — then spawns the supplied speculative
    /// `next` rather than the (exhausted) queue front. In the beam's speculation the
    /// queue is already empty *and* movegen only offers `used_hold` when hold is
    /// occupied, so the empty-hold funding pop never fires; sharing the transition keeps
    /// the swap rule in one place instead of re-open-coded at the call site.
    ///
    /// Returns the [`LockOutcome`] from the lock for the evaluator's reward half.
    ///
    /// [`commit_placement`]: SearchState::commit_placement
    /// [`commit_with_next`]: SearchState::commit_with_next
    pub fn commit_placement_with_next(
        &mut self,
        placement: &Placement,
        next: crate::engine::PieceType,
    ) -> LockOutcome {
        let outcome = self.apply_placement(placement);
        self.spawn(next); // the ONLY deal — the speculative next ply's active
        outcome
    }

    /// The hold-aware lock shared by [`commit_placement`](Self::commit_placement) and
    /// [`commit_placement_with_next`](Self::commit_placement_with_next): honour a
    /// `used_hold` swap (funding an empty hold from the queue front — the engine's
    /// empty-hold rule), classify the T-spin against the PRE-lock board (engine order),
    /// lock `placement.piece`, then transition the Back-to-Back and combo chains. Does
    /// **not** deal the next active piece — the caller supplies it (from the queue, or
    /// speculatively).
    fn apply_placement(&mut self, placement: &Placement) -> LockOutcome {
        if placement.used_hold {
            let displaced = self.active.piece_type();
            if self.hold.is_none() {
                // An empty hold pulls the queue front to fund the swap; the
                // swapped-in piece is already dealt (it is `placement.piece`), so
                // no extra deal happens here — only this queue advance.
                self.deal_from_queue();
            }
            self.hold = Some(displaced);
        }
        // Classify against the PRE-lock board (engine order), then lock
        // `placement.piece` — the swapped-in piece when `used_hold`, else the
        // current active at its resting pose. Locking the placement's piece (not
        // `self.active`) is what makes the swapped-in piece active without a deal.
        let t_spin = classify_t_spin(&placement.piece, &self.board);
        let outcome = self.board.lock_piece(&placement.piece);
        self.update_b2b(&outcome, t_spin);
        self.update_combo(&outcome);
        outcome
    }

    /// Pop the next revealed piece from the front of the queue (or `None` if the
    /// queue is exhausted). `SmallVec` has no `pop_front`; this is the empty-safe
    /// `remove(0)`. The shift is O(len) but len ≤ preview depth and allocation-free,
    /// so the per-child `SearchState` clone stays off the heap (the board is already
    /// `Copy`).
    fn deal_from_queue(&mut self) -> Option<crate::engine::PieceType> {
        if self.queue.is_empty() {
            None
        } else {
            Some(self.queue.remove(0))
        }
    }

    /// Transition the Back-to-Back flag for a freshly locked placement.
    ///
    /// A clear keeps the chain alive only when it is a "difficult" clear — any
    /// T-spin (Mini or Full) or a Tetris (four lines) — and breaks it on any other
    /// clear; a placement that clears no lines preserves the chain. This mirrors the
    /// engine's Back-to-Back rule and is kept in sync with the `b2b_eligible`
    /// categories in the `b2b_eligible` arms of `eval::compute_reward`.
    fn update_b2b(&mut self, outcome: &LockOutcome, t_spin: Option<TSpinKind>) {
        let lines = outcome.cleared_rows.len();
        if lines == 0 {
            return; // no clear: chain preserved
        }
        self.b2b = matches!(
            (t_spin, lines),
            (Some(TSpinKind::Full | TSpinKind::Mini), _) | (None, 4)
        );
    }

    /// Advance the combo chain for a freshly locked placement: a line clear extends
    /// it (`combo += 1`), a clear-less lock breaks it (`combo = 0`). A search reads
    /// the pre-placement [`combo`](Self::combo) to value the clear's combo attack.
    fn update_combo(&mut self, outcome: &LockOutcome) {
        self.combo = if outcome.cleared_rows.is_empty() {
            0
        } else {
            self.combo + 1
        };
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
        queue: impl IntoIterator<Item = crate::engine::PieceType>,
    ) -> Self {
        let board_width = board.width();
        let visible_height = board.height();
        Self {
            board: BitBoard::from_board(&board),
            active,
            hold,
            queue: queue.into_iter().collect(),
            bag: BagState::full(),
            b2b: false,
            combo: 0,
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
fn hard_drop<B: crate::engine::Occupancy>(active: &ActivePiece, board: &B) -> ActivePiece {
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
            // `BitBoard` tracks occupancy, not colour, so assert the cell is filled.
            assert!(
                state.board.occupied(cell.x, cell.y),
                "reconstructed board should occupy cell ({}, {})",
                cell.x,
                cell.y
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

        let next_up = state.queue.first().copied().unwrap();
        let queue_len_before = state.queue.len();
        // Hard-drop the active piece to a realistic resting pose, then commit it.
        state.active = hard_drop(&state.active, &state.board);
        let landed_cells = state.active.piece().cells();
        let landed_origin = state.active.origin();

        let outcome = state.commit();

        // The locked cells are now on the board (empty board → no line clear).
        assert!(outcome.cleared_rows.is_empty());
        for (cx, cy) in landed_cells {
            assert!(
                state
                    .board
                    .occupied(cx + landed_origin.0, cy + landed_origin.1),
                "locked cell should be occupied"
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
        assert!(state.board.is_empty());
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
        for pt in PieceType::all() {
            assert!(bag.contains(pt));
            bag.deal(pt);
        }
        // A full bag dealt out is empty; the next deal refills it.
        bag.deal(PieceType::I);
        assert!(!bag.contains(PieceType::I));
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

    // --- commit_placement (hold-aware transition, STEP 0) -----------------------

    /// Build a crafted, empty-board state with the given active/hold/queue, on a
    /// 10x20 board (no top margin) so placements are easy to reason about and no
    /// stray line clears perturb bag/b2b accounting.
    fn crafted_state(
        active: PieceType,
        hold: Option<PieceType>,
        queue: &[PieceType],
    ) -> SearchState {
        let board = Board::new(10, 20);
        let start = crate::ai::movegen::spawn_piece(active, 10, 20);
        SearchState::for_test(board, start, hold, queue.iter().copied())
    }

    /// Enumerate this state's hold-aware placements (the production seam a search
    /// uses), with the spawn-pose closure wired to the state's geometry.
    fn placements_of(state: &SearchState) -> Vec<Placement> {
        let (w, h) = (state.board.width(), state.board.height());
        crate::ai::movegen::generate_with_hold(
            &state.board,
            &state.active,
            state.hold,
            state.queue.first().copied(),
            |pt| crate::ai::movegen::spawn_piece(pt, w, h),
        )
    }

    /// The absolute occupied cells of a piece at its pose, sorted for comparison.
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

    #[test]
    fn commit_placement_without_hold_matches_commit() {
        // A non-hold placement must behave exactly like setting `active` to that
        // resting pose and calling the plain `commit` (which locks `self.active`).
        let state = crafted_state(PieceType::L, None, &[PieceType::S, PieceType::Z]);
        let placement = placements_of(&state)
            .into_iter()
            .find(|p| !p.used_hold)
            .expect("a no-hold placement exists on an empty board");

        // Parallel: plain commit after moving the active piece onto the pose.
        let mut via_commit = state.clone();
        via_commit.active = placement.piece.clone();
        let out_commit = via_commit.commit();

        // Under test: commit_placement on the same (no-hold) placement.
        let mut via_placement = state.clone();
        let out_placement = via_placement.commit_placement(&placement);

        assert_eq!(out_commit.cleared_rows, out_placement.cleared_rows);
        assert_eq!(
            via_commit.board.cell_coords(),
            via_placement.board.cell_coords()
        );
        assert_eq!(via_commit.active, via_placement.active);
        assert_eq!(via_commit.hold, via_placement.hold);
        assert_eq!(via_commit.queue, via_placement.queue);
        assert_eq!(via_commit.bag, via_placement.bag);
        assert_eq!(via_commit.b2b, via_placement.b2b);
    }

    #[test]
    fn commit_placement_with_occupied_hold_swaps() {
        // hold = Some(I), active = O; committing a used_hold placement swaps the
        // active O into hold and locks the swapped-in I at its resting pose.
        let state = crafted_state(PieceType::O, Some(PieceType::I), &[PieceType::T, PieceType::Z]);
        let queue_len_before = state.queue.len();
        let next_up = state.queue.first().copied().unwrap();
        let bag_before = state.bag;

        let placement = placements_of(&state)
            .into_iter()
            .find(|p| p.used_hold)
            .expect("a hold placement exists");
        assert_eq!(
            placement.piece_type(),
            PieceType::I,
            "the swapped-in piece is the held I"
        );
        let locked_cells = cells_of(&placement.piece);

        let mut s = state.clone();
        s.commit_placement(&placement);

        // The displaced active (O) is now held; the I's cells are on the board.
        assert_eq!(s.hold, Some(PieceType::O));
        for (x, y) in &locked_cells {
            assert!(s.board.occupied(*x, *y), "I cell ({x}, {y}) should be occupied");
        }
        // An occupied hold does NOT pull the queue: the swap consumed no queued
        // piece, so the only advance is the next piece becoming active.
        assert_eq!(s.queue.len(), queue_len_before - 1);
        assert_eq!(s.active.piece_type(), next_up);

        // The bag advanced by exactly one deal — the *next* queued piece's spawn —
        // not by the swapped-in piece (which was already dealt).
        let mut expected_bag = bag_before;
        expected_bag.deal(next_up);
        assert_eq!(s.bag, expected_bag);
    }

    #[test]
    fn commit_placement_with_empty_hold_pulls_queue() {
        // hold = None, queue = [A, B, C]; committing a used_hold placement (the
        // swapped-in A) parks the old active in hold and pulls the queue front to
        // fund the empty-hold swap, so B becomes the new active (A consumed by the
        // swap, B by the following spawn).
        let queue = [PieceType::J, PieceType::L, PieceType::S];
        let state = crafted_state(PieceType::T, None, &queue);
        let old_active = state.active.piece_type();
        let bag_before = state.bag;

        let placement = placements_of(&state)
            .into_iter()
            .find(|p| p.used_hold)
            .expect("a hold placement exists");
        assert_eq!(
            placement.piece_type(),
            queue[0],
            "an empty hold swaps in the queue front"
        );

        let mut s = state.clone();
        s.commit_placement(&placement);

        assert_eq!(s.hold, Some(old_active));
        // Queue advanced by TWO: the swap pulled A, the spawn pulled B → C active.
        assert_eq!(s.active.piece_type(), queue[1]);
        assert_eq!(
            s.queue.iter().copied().collect::<Vec<_>>(),
            vec![queue[2]]
        );

        // Bag advances for the spawn ONLY. The swapped-in A was already dealt (in
        // production it left the bag when it was revealed into the queue), so the
        // funding pop must NOT re-deal it — `spawn`/`deal` fires only for B, the
        // following queued piece. (STEP 0 invariant: never re-deal the swapped-in
        // piece.)
        let mut expected_bag = bag_before;
        expected_bag.deal(queue[1]);
        assert_eq!(s.bag, expected_bag);
    }

    #[test]
    fn commit_placement_locks_the_placement_pose_not_active() {
        // Regression for the verified gap: commit_placement must lock the
        // placement's (swapped-in) piece at its resting pose, never `self.active`.
        let state = crafted_state(PieceType::O, Some(PieceType::I), &[PieceType::T]);
        let placement = placements_of(&state)
            .into_iter()
            .find(|p| p.used_hold)
            .expect("a hold placement exists");
        let expected_cells = cells_of(&placement.piece);

        let mut s = state.clone();
        s.commit_placement(&placement);

        // Exactly the placement's I cells are filled — none of the active O's.
        let mut filled = s.board.cell_coords();
        filled.sort();
        // `filled == expected_cells` already proves *exactly* the I's cells are
        // occupied and none of the O's — the colour check is redundant on a `BitBoard`.
        assert_eq!(filled, expected_cells);
    }

    #[test]
    fn update_b2b_transitions() {
        // Drive the helper directly through each category. Start b2b=true so a
        // chain-breaking clear is observable, and b2b is preserved across no-clear.
        let mut s = crafted_state(PieceType::T, None, &[PieceType::I]);

        // A Tetris (no t-spin) keeps the chain.
        s.b2b = false;
        s.update_b2b(&lock_with_rows(&[0, 1, 2, 3]), None);
        assert!(s.b2b, "tetris sets b2b");

        // A single line clear (no t-spin) breaks it.
        s.update_b2b(&lock_with_rows(&[0]), None);
        assert!(!s.b2b, "single clear breaks b2b");

        // No clear preserves whatever the chain was.
        s.b2b = true;
        s.update_b2b(&lock_with_rows(&[]), None);
        assert!(s.b2b, "no clear preserves b2b");
        s.b2b = false;
        s.update_b2b(&lock_with_rows(&[]), None);
        assert!(!s.b2b, "no clear preserves a broken b2b");

        // A full T-spin double keeps the chain alive.
        s.b2b = false;
        s.update_b2b(&lock_with_rows(&[0, 1]), Some(TSpinKind::Full));
        assert!(s.b2b, "T-spin double sets b2b");
    }

    /// A minimal [`LockOutcome`] carrying only the cleared rows (the field
    /// `update_b2b` reads); the other fields are irrelevant to the b2b transition.
    fn lock_with_rows(rows: &[isize]) -> LockOutcome {
        LockOutcome {
            cells_locked: Vec::new(),
            cleared_rows: rows.to_vec(),
            top_y_after_lock: None,
        }
    }

    #[test]
    fn commit_placement_deals_bag_once() {
        // commit_placement performs exactly ONE bag deal per call — the trailing
        // `spawn` of the next queued piece — and never re-deals a swapped-in piece.
        // After a hold commit then a plain commit, the bag must reflect precisely
        // those two spawns.
        let queue = [
            PieceType::I,
            PieceType::O,
            PieceType::T,
            PieceType::S,
            PieceType::Z,
        ];
        let mut s = crafted_state(PieceType::L, None, &queue);
        // `for_test` starts the bag FULL (it does not pre-deal the active or the
        // revealed queue), so every legitimate deal below is observable from full.
        assert_eq!(s.bag, BagState::full());

        // 1) Hold commit (empty hold): locks the swapped-in queue[0]=I WITHOUT a
        //    deal (already-dealt invariant), then spawns queue[1]=O — one deal.
        let hold_placement = placements_of(&s)
            .into_iter()
            .find(|p| p.used_hold)
            .expect("a hold placement exists");
        assert_eq!(hold_placement.piece_type(), queue[0]);
        s.commit_placement(&hold_placement);

        // 2) Plain commit of the now-active O: spawns queue[2]=T — one deal.
        let plain_placement = placements_of(&s)
            .into_iter()
            .find(|p| !p.used_hold)
            .expect("a no-hold placement exists");
        s.commit_placement(&plain_placement);

        // Exactly two spawns dealt across the two commits: O then T. The starting
        // active L and the swapped-in I were never dealt by commit_placement.
        let expected = BagState::from_dealt([queue[1], queue[2]]);
        assert_eq!(s.bag, expected);
    }
}
