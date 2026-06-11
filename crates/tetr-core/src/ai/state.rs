//! Cheap, cloneable search state for the placement-search AI.
//!
//! [`Engine`](crate::engine::Engine) is intentionally **not** `Clone` — it owns
//! the seeded piece generator, the score machine, and per-tick timing that a
//! search must not fork or advance. So the AI carries its own lightweight,
//! cloneable mirror of just the state a placement search needs: the board, the
//! active piece, the hold slot, the revealed Next queue, the engine-exported
//! 7-bag remainder, and the Back-to-Back flag. A [`SearchState`] is built from an
//! [`EngineSnapshot`] via [`SearchState::from_snapshot`] and advanced one
//! placement at a time by [`SearchState::commit`], which locks the active piece
//! via `BitBoard::lock_piece` (the bitboard mirror of `lock_and_clear`) so the board
//! can never disagree with the real rules.
//!
//! # Bag tracking
//!
//! The snapshot exports the generator's own current-bag remainder
//! ([`EngineSnapshot::bag_remainder`]) — the **exact** set the next piece beyond
//! the revealed queue draws from — so the search starts from the truth rather
//! than a reconstruction. (A reconstruction from the active+queue window alone
//! is impossible: the window straddles bag boundaries, so walking it from a
//! fresh bag under-claims whenever the active piece is not its bag's first
//! piece, and at `preview_count <= 4` even over-claims pieces the bag already
//! dealt. Pinned by `bag_matches_the_generator_truth_at_any_preview`.)
//!
//! We mirror Cold Clear's `bag: EnumSet<Piece>` with a small [`BagState`] bitset.
//! The accounting convention along a search path: pieces spawned **from the
//! revealed queue** never touch the bag (the generator already dealt them — the
//! exported remainder accounts for them, and a queue piece may even belong to a
//! *previous* bag whose value legitimately remains available in the current
//! one). Only **speculative** deals — `commit_with_next` /
//! `commit_placement_with_next`, past the queue — consume the bag, refilling on
//! the seven-bag boundary like the real generator.

use smallvec::SmallVec;

use crate::ai::movegen::Placement;
use crate::engine::garbage::{self, BatchQueue};
use crate::engine::{
    ActivePiece, BitBoard, Board, CellKind, EngineScoreAction, EngineSnapshot, LockOutcome, Piece,
    TSpinKind, attack_lines, breaks_back_to_back, classify_t_spin, is_lock_out,
    qualifies_for_back_to_back,
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

    /// Whether the **next deal** can produce `piece_type`.
    ///
    /// An exhausted bag (empty remainder — a seven-bag boundary) refills before
    /// the next draw, so at a boundary *every* piece is possible: this returns
    /// `true` for all seven then, mirroring [`deal`](Self::deal)'s lazy refill.
    /// Without this, speculation at a bag boundary would enumerate nothing and a
    /// search line would silently dead-end every seventh piece.
    pub fn contains(self, piece_type: crate::engine::PieceType) -> bool {
        self.remaining == 0 || self.remaining & Self::bit(piece_type) != 0
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

    /// A bag whose remainder is exactly `pieces` — the constructor for the
    /// engine-exported [`EngineSnapshot::bag_remainder`]. An empty iterator is a
    /// bag boundary; [`contains`](Self::contains)/[`deal`](Self::deal) treat it
    /// as refilling on the next draw.
    ///
    /// [`EngineSnapshot::bag_remainder`]: crate::engine::EngineSnapshot::bag_remainder
    pub fn from_pieces(pieces: impl IntoIterator<Item = crate::engine::PieceType>) -> Self {
        let mut remaining = 0u8;
        for piece_type in pieces {
            remaining |= Self::bit(piece_type);
        }
        Self { remaining }
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
    /// The evaluator seam reads it only as a [`ColumnView`](crate::engine::ColumnView)
    /// (via [`BitBoard::view`]); the search internals and the criterion benches drive the
    /// concrete board directly.
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
    /// The engine-exported 7-bag remainder the next *unknown* piece (beyond the
    /// revealed queue) is drawn from; empty = a bag boundary (next draw refills).
    pub bag: BagState,
    /// Whether a Back-to-Back chain is currently active.
    pub b2b: bool,
    /// Length of the current combo chain: consecutive line-clearing placements so
    /// far (`0` after a clear-less lock or at the start). The index a search uses to
    /// value combo attack for the *next* clear. Tracked along the path like `b2b`;
    /// a search reads the pre-placement value to score a clear's combo bonus.
    pub combo: u32,
    /// The engine's game ended on this path — a dying lock (lock-out), an
    /// overflowing garbage rise, or a blocked spawn. The planners treat a dead
    /// state as a terminal leaf scored at the
    /// search's `DEATH_SCORE` and never expand it:
    /// without this, a death's truncated board can evaluate BETTER than a
    /// cramped survival, and the search walks futures the engine forbids.
    pub dead: bool,
    /// The pending-garbage queue against this player (oldest batch first),
    /// mirrored from the snapshot so the search models cancellation and rising
    /// exactly — see `transition_garbage`. Inline
    /// storage: forking a child never allocates for the common 0-4 batches.
    pub pending: BatchQueue,
    /// The per-lock rising cap, captured from the snapshot config.
    garbage_cap: u32,
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

        // The engine exports its generator's current-bag remainder directly —
        // already net of every dealt piece (active, queue, and any held piece), so
        // no reconstruction or hold special-casing is needed here.
        let bag = BagState::from_pieces(snapshot.bag_remainder.iter().copied());

        Some(Self {
            board,
            active,
            hold: snapshot.hold,
            queue,
            bag,
            b2b: snapshot.back_to_back_active,
            combo: snapshot.combo, // resume the real in-game combo, so the search can value continuing it
            dead: false,           // a snapshot with an active piece is a live game
            pending: snapshot.pending_garbage.iter().copied().collect(),
            garbage_cap: config.garbage_cap,
            board_width: config.board_width,
            visible_height: config.visible_height,
        })
    }

    /// Lock the active piece at its current pose and advance to the next piece.
    ///
    /// Locks via `BitBoard::lock_piece` (so the cleared rows and
    /// resulting board match the real rules exactly), then deals the next piece:
    /// the front of the revealed queue becomes the new active piece (spawned at
    /// its guideline origin); the bag is untouched (queue pieces are already
    /// accounted in the exported remainder — see the module docs). When the revealed
    /// queue runs dry, the search is expected to supply a speculative next piece
    /// via [`SearchState::commit_with_next`]; calling `commit` with an empty queue
    /// leaves `active` unchanged and only mutates the board, which a caller can
    /// detect by inspecting whether the queue was empty beforehand.
    ///
    /// Returns the [`LockOutcome`] from the lock for the evaluator's reward half.
    pub fn commit(&mut self) -> LockOutcome {
        let outcome = self.lock_active();
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
        let outcome = self.lock_active();
        self.bag.deal(next); // a speculative draw consumes the bag (queue spawns don't)
        self.spawn(next);
        outcome
    }

    /// Classify-then-lock `self.active` at its current pose and transition the B2B /
    /// combo chains — the shared core of [`commit`](Self::commit) and
    /// [`commit_with_next`](Self::commit_with_next), which differ only in where the
    /// next active piece comes from (the queue front vs a speculative deal).
    fn lock_active(&mut self) -> LockOutcome {
        let piece = self.active.clone();
        self.lock_and_transition(&piece)
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
    /// is then transitioned and the *next queued* piece is spawned (no
    /// [`BagState::deal`] — queue pieces are already accounted in the exported
    /// remainder; only speculative commits consume the bag).
    ///
    /// Returns the [`LockOutcome`] from the lock for the evaluator's reward half.
    ///
    /// [`generate_with_hold`]: crate::ai::movegen::generate_with_hold
    /// [`commit`]: SearchState::commit
    /// [`commit_with_next`]: SearchState::commit_with_next
    pub fn commit_placement(&mut self, placement: &Placement) -> LockOutcome {
        let outcome = self.apply_placement(placement);
        if let Some(next) = self.deal_from_queue() {
            self.spawn(next); // queue-sourced: already accounted in the bag
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
    /// `apply_placement` — then spawns the supplied speculative
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
        self.bag.deal(next); // a speculative draw consumes the bag (queue spawns don't)
        self.spawn(next);
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
        self.lock_and_transition(&placement.piece)
    }

    /// The engine-order lock shared by every commit path: classify the T-spin
    /// and the lock-out against the PRE-lock state, lock, mirror the garbage
    /// transition, then advance the B2B / combo chains. Exactly
    /// `Engine::lock_active_piece`'s order, so the search's imagined future and
    /// the engine's real one cannot drift.
    fn lock_and_transition(&mut self, piece: &ActivePiece) -> LockOutcome {
        let t_spin = classify_t_spin(piece, &self.board);
        let lock_out = is_lock_out(piece.piece(), piece.origin(), self.visible_height);
        let outcome = self.board.lock_piece(piece);
        if lock_out {
            // A dying lock moves no garbage (the engine's rule: death takes
            // priority over offense) and ends the branch.
            self.dead = true;
        } else if !self.pending.is_empty() {
            // Empty queue: both transition branches are no-ops — skipping them
            // keeps the solo / no-pressure hot path free of garbage overhead.
            self.transition_garbage(&outcome, t_spin);
        }
        self.update_b2b(&outcome, t_spin);
        self.update_combo(&outcome);
        outcome
    }

    /// Mirror the engine's lock-path garbage rules, via the SAME shared rule
    /// functions (`engine::garbage::{cancel, rise}`) the engine itself calls:
    /// a clear's attack (same award action, same pre-update B2B/combo inputs,
    /// same post-lock perfect-clear check) cancels pending oldest-first; a
    /// clear-less lock lets pending rise onto the board up to the per-lock cap,
    /// with the hole columns the snapshot exported. An overflowing rise tops
    /// the real game out; the search just sees the (terrible) resulting board.
    fn transition_garbage(&mut self, outcome: &LockOutcome, t_spin: Option<TSpinKind>) {
        let lines = outcome.cleared_rows.len();
        if lines > 0 {
            let action = EngineScoreAction::from_lock_result(t_spin, lines);
            let b2b_bonus = qualifies_for_back_to_back(t_spin, lines) && self.b2b;
            let attack = attack_lines(action, b2b_bonus, self.combo, self.board.is_empty());
            garbage::cancel(&mut self.pending, attack);
        } else {
            let mut overflow = false;
            for batch in garbage::rise(&mut self.pending, self.garbage_cap) {
                overflow |= self
                    .board
                    .insert_garbage_lines(batch.lines as usize, batch.hole_col);
            }
            if overflow {
                // The engine latches a BlockOut on this same verdict: the rise
                // forced the stack past the ceiling, the branch is dead.
                self.dead = true;
            }
        }
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
    /// Delegates to the engine's own predicates — the exact transition
    /// `ScoreState::lock_result` performs — so the search's chain can never drift
    /// from what the engine will do when the move is actually played: a qualifying
    /// clear (a Tetris, or any T-spin line clear, Mini included) sets the chain, a
    /// plain 1-3 line clear breaks it, and anything else (a no-clear lock or a
    /// zero-line spin) preserves it.
    fn update_b2b(&mut self, outcome: &LockOutcome, t_spin: Option<TSpinKind>) {
        let lines = outcome.cleared_rows.len();
        if qualifies_for_back_to_back(t_spin, lines) {
            self.b2b = true;
        } else if breaks_back_to_back(t_spin, lines) {
            self.b2b = false;
        }
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

    /// Replace the active piece with `piece_type` at its spawn origin.
    ///
    /// Deliberately does **not** touch the bag: queue-sourced spawns are already
    /// accounted for by the engine-exported remainder (see the module's bag
    /// convention), and the speculative commit paths deal the bag explicitly
    /// before spawning.
    fn spawn(&mut self, piece_type: crate::engine::PieceType) {
        let piece = Piece::from(piece_type);
        let origin = piece.spawn_coords(self.board_width, self.visible_height);
        if piece.collide_with(&self.board, origin) {
            // The engine's spawn block-out (`is_block_out`): the next piece has
            // nowhere to appear — reachable in-search once garbage can rise.
            self.dead = true;
        }
        self.active = ActivePiece::new(piece_type, origin);
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
            dead: false,
            pending: BatchQueue::new(),
            garbage_cap: 8, // the engine default; garbage tests inject their own pending
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
        crate::engine::BUFFER_HEIGHT,
    );
    for cell in &snapshot.board_cells {
        let kind = if cell.garbage {
            CellKind::Garbage
        } else {
            CellKind::Some(cell.piece_type)
        };
        board.set(cell.x, cell.y, kind);
    }
    board
}

/// Reconstruct an [`ActivePiece`] at the pose reported by the snapshot, with
/// **spawn-fresh history** ([`ActivePiece::at_pose`]).
///
/// The snapshot does not carry the piece's action/kick history, so the engine's
/// true T-spin state for an in-place lock is unknowable here. Spawn-fresh history
/// is the conservative reconstruction: it never classifies a spin the engine might
/// not award. (Movegen re-derives placements — and their spin states — from
/// scratch along explicit paths, so search placements are unaffected.)
fn rebuild_active(snapshot: &crate::engine::ActivePieceSnapshot) -> ActivePiece {
    ActivePiece::at_pose(snapshot.piece_type, snapshot.origin, snapshot.rotation)
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
/// placements; the real movegen supplies the lateral path.
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
    use crate::engine::{Engine, EngineConfig, InputFrame, PieceRotation, PieceType};

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

        // A speculative deal consumes the bag (unlike queue-sourced spawns). Use a
        // crafted full bag so the membership flip is deterministic.
        let mut s = crafted_state(PieceType::L, None, &[PieceType::O]);
        assert!(s.bag.contains(PieceType::T));
        s.active = hard_drop(&s.active, &s.board);
        s.commit_with_next(PieceType::T);
        assert!(
            !s.bag.contains(PieceType::T),
            "commit_with_next must deal the speculative piece out of the bag"
        );
    }

    #[test]
    fn bag_comes_from_the_engine_exported_remainder() {
        // from_snapshot adopts the engine's exported remainder verbatim: bag
        // membership must equal `snapshot.bag_remainder` membership. (At 6 pieces
        // consumed the remainder is mid-bag and non-empty, so plain membership —
        // not the boundary draw-set rule — is what's exercised.)
        let snapshot = spawned_snapshot(123);
        assert!(!snapshot.bag_remainder.is_empty(), "mid-bag fixture");
        let state = SearchState::from_snapshot(&snapshot).unwrap();

        for pt in PieceType::all() {
            assert_eq!(
                state.bag.contains(pt),
                snapshot.bag_remainder.contains(&pt),
                "bag membership for {pt:?} must mirror the exported remainder"
            );
        }
    }

    #[test]
    fn bag_deal_refills_when_empty_and_tracks_membership() {
        let mut bag = BagState::full();
        for pt in PieceType::all() {
            assert!(bag.contains(pt));
            bag.deal(pt);
        }
        // A fully dealt bag is a boundary: the next draw refills, so EVERY piece
        // is a possible next deal (the draw-set semantics speculation relies on).
        for pt in PieceType::all() {
            assert!(bag.contains(pt), "at a boundary every piece is drawable");
        }
        // The next deal refills then consumes the dealt piece.
        bag.deal(PieceType::I);
        assert!(!bag.contains(PieceType::I));
        assert!(bag.contains(PieceType::T));
    }

    #[test]
    fn bag_matches_the_generator_truth_at_any_preview() {
        // The exactness contract the engine-exported remainder buys: at EVERY
        // position — any preview length, across bag boundaries, through a hold —
        // the search bag's draw-set equals the true bag (the complement of what
        // the current bag has dealt, derived from a same-seed generator replay).
        // A window-style reconstruction (rebuilding the bag from the visible
        // preview alone) is wrong in ~78% of positions at preview 5 and unsoundly
        // over-claims dealt pieces at preview <= 4 — hence the replay derivation.
        for preview_count in [5usize, 2] {
            for seed in [0u64, 7, 42, 12345] {
                let stream: Vec<PieceType> = crate::engine::PieceGenerator::with_seed(seed)
                    .take(64)
                    .collect();
                // Tall field so naive center hard-drops never top out (bag
                // accounting is board-independent).
                let config = EngineConfig {
                    preview_count,
                    visible_height: 40,
                    ..EngineConfig::default()
                };
                let mut engine = Engine::new(config, seed);
                engine.step(InputFrame::default()); // spawn the first piece
                let mut consumed = 1 + preview_count; // active + revealed queue

                for k in 0..16 {
                    if k == 4 {
                        // An empty-hold swap consumes one extra generator deal.
                        engine.step(InputFrame {
                            hold: true,
                            ..InputFrame::default()
                        });
                        consumed += 1;
                    }
                    let state = SearchState::from_snapshot(&engine.snapshot()).unwrap();
                    let dealt_this_bag = &stream[(consumed / 7) * 7..consumed];
                    for pt in PieceType::all() {
                        assert_eq!(
                            state.bag.contains(pt),
                            !dealt_this_bag.contains(&pt),
                            "preview {preview_count}, seed {seed}, piece {k}: \
                             bag draw-set diverged from the deal-stream truth for {pt:?}"
                        );
                    }
                    engine.step(InputFrame {
                        hard_drop: true,
                        ..InputFrame::default()
                    });
                    consumed += 1;
                }
            }
        }
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
        let state = crafted_state(
            PieceType::O,
            Some(PieceType::I),
            &[PieceType::T, PieceType::Z],
        );
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
            assert!(
                s.board.occupied(*x, *y),
                "I cell ({x}, {y}) should be occupied"
            );
        }
        // An occupied hold does NOT pull the queue: the swap consumed no queued
        // piece, so the only advance is the next piece becoming active.
        assert_eq!(s.queue.len(), queue_len_before - 1);
        assert_eq!(s.active.piece_type(), next_up);

        // The bag is untouched: every piece involved (the swapped-in I, the next
        // queued spawn) was already dealt by the generator and accounted in the
        // exported remainder — only SPECULATIVE deals consume the search's bag.
        assert_eq!(s.bag, bag_before);
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
        assert_eq!(s.queue.iter().copied().collect::<Vec<_>>(), vec![queue[2]]);

        // The bag is untouched: the swapped-in A and the spawned B are both
        // queue-sourced (the generator dealt them long ago; the exported remainder
        // already accounts for them). Re-dealing either would corrupt the bag —
        // queue pieces can even belong to a previous bag whose values are
        // legitimately still available in the current one.
        assert_eq!(s.bag, bag_before);
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

        // Mini T-spin LINE clears qualify — single and double alike, now that the
        // Mini-Double row is unified across the rule tables (score 400, 4 goal
        // units, attack 1, B2B-qualifying).
        s.b2b = false;
        s.update_b2b(&lock_with_rows(&[0]), Some(TSpinKind::Mini));
        assert!(s.b2b, "mini single sets b2b");
        s.b2b = false;
        s.update_b2b(&lock_with_rows(&[0, 1]), Some(TSpinKind::Mini));
        assert!(s.b2b, "mini double sets b2b");

        // A zero-line mini neither starts nor breaks a chain.
        s.b2b = false;
        s.update_b2b(&lock_with_rows(&[]), Some(TSpinKind::Mini));
        assert!(!s.b2b, "zero-line mini does not start a chain");
        s.b2b = true;
        s.update_b2b(&lock_with_rows(&[]), Some(TSpinKind::Mini));
        assert!(s.b2b, "zero-line mini preserves an active chain");
    }

    #[test]
    fn update_combo_transitions() {
        // The mirror of `update_b2b_transitions` for the combo chain the search feeds
        // to the evaluator: a clear-less lock holds/resets combo to 0, consecutive
        // clears escalate it, and a clear-less lock breaks it back to 0.
        let mut s = crafted_state(PieceType::T, None, &[PieceType::I]);
        assert_eq!(s.combo, 0, "a fresh state has no combo");

        s.update_combo(&lock_with_rows(&[]));
        assert_eq!(s.combo, 0, "a clear-less lock keeps combo at 0");

        s.update_combo(&lock_with_rows(&[0]));
        assert_eq!(s.combo, 1, "the first clear advances combo to 1");
        s.update_combo(&lock_with_rows(&[0, 1]));
        assert_eq!(s.combo, 2, "a consecutive clear advances combo to 2");

        s.update_combo(&lock_with_rows(&[]));
        assert_eq!(s.combo, 0, "a clear-less lock breaks the combo back to 0");

        s.update_combo(&lock_with_rows(&[0]));
        assert_eq!(s.combo, 1, "a fresh clear restarts the combo at 1");
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
    fn only_speculative_commits_consume_the_bag() {
        // The bag convention: queue-sourced transitions (plain commits, hold
        // commits, the empty-hold funding pop) never touch the bag — the engine's
        // exported remainder already accounts for every revealed piece. Only a
        // speculative deal (commit_placement_with_next / commit_with_next), which
        // models drawing a genuinely unknown piece, consumes it.
        let queue = [PieceType::I, PieceType::O];
        let mut s = crafted_state(PieceType::L, None, &queue);
        assert_eq!(s.bag, BagState::full());

        // 1) Hold commit (empty hold): swap + funding pop + spawn — no bag change.
        let hold_placement = placements_of(&s)
            .into_iter()
            .find(|p| p.used_hold)
            .expect("a hold placement exists");
        assert_eq!(hold_placement.piece_type(), queue[0]);
        s.commit_placement(&hold_placement);
        assert_eq!(
            s.bag,
            BagState::full(),
            "queue-sourced hold commit left the bag alone"
        );

        // 2) Plain commit draining the queue — still no bag change.
        let plain_placement = placements_of(&s)
            .into_iter()
            .find(|p| !p.used_hold)
            .expect("a no-hold placement exists");
        s.commit_placement(&plain_placement);
        assert_eq!(
            s.bag,
            BagState::full(),
            "queue-sourced plain commit left the bag alone"
        );
        assert!(
            s.queue.is_empty(),
            "the two commits drained the 2-piece queue"
        );

        // 3) Speculative commit past the queue: the supplied piece IS a bag draw.
        let spec_placement = placements_of(&s)
            .into_iter()
            .find(|p| !p.used_hold)
            .expect("a placement for the active piece exists");
        s.commit_placement_with_next(&spec_placement, PieceType::S);
        assert!(
            !s.bag.contains(PieceType::S),
            "the speculative S was dealt from the bag"
        );
        assert!(
            s.bag.contains(PieceType::Z),
            "undealt pieces remain available"
        );
    }

    // ---- The garbage-mirror gold gate ----

    #[test]
    fn search_transition_predicts_the_engine_under_garbage_pressure() {
        // THE differential gate for garbage awareness: across a real game with
        // attack queued against the player, committing the chosen placement on
        // the SearchState must predict the engine's actual post-lock world —
        // board occupancy (rises included, exact hole columns), the pending
        // meter (cancellation included), and the B2B/combo chains. Any drift
        // between the two models shows up here as a board mismatch.
        use crate::ai::plan::placement_to_inputs;
        use crate::ai::search::{BestFirstPlanner, SearchBudget, think_to_completion};
        use crate::engine::{Engine, EngineConfig, InputFrame};

        let mut engine = Engine::new(EngineConfig::default(), 21);
        engine.step(InputFrame::default()); // spawn the first piece
        let eval = crate::ai::eval::LinearEvaluator::default();
        let mut planner = BestFirstPlanner::new();
        let (mut rises, mut sends) = (0u32, 0u32);

        for piece_index in 0..50 {
            // Periodic pressure: 1-3 lines queued every couple of pieces, so the
            // game exercises queue/cancel/rise (greedy clears often enough that
            // both branches run).
            if piece_index % 2 == 0 {
                engine.queue_garbage(1 + (piece_index / 2) % 3);
            }

            let snapshot = engine.snapshot();
            if snapshot.game_over.is_some() {
                break;
            }
            let state = SearchState::from_snapshot(&snapshot).expect("active piece");

            let Some(plan) =
                think_to_completion(&mut planner, &state, &eval, SearchBudget::single_ply())
            else {
                break; // no legal placement: the engine will top out shortly
            };

            // The search's imagined future.
            let mut predicted = state.clone();
            predicted.commit_placement(&plan.placement);

            // The engine's real future: execute the same plan.
            let frames =
                placement_to_inputs(&state.board.to_array2d(), &state.active, &plan.placement);
            for frame in frames {
                for event in engine.step(frame) {
                    match event {
                        crate::engine::EngineEvent::GarbageInserted { .. } => rises += 1,
                        crate::engine::EngineEvent::AttackSent { .. } => sends += 1,
                        _ => {}
                    }
                }
            }

            let after = engine.snapshot();
            if after.game_over.is_some() {
                // The strongest form of the gate: the engine died executing our
                // placement (a rise overflow / blocked spawn), and the search's
                // model must have seen that death coming.
                assert!(
                    predicted.dead,
                    "the engine died at piece {piece_index} but the search predicted a live future"
                );
                break;
            }
            assert!(
                !predicted.dead,
                "the search predicted death at piece {piece_index} but the engine plays on"
            );

            let mut engine_cells: Vec<(isize, isize)> =
                after.board_cells.iter().map(|c| (c.x, c.y)).collect();
            engine_cells.sort_unstable();
            let mut predicted_cells = predicted.board.cell_coords();
            predicted_cells.sort_unstable();
            assert_eq!(
                predicted_cells, engine_cells,
                "board diverged at piece {piece_index} (incl. garbage rises/holes)"
            );
            assert_eq!(
                predicted.pending.iter().map(|b| b.lines).sum::<u32>(),
                after.pending_garbage_total(),
                "pending meter diverged at piece {piece_index} (cancellation)"
            );
            assert_eq!(
                predicted.b2b, after.back_to_back_active,
                "B2B diverged at piece {piece_index}"
            );
            assert_eq!(
                predicted.combo, after.combo,
                "combo diverged at piece {piece_index}"
            );
        }

        // The duel is only meaningful if rising actually ran (greedy's clears
        // are mostly 0-attack Singles, so net sends are seed luck — the
        // cancellation MATH is pinned deterministically below instead).
        assert!(rises > 0, "the scenario never made garbage rise");
        let _ = sends;
    }

    /// A lock entirely above the skyline (lock-out) ends the engine's game:
    /// the branch must be marked dead and move no garbage.
    #[test]
    fn a_dying_lock_marks_the_branch_dead() {
        use crate::engine::GarbageBatch;
        let board = Board::with_top_margin(10, 20, 20);
        // An O resting on a pillar so high its cells sit at/above the skyline.
        let mut state = SearchState::for_test(
            board,
            ActivePiece::new(crate::engine::PieceType::O, (3, 20)),
            None,
            std::iter::empty(),
        );
        state.pending.push(GarbageBatch {
            lines: 2,
            hole_col: 0,
        });

        state.commit();

        assert!(state.dead, "a lock-out is terminal");
        assert_eq!(
            state.pending.len(),
            1,
            "a dying lock moves no garbage (no cancel, no rise)"
        );
    }

    /// A rise that forces the stack past the ceiling is the engine's BlockOut:
    /// the branch must be marked dead, not scored as a conveniently truncated
    /// (lower!) board.
    #[test]
    fn an_overflowing_rise_marks_the_branch_dead() {
        use crate::engine::GarbageBatch;
        let mut board = Board::new(10, 12);
        for y in 8..12 {
            board.set(0, y, CellKind::Some(crate::engine::PieceType::J));
        }
        let active = crate::ai::movegen::spawn_piece(crate::engine::PieceType::O, 10, 12);
        let mut state = SearchState::for_test(board, active, None, std::iter::empty());
        state.pending.push(GarbageBatch {
            lines: 8,
            hole_col: 5,
        });

        // Drop the O to the floor (clear-less), triggering the rise.
        while let Some(origin) = state.active.piece().try_move(
            &state.board,
            state.active.origin(),
            crate::engine::MoveDirection::Down,
        ) {
            state
                .active
                .move_to(origin, crate::engine::PieceAction::Fall);
        }
        state.commit();

        assert!(state.dead, "an overflowing rise is a BlockOut");
    }

    /// A spawn into occupied cells is the engine's BlockOut: the branch must be
    /// marked dead instead of searching from an impossible pose.
    #[test]
    fn a_blocked_spawn_marks_the_branch_dead() {
        let mut board = Board::new(10, 12);
        // The top two visible rows filled except one column: not clearable,
        // and any spawn pose collides.
        for y in [10, 11] {
            for x in 0..9 {
                board.set(x, y, CellKind::Some(crate::engine::PieceType::J));
            }
        }
        let active = crate::ai::movegen::spawn_piece(crate::engine::PieceType::O, 10, 12);
        let mut state = SearchState::for_test(board, active, None, [crate::engine::PieceType::T]);

        // Drop to the floor (no clear, no pending), then the T must spawn into
        // the filled rows.
        while let Some(origin) = state.active.piece().try_move(
            &state.board,
            state.active.origin(),
            crate::engine::MoveDirection::Down,
        ) {
            state
                .active
                .move_to(origin, crate::engine::PieceAction::Fall);
        }
        state.commit();

        assert!(state.dead, "spawning into the stack is a BlockOut");
    }

    /// Deterministic pin of the mirrored cancellation math: a Tetris that
    /// perfect-clears (4 + 10 attack, no B2B, no combo) against pending
    /// [3, 12] must kill the first batch and eat 11 lines of the second —
    /// exactly the engine's oldest-first, line-for-line rule.
    #[test]
    fn mirrored_cancellation_matches_the_attack_table() {
        use crate::engine::{GarbageBatch, PieceRotation, RotationDirection};

        // The 4-wide Tetris well: cols 0-2 filled four rows high.
        let mut board = Board::new(4, 12);
        for y in 0..4 {
            for x in 0..3 {
                board.set(x, y, CellKind::Some(crate::engine::PieceType::O));
            }
        }
        let mut vertical_i = ActivePiece::new(crate::engine::PieceType::I, (1, 0));
        vertical_i.rotate_to(
            PieceRotation::R90,
            (1, 0),
            RotationDirection::Clockwise,
            1,
            false,
        );
        let mut state = SearchState::for_test(board, vertical_i, None, std::iter::empty());
        state.pending.push(GarbageBatch {
            lines: 3,
            hole_col: 1,
        });
        state.pending.push(GarbageBatch {
            lines: 12,
            hole_col: 2,
        });

        state.commit();

        assert!(state.board.is_empty(), "the Tetris perfect-cleared");
        assert_eq!(
            state.pending.as_slice(),
            [GarbageBatch {
                lines: 1,
                hole_col: 2
            }],
            "14 attack cancels 3 then 11 of 12, oldest first"
        );
    }

    /// Deterministic pin of mirrored rising and deferral: a clear-less lock
    /// raises pending rows (cap respected, snapshot hole columns), while a
    /// clearing lock defers rising entirely.
    #[test]
    fn mirrored_rise_obeys_cap_and_clears_defer() {
        use crate::engine::GarbageBatch;

        // Clear-less lock: an O dropped on an empty board.
        let board = Board::new(10, 24);
        let active = crate::ai::movegen::spawn_piece(crate::engine::PieceType::O, 10, 24);
        let mut state = SearchState::for_test(board, active, None, std::iter::empty());
        state.garbage_cap = 4;
        state.pending.push(GarbageBatch {
            lines: 6,
            hole_col: 3,
        });

        // Drop to the floor, then lock.
        while state
            .active
            .piece()
            .try_move(
                &state.board,
                state.active.origin(),
                crate::engine::MoveDirection::Down,
            )
            .is_some()
        {
            let origin = state
                .active
                .piece()
                .try_move(
                    &state.board,
                    state.active.origin(),
                    crate::engine::MoveDirection::Down,
                )
                .unwrap();
            state
                .active
                .move_to(origin, crate::engine::PieceAction::Fall);
        }
        state.commit();

        // 4 of 6 lines rose (cap), hole col 3; remainder stays pending.
        assert_eq!(
            state.pending.as_slice(),
            [GarbageBatch {
                lines: 2,
                hole_col: 3
            }]
        );
        let cells = state.board.cell_coords();
        for y in 0..4 {
            let row: Vec<isize> = cells
                .iter()
                .filter(|&&(_, cy)| cy == y)
                .map(|&(x, _)| x)
                .collect();
            assert_eq!(row.len(), 9, "garbage row {y} is full except the hole");
            assert!(!row.contains(&3), "hole col 3 in garbage row {y}");
        }
    }
}
