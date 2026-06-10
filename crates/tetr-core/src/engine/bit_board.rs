//! A `Copy`, allocation-free occupancy board for the AI search hot path.
//!
//! # Why this exists
//!
//! The search forks its state once per candidate placement (thousands per piece).
//! The game's [`Board`](super::Board) is an `Array2D<Cell>` (~9.6 KB, heap-backed), so
//! each fork **heap-allocates and copies the whole grid** — profiling showed this clone
//! is the dominant remaining per-piece cost once the lock path and hashing were fixed.
//!
//! [`BitBoard`] is the search's answer: occupancy packed as one `u64` per column (bit
//! `y` set ⇔ `(x, y)` filled), in a **fixed inline array**, so the whole board is
//! `Copy` — a fork is a register/stack copy with **zero allocation**, collision is a
//! bit test, and "column bits" (the evaluator + transposition key input) is the board
//! itself. Piece colour is intentionally dropped: the search only ever needs occupancy
//! (collision, line clears, features); colour is a rendering concern that stays in the
//! game's `Board`.
//!
//! # Equivalence to the engine
//!
//! Every operation mirrors the engine's `Board`/`clear_lines` semantics **exactly**,
//! asserted by `bit_board_matches_engine_*` differential tests on randomized boards:
//! out-of-bounds (walls/floor) and occupied cells block identically, and a line clear
//! removes the same rows and compacts to the same occupancy. In particular the clear
//! reproduces two engine behaviours the differential test pins down:
//! - it clears every completely-filled row across the **full backing matrix** (visible
//!   field + buffer) — `clear_full_rows` loops `y < total_rows`, mirroring the engine's
//!   `clear_lines` over `backing_rows()` — so a full buffer row is removed, not left
//!   floating, and
//! - it clears **iteratively, re-scanning** the same row index after each clear, so a
//!   row that shifts down into a just-cleared slot is itself examined.

use super::{ActivePiece, Board, CellKind, LockOutcome, PieceType};
use smallvec::SmallVec;

/// Max supported board width. Standard guideline Tetris is 10; the cap leaves headroom
/// while keeping a board to `16 * 8 = 128` bytes (a cheap `Copy`). A column is a `u64`,
/// so up to 64 rows (visible + buffer) — comfortably above the 40 the engine uses.
pub const MAX_WIDTH: usize = 16;

/// Read-only collision query, shared by the engine [`Board`] and the search [`BitBoard`]
/// so collision-using logic (T-spin corners, and eventually movegen) is written **once**
/// against the interface, not the representation. `blocked(x, y)` is true for a wall/floor
/// (out of the side or bottom bounds) or an occupied cell; space above the stack reads
/// false (a piece may occupy it). This is the "clear interface" the search board hides behind.
pub trait Occupancy {
    fn blocked(&self, x: isize, y: isize) -> bool;
}

impl Occupancy for Board {
    fn blocked(&self, x: isize, y: isize) -> bool {
        !matches!(self.get_cell_kind(x, y), crate::engine::CellKind::None)
    }
}

/// A fixed-size, `Copy` occupancy board: `cols[x]` has bit `y` set iff `(x, y)` is filled.
///
/// Cloning is a flat copy — no heap, no allocator traffic — so the search's
/// fork-per-placement hot path stays allocation-free.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct BitBoard {
    cols: [u64; MAX_WIDTH],
    width: usize,
    /// Visible field height — mirrors `Board::height` for spawn coords and the game-over
    /// skyline. Rows at or above this are the hidden buffer; clears span the full matrix
    /// (`total_rows`), not just here.
    visible_rows: usize,
    /// Total backing rows (visible + buffer). A cell cannot be placed at or above this —
    /// a lock overhanging the top of the grid drops those cells, matching `Board::set`.
    total_rows: usize,
}

impl BitBoard {
    /// An empty board of the given `width` (clamped to [`MAX_WIDTH`]), visible height,
    /// and total backing rows (clamped to 64, a `u64` column's capacity).
    pub fn empty(width: usize, visible_rows: usize, total_rows: usize) -> Self {
        Self {
            cols: [0; MAX_WIDTH],
            width: width.min(MAX_WIDTH),
            visible_rows,
            total_rows: total_rows.min(64),
        }
    }

    /// Pack an engine [`Board`]'s occupancy into a `BitBoard` (colour discarded).
    ///
    /// The mirror's envelope is [`MAX_WIDTH`] × 64 rows; a larger board would be
    /// silently truncated by the clamps below, so it is rejected in debug builds.
    pub fn from_board(board: &Board) -> Self {
        debug_assert!(
            board.width() <= MAX_WIDTH && board.backing_rows() <= 64,
            "BitBoard mirrors boards up to {MAX_WIDTH}x64; a {}x{} board would be truncated",
            board.width(),
            board.backing_rows(),
        );
        let mut bb = Self::empty(board.width(), board.height(), board.backing_rows());
        for (x, col) in board.column_bits().iter().enumerate().take(MAX_WIDTH) {
            bb.cols[x] = *col;
        }
        bb
    }

    /// Board width (number of active columns).
    pub fn width(&self) -> usize {
        self.width
    }

    /// Visible playfield height (excludes the buffer) — mirrors `Board::height`, for
    /// spawn coordinates and iteration bounds in the search.
    pub fn height(&self) -> usize {
        self.visible_rows
    }

    /// All occupied cells as `(x, y)` pairs — mirrors `Board::cell_coords` for the
    /// search's state snapshots / dedup fingerprints.
    pub fn cell_coords(&self) -> Vec<(isize, isize)> {
        let mut out = Vec::new();
        for (x, &col) in self.cols[..self.width].iter().enumerate() {
            let mut bits = col;
            while bits != 0 {
                let y = bits.trailing_zeros() as isize;
                out.push((x as isize, y));
                bits &= bits - 1;
            }
        }
        out
    }

    /// The column bitboard — the evaluator + transposition-key input, returned with no
    /// copy or allocation (the board *is* the bitboard).
    pub fn columns(&self) -> &[u64] {
        &self.cols[..self.width]
    }

    /// Whether `(x, y)` is occupied (in-bounds only; out-of-bounds reads `false`).
    pub fn occupied(&self, x: isize, y: isize) -> bool {
        if x < 0 || y < 0 || x as usize >= self.width || y >= 64 {
            return false;
        }
        self.cols[x as usize] & (1u64 << y) != 0
    }

    /// Fill `(x, y)` (no-op if out of bounds).
    pub(crate) fn set(&mut self, x: isize, y: isize) {
        if x < 0 || y < 0 || x as usize >= self.width || y as usize >= self.total_rows {
            return; // off the side/floor, or above the backing grid (cell is dropped)
        }
        self.cols[x as usize] |= 1u64 << y;
    }

    /// True iff no cell is filled.
    pub fn is_empty(&self) -> bool {
        self.cols[..self.width].iter().all(|&c| c == 0)
    }

    /// Highest occupied row across all columns, or `None` if empty. (The skyline the
    /// engine reports as `top_y_after_lock`.) Delegates to [`highest_occupied_y`], the
    /// one impl shared with the engine's `lock_and_clear`.
    pub(crate) fn highest_y(&self) -> Option<isize> {
        highest_occupied_y(self.columns())
    }

    /// Indices of completely-filled rows across the whole 64-bit range, ascending — the
    /// `cleared_rows` the engine reports (`lock_clear::full_rows`, buffer included).
    /// These are exactly the rows [`clear_full_rows`](Self::clear_full_rows) removes —
    /// report and clear coincide. Delegates to the free [`full_rows`], the one impl
    /// shared with the engine's `lock_and_clear`.
    pub(crate) fn full_rows(&self) -> Vec<isize> {
        full_rows(self.columns())
    }

    /// Clear full rows and compact the stack downward, **exactly** as the engine's
    /// `clear_lines`: scan the full backing range (visible field + buffer) bottom-up,
    /// and whenever the current row is full, drop it (shifting every row above down one)
    /// and re-examine the same index.
    pub(crate) fn clear_full_rows(&mut self) {
        let mut y = 0u32;
        while (y as usize) < self.total_rows {
            let bit = 1u64 << y;
            if self.cols[..self.width].iter().all(|&c| c & bit != 0) {
                // Row `y` is full: remove it and shift everything above down by one
                // (carrying buffer rows down too). Do NOT advance `y` — the row that fell
                // into `y` must be re-examined. `checked_shr` keeps the top row safe:
                // at `y == 63` (reachable only at the `total_rows == 64` ceiling) nothing
                // sits above it, so the carried-down part is 0 rather than a `>> 64`
                // overflow — matching the engine `Board`, which clears the top row fine.
                let below_mask = bit - 1;
                for col in &mut self.cols[..self.width] {
                    let above = (*col).checked_shr(y + 1).unwrap_or(0);
                    *col = (*col & below_mask) | (above << y);
                }
            } else {
                y += 1;
            }
        }
    }

    /// Lock a piece's absolute `cells` onto the board, clear any resulting lines, and
    /// report `(cleared_rows, skyline)` — the bitboard analogue of the engine's
    /// [`lock_and_clear`](super::lock_and_clear). `cleared_rows` spans the full backing
    /// range (matching the engine's `full_rows`), and the board is mutated by
    /// [`clear_full_rows`](Self::clear_full_rows), which removes exactly those rows. Cells
    /// out of bounds are skipped, exactly as `Board::set` does on the engine side.
    pub(crate) fn lock(&mut self, cells: &[(isize, isize)]) -> (Vec<isize>, Option<isize>) {
        for &(x, y) in cells {
            self.set(x, y);
        }
        let cleared_rows = self.full_rows();
        if !cleared_rows.is_empty() {
            self.clear_full_rows();
        }
        (cleared_rows, self.highest_y())
    }

    /// Lock `piece` (at its current pose) onto the board and build the engine-shaped
    /// [`LockOutcome`] — the bitboard equivalent of `lock_and_clear(piece, board)`.
    /// Shared by `SearchState::commit*` and `score_placement` so they cannot diverge;
    /// its equivalence to `lock_and_clear` is differential-tested in `tests`.
    pub fn lock_piece(&mut self, piece: &ActivePiece) -> LockOutcome {
        let (ox, oy) = piece.origin();
        let kind = CellKind::Some(piece.piece_type());
        let cells: SmallVec<[(isize, isize); 4]> = piece
            .piece()
            .cells()
            .iter()
            .map(|&(cx, cy)| (cx + ox, cy + oy))
            .collect();
        let (cleared_rows, top_y_after_lock) = self.lock(&cells);
        LockOutcome {
            cells_locked: cells.into_iter().map(|(x, y)| (x, y, kind)).collect(),
            cleared_rows,
            top_y_after_lock,
        }
    }

    /// Reconstruct an engine [`Board`] carrying this occupancy (cell *colour* is
    /// arbitrary — only occupancy is preserved). The inverse of [`from_board`](Self::from_board),
    /// for the fallback paths that still want an `Array2D` (e.g. the default
    /// `Evaluator::evaluate_cols`). Allocating; the hot paths avoid it.
    pub fn to_array2d(&self) -> Board {
        let margin = self.total_rows.saturating_sub(self.visible_rows);
        let mut board = Board::with_top_margin(self.width, self.visible_rows, margin);
        for (x, &col) in self.cols[..self.width].iter().enumerate() {
            let mut bits = col;
            while bits != 0 {
                let y = bits.trailing_zeros() as isize;
                board.set(x as isize, y, CellKind::Some(PieceType::I));
                bits &= bits - 1; // clear the lowest set bit
            }
        }
        board
    }

    /// A read-only [`ColumnView`] of this board — the evaluator-facing handle that
    /// keeps the evaluator seam from naming the concrete search board.
    pub fn view(&self) -> ColumnView<'_> {
        ColumnView { board: self }
    }
}

/// A representation-neutral, read-only column view of a board: the evaluator's input.
/// It exposes only the column bitset, emptiness, and a dense-[`Board`] reconstruction
/// for the fallback path, so the evaluator depends on this small surface rather than on
/// the concrete [`BitBoard`] and its full API.
#[derive(Clone, Copy)]
pub struct ColumnView<'a> {
    board: &'a BitBoard,
}

impl ColumnView<'_> {
    /// The column bitset: `columns()[x]` has bit `y` set iff `(x, y)` is filled.
    pub fn columns(&self) -> &[u64] {
        self.board.columns()
    }

    /// True iff no cell is filled.
    pub fn is_empty(&self) -> bool {
        self.board.is_empty()
    }

    /// Reconstruct a dense engine [`Board`] with this occupancy — the fallback for
    /// evaluators that score the dense board (allocating; the fast path uses `columns`).
    pub fn to_board(&self) -> Board {
        self.board.to_array2d()
    }
}

/// Indices of completely-filled rows in a column bitboard, ascending. A row is full
/// iff every column has its bit set there, so the bitwise-AND of all columns has exactly
/// the full rows' bits set; the scan spans the full 64-bit range so buffer-zone clears
/// are reported too (the `lock_clear` buffer-zone note). The **single** implementation
/// behind [`BitBoard::full_rows`] and the engine's [`lock_and_clear`](super::lock_and_clear),
/// so the search bitboard and the engine board can never disagree on what cleared.
pub(crate) fn full_rows(cols: &[u64]) -> Vec<isize> {
    // A zero-width board AND-folds to `!0` ("all 64 rows full"); there are no
    // columns, so there are no rows to clear — not 64 phantom ones.
    if cols.is_empty() {
        return Vec::new();
    }
    // AND-fold every column: a row is full iff its bit is set in all columns.
    let full = cols.iter().fold(!0u64, |acc, &c| acc & c);
    // Hot common case — nothing full (most locks clear no line): skip the per-row scan
    // and the allocation entirely. `Vec::new()` does not allocate.
    if full == 0 {
        return Vec::new();
    }
    // Walk only the set bits (lowest→highest ⇒ ascending), cheaper than a 0..64 filter.
    let mut rows = Vec::new();
    let mut bits = full;
    while bits != 0 {
        rows.push(bits.trailing_zeros() as isize);
        bits &= bits - 1;
    }
    rows
}

/// Highest occupied row across a column bitboard, or `None` if every column is empty —
/// the skyline the engine reports as `top_y_after_lock`. The **single** implementation
/// behind [`BitBoard::highest_y`] and the engine's [`lock_and_clear`](super::lock_and_clear).
pub(crate) fn highest_occupied_y(cols: &[u64]) -> Option<isize> {
    cols.iter()
        .filter(|&&c| c != 0)
        .map(|&c| (u64::BITS - 1 - c.leading_zeros()) as isize)
        .max()
}

impl Occupancy for BitBoard {
    /// Blocks for a wall/floor (out of the side or bottom bounds) or an occupied cell;
    /// space above the stack reads `false`. Mirrors the engine `Board`'s collision rule.
    fn blocked(&self, x: isize, y: isize) -> bool {
        if x < 0 || y < 0 || x as usize >= self.width {
            return true; // wall / floor
        }
        y < 64 && self.cols[x as usize] & (1u64 << y) != 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{Board, CellKind, PieceType};

    /// A tiny deterministic PRNG (SplitMix64) — seeded, no OS entropy, reproducible.
    struct SplitMix64(u64);
    impl SplitMix64 {
        fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }
        fn below(&mut self, n: u64) -> u64 {
            self.next() % n
        }
    }

    /// Build a random engine `Board` (10 wide, 20 visible + 20 buffer) where each cell is
    /// filled with probability `fill_pct/100`, plus the equivalent `BitBoard`. Higher
    /// `fill_pct` makes full rows (and thus clears) likely.
    fn random_pair(rng: &mut SplitMix64, fill_pct: u64) -> (Board, BitBoard) {
        let (w, h, margin) = (10usize, 20usize, 20usize);
        let mut board = Board::with_top_margin(w, h, margin);
        for y in 0..(h + margin) as isize {
            for x in 0..w as isize {
                if rng.below(100) < fill_pct {
                    board.set(x, y, CellKind::Some(PieceType::I));
                }
            }
        }
        let bb = BitBoard::from_board(&board);
        (board, bb)
    }

    #[test]
    fn bit_board_matches_engine_occupancy_and_collision() {
        let mut rng = SplitMix64(0xC0FF_EE12_3456_789A);
        for _ in 0..200 {
            let (board, bb) = random_pair(&mut rng, 50);
            for y in -2..62 {
                for x in -2..12 {
                    let engine_blocked = !matches!(board.get_cell_kind(x, y), CellKind::None);
                    assert_eq!(
                        bb.blocked(x, y),
                        engine_blocked,
                        "blocked mismatch at ({x},{y})"
                    );
                    let engine_occupied = matches!(board.get_cell_kind(x, y), CellKind::Some(_));
                    assert_eq!(bb.occupied(x, y), engine_occupied, "occupied at ({x},{y})");
                }
            }
        }
    }

    #[test]
    fn bit_board_matches_engine_line_clear_and_skyline() {
        let mut rng = SplitMix64(0x1234_5678_9ABC_DEF0);
        for _ in 0..400 {
            // High fill so full rows (and multi-row clears, including buffer rows that
            // shift into the visible field) are common.
            let fill = 60 + rng.below(35); // 60..95%
            let (mut board, mut bb) = random_pair(&mut rng, fill);

            // Clearing must leave the identical occupancy as the engine's `clear_lines`
            // (the compaction == the engine's iterative, full-matrix per-row drop) and
            // the same skyline afterward.
            bb.clear_full_rows();
            board.clear_lines();
            let engine_after = BitBoard::from_board(&board);
            assert_eq!(
                bb.columns(),
                engine_after.columns(),
                "post-clear occupancy mismatch"
            );
            assert_eq!(bb.highest_y(), engine_after.highest_y(), "skyline mismatch");
        }
    }

    #[test]
    fn full_rows_reports_buffer_rows_too() {
        // The reported full rows span the whole range (matching `lock_clear::full_rows`);
        // `clear_full_rows` removes exactly these, so report and clear coincide.
        let mut bb = BitBoard::empty(4, 20, 40);
        for x in 0..4 {
            bb.set(x, 0); // a full visible row
            bb.set(x, 25); // a full buffer row
        }
        assert_eq!(bb.full_rows(), vec![0, 25]);
    }

    #[test]
    fn clear_full_rows_clears_buffer_rows() {
        // The search-side mirror of `clear_lines_clears_a_full_buffer_row`. The randomized
        // differential test only proves engine == bitboard, so pin the full-matrix clear
        // directly: a full buffer row must be removed, not left floating.
        let mut bb = BitBoard::empty(4, 20, 40);
        for x in 0..4 {
            bb.set(x, 0); // a full visible row
            bb.set(x, 25); // a full buffer row
        }
        bb.set(0, 30); // a lone sentinel above both full rows

        bb.clear_full_rows();

        assert!(
            bb.full_rows().is_empty(),
            "both full rows (visible and buffer) are gone"
        );
        assert!(
            bb.occupied(0, 28),
            "the sentinel fell by the two cleared rows (30 -> 28)"
        );
        assert!(
            !bb.occupied(0, 29) && !bb.occupied(0, 30),
            "nothing left above it"
        );
    }

    #[test]
    fn clear_full_rows_handles_the_top_row_at_the_64_row_ceiling() {
        // ENG-3 guard: at the `total_rows == 64` clamp ceiling the clear loop reaches
        // y == 63, where the carry-down shift would be `>> 64`. `checked_shr` keeps it
        // safe (nothing sits above the top row), matching the engine Board.
        let mut bb = BitBoard::empty(4, 20, 64);
        for x in 0..4 {
            bb.set(x, 63); // a full row at the very top of the matrix
        }
        bb.clear_full_rows(); // must not panic
        assert!(bb.full_rows().is_empty(), "the top row cleared");
        assert!(
            (0..4).all(|x| !bb.occupied(x, 63)),
            "nothing left at the top"
        );
    }

    #[test]
    fn clone_is_copy_and_independent() {
        let mut a = BitBoard::empty(10, 20, 40);
        a.set(3, 5);
        let b = a; // Copy
        a.set(4, 6);
        assert!(b.occupied(3, 5));
        assert!(
            !b.occupied(4, 6),
            "the copy must not see mutations to the original"
        );
    }

    #[test]
    fn bit_board_lock_matches_engine_lock_and_clear() {
        use crate::engine::{lock_and_clear, ActivePiece};
        let mut rng = SplitMix64(0xACE1_2345_6789_BCDE);
        for _ in 0..500 {
            let fill = 40 + rng.below(45); // 40..85%, so clears happen often enough
            let (board, bb) = random_pair(&mut rng, fill);

            // A random piece at a random origin (it may overhang the field — out-of-bounds
            // cells are skipped by both sides, exactly like `Board::set`).
            let piece = PieceType::all()[rng.below(7) as usize];
            let origin = (rng.below(12) as isize - 1, rng.below(42) as isize);
            let active = ActivePiece::new(piece, origin);
            let cells: Vec<(isize, isize)> = active
                .piece()
                .cells()
                .iter()
                .map(|&(cx, cy)| (cx + origin.0, cy + origin.1))
                .collect();

            let mut engine_board = board.clone();
            let outcome = lock_and_clear(&active, &mut engine_board);

            let mut bb = bb;
            let (cleared, top) = bb.lock(&cells);

            assert_eq!(cleared, outcome.cleared_rows, "cleared_rows mismatch");
            assert_eq!(top, outcome.top_y_after_lock, "skyline mismatch");
            assert_eq!(
                bb.columns(),
                BitBoard::from_board(&engine_board).columns(),
                "post-lock occupancy mismatch"
            );
        }
    }

    #[test]
    fn bit_board_t_spin_matches_engine() {
        use crate::engine::{classify_t_spin, ActivePiece, PieceRotation, RotationDirection};
        let rotations = [
            PieceRotation::R0,
            PieceRotation::R90,
            PieceRotation::R180,
            PieceRotation::R270,
        ];
        let mut rng = SplitMix64(0xBADC_0FFE_E0DD_F00D);
        for _ in 0..300 {
            let (board, bb) = random_pair(&mut rng, 45);
            let origin = (rng.below(10) as isize, rng.below(38) as isize);
            let mut t = ActivePiece::new(PieceType::T, origin);
            t.rotate_to(
                rotations[rng.below(4) as usize],
                origin,
                RotationDirection::Clockwise,
                1,
                false,
            );
            // `classify_t_spin` is now generic over `Occupancy`: the engine `Board` and the
            // search `BitBoard` must classify identically (their `blocked` semantics match).
            assert_eq!(
                classify_t_spin(&t, &board),
                classify_t_spin(&t, &bb),
                "t-spin classification mismatch at {origin:?}"
            );
        }
    }
}
