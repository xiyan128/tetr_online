//! The playfield: ONE board, two planes.
//!
//! [`Board`] composes the [`BitBoard`] **occupancy plane** (the single home of
//! every board rule: full rows, compaction, garbage insertion, overflow, the
//! skyline) with a flat **colour plane** (`Vec<CellKind>`, row-major over the
//! backing grid) carrying per-cell identity for snapshots and rendering.
//! Mutations update both planes in lockstep; reads dispatch to whichever
//! plane answers. The rules cannot disagree with the search's view by
//! construction — the colour plane is *driven by* the bitboard's own results
//! (its full-row list, its overflow verdict), never recomputed beside them.
//! (Before the 2026-06-10 unification these were two complete board
//! implementations kept aligned by randomized differential tests; see
//! `docs/adr-board-unification.md`.)
//!
//! Addressing is signed `(x, y)` with the origin at the bottom-left; an
//! optional top margin holds the hidden spawn rows. Off-grid reads resolve to
//! [`CellKind::Wall`] (sides/floor) so collision checks need no bounds
//! special-casing.

use std::fmt::{Display, Write};

use crate::engine::bit_board::{BitBoard, MAX_WIDTH};
use crate::engine::pieces::PieceType;
use smallvec::SmallVec;

#[derive(Clone)]
pub struct Board {
    width: usize,
    height: usize,
    /// Total rows (visible + hidden buffer).
    backing: usize,
    /// Occupancy truth — the rules live here.
    bits: BitBoard,
    /// Identity truth — `colors[y * width + x]`, `None` where unoccupied.
    colors: Vec<CellKind>,
}

impl Board {
    pub fn new(width: usize, height: usize) -> Self {
        Self::with_top_margin(width, height, 0)
    }

    pub fn with_top_margin(width: usize, height: usize, margin: usize) -> Self {
        let backing = height + margin;
        assert!(
            width <= MAX_WIDTH && backing <= 64,
            "Board envelope is {MAX_WIDTH}x64 (requested {width}x{backing}): the occupancy \
             plane is a u64-column bitboard"
        );
        Self {
            width,
            height,
            backing,
            bits: BitBoard::empty(width, height, backing),
            colors: vec![CellKind::None; backing * width],
        }
    }

    #[inline]
    fn index(&self, x: usize, y: usize) -> usize {
        y * self.width + x
    }

    /// Write `cell_kind` at `(x, y)`, keeping both planes in lockstep. Returns
    /// `false` (a dropped write) outside the backing grid — a cell cannot
    /// exist off the grid.
    pub fn set(&mut self, x: isize, y: isize, cell_kind: CellKind) -> bool {
        // Wall is the off-grid sentinel, never a stored cell: writing it would
        // silently diverge the planes (Wall.is_some() is false, so the bit
        // would clear while the colour said Wall). Reject it loudly.
        debug_assert!(
            cell_kind != CellKind::Wall,
            "CellKind::Wall is a boundary sentinel, not a storable cell"
        );
        if x < 0 || y < 0 || x >= self.width as isize || y >= self.backing as isize {
            return false;
        }
        let idx = self.index(x as usize, y as usize);
        self.colors[idx] = cell_kind;
        if cell_kind.is_some() {
            self.bits.set(x, y);
        } else {
            self.bits.clear(x, y);
        }
        true
    }

    /// Total backing rows (visible field + hidden buffer): the grid height a cell can
    /// occupy. A `set`/lock at or above this is dropped — a cell cannot exist off the
    /// top of the grid.
    pub fn backing_rows(&self) -> usize {
        self.backing
    }

    /// Every occupied cell as `(x, y, kind)`, row-major (bottom row first).
    pub(crate) fn cells(&self) -> Vec<(isize, isize, CellKind)> {
        let mut out = Vec::new();
        for y in 0..self.backing {
            for x in 0..self.width {
                let kind = self.colors[self.index(x, y)];
                if kind.is_some() {
                    out.push((x as isize, y as isize, kind));
                }
            }
        }
        out
    }

    /// True iff no playfield cell (visible **or** buffer) is filled — i.e. a perfect
    /// clear. O(width) bit test; called on the line-clear hot path.
    pub fn is_empty(&self) -> bool {
        self.bits.is_empty()
    }

    pub fn get_cell_kind(&self, x: isize, y: isize) -> CellKind {
        if x < 0 || y < 0 || x >= self.width as isize {
            return CellKind::Wall;
        }
        if (y as usize) < self.backing {
            self.colors[y as usize * self.width + x as usize]
        } else {
            CellKind::None
        }
    }

    pub fn cell_coords(&self) -> Vec<(isize, isize)> {
        self.cells().iter().map(|&(x, y, _)| (x, y)).collect()
    }

    /// The column bitboard: `result[x]` has bit `y` set iff `(x, y)` is occupied
    /// (buffer rows included). A direct copy out of the occupancy plane — no scan.
    /// Shared by the evaluators and `lock_and_clear`'s full-row/skyline queries.
    pub fn column_bits(&self) -> SmallVec<[u64; 16]> {
        SmallVec::from_slice(self.bits.columns())
    }

    /// The occupancy plane itself (`Copy`) — the search's fork currency. This
    /// is what `SearchState::from_snapshot` seeds from; identical by
    /// construction to the colours' occupancy (the lockstep invariant).
    pub fn bits(&self) -> BitBoard {
        self.bits
    }

    /// Remove every completely-filled row across the **full backing matrix** (visible
    /// field + hidden buffer) and compact the stack downward, returning the count
    /// removed. The full-row list comes from the occupancy plane (the ONE rule
    /// home); the colour plane compacts by that list, so the two planes cannot
    /// diverge on what cleared.
    pub fn clear_lines(&mut self) -> usize {
        let full = self.bits.full_rows();
        if full.is_empty() {
            return 0;
        }
        // Compact colours: keep non-full rows in order, pad the top with empties.
        let mut compacted = Vec::with_capacity(self.colors.len());
        let mut is_full = [false; 64];
        for &y in &full {
            is_full[y as usize] = true;
        }
        for (y, full) in is_full.iter().enumerate().take(self.backing) {
            if !full {
                let start = self.index(0, y);
                compacted.extend_from_slice(&self.colors[start..start + self.width]);
            }
        }
        compacted.resize(self.colors.len(), CellKind::None);
        self.colors = compacted;
        self.bits.clear_full_rows();
        full.len()
    }

    /// Insert `count` garbage rows at the bottom, shifting the whole stack up.
    /// Each new row is full except `hole_col`, painted [`CellKind::Garbage`] so
    /// a renderer can tell attack from the player's own stack. Returns `true`
    /// if any pre-existing cell was forced past the backing top (a
    /// garbage-induced top-out for the caller to act on) — the verdict comes
    /// from the occupancy plane's own insertion.
    pub fn insert_garbage_lines(&mut self, count: usize, hole_col: usize) -> bool {
        if count == 0 {
            return false;
        }
        // A hole column past the right wall would fill the whole row (no gap); clamp
        // so out-of-range garbage always leaves a diggable hole rather than a free clear.
        let hole_col = hole_col.min(self.width.saturating_sub(1));
        let overflow = self.bits.insert_garbage_lines(count, hole_col);

        // Shift colours up by `count` rows (rows pushed past the top drop), then
        // paint the new bottom rows garbage-except-hole — mirroring the bit shift.
        let mut shifted = vec![CellKind::None; self.colors.len()];
        for y in 0..self.backing.saturating_sub(count) {
            let src = self.index(0, y);
            let dst = (y + count) * self.width;
            shifted[dst..dst + self.width].copy_from_slice(&self.colors[src..src + self.width]);
        }
        for y in 0..count.min(self.backing) {
            for x in 0..self.width {
                if x != hole_col {
                    shifted[y * self.width + x] = CellKind::Garbage;
                }
            }
        }
        self.colors = shifted;
        overflow
    }

    pub fn width(&self) -> usize {
        self.width
    }

    pub fn height(&self) -> usize {
        self.height
    }
}

impl crate::engine::bit_board::Occupancy for Board {
    fn blocked(&self, x: isize, y: isize) -> bool {
        // Collision is an occupancy question: answer from the bit plane (the
        // colour plane would give the identical answer — the lockstep
        // invariant — but the bits answer in two compares and a mask).
        self.bits.blocked(x, y)
    }
}

impl Display for Board {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Render top row first so the output reads like the on-screen board.
        for y in (0..self.height as isize).rev() {
            for x in 0..self.width as isize {
                f.write_str(match self.get_cell_kind(x, y) {
                    CellKind::Some(_) => "X",
                    CellKind::Garbage => "G",
                    CellKind::None => "#",
                    CellKind::Wall => " ",
                })?;
            }
            f.write_char('\n')?;
        }
        Ok(())
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum CellKind {
    Some(PieceType),
    None,
    Wall,
    /// A garbage-row cell (versus). Occupied exactly like `Some` — it collides,
    /// fills rows, and clears — but carries no piece identity, so a renderer
    /// can paint it neutral instead of a piece colour. Occupancy predicates go
    /// through [`CellKind::is_some`] / [`CellKind::is_none`], which treat it as
    /// filled.
    Garbage,
}

impl CellKind {
    /// A filled mino cell — a locked piece or a garbage cell. This is the
    /// "counts toward a full row / collides / tops out" predicate; `Wall` is
    /// not `some`.
    pub fn is_some(&self) -> bool {
        matches!(self, CellKind::Some(_) | CellKind::Garbage)
    }

    pub fn is_none(&self) -> bool {
        matches!(self, CellKind::None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fill_row(board: &mut Board, y: isize, piece_type: PieceType) {
        for x in 0..board.width {
            assert!(board.set(x as isize, y, CellKind::Some(piece_type)));
        }
    }

    /// THE lockstep invariant: occupancy derived from the colour plane equals
    /// the bit plane, under a randomized op sequence (sets, clears, garbage
    /// inserts, line clears). This replaces the five cross-representation
    /// differential tests the unification retired.
    #[test]
    fn planes_stay_in_lockstep_under_random_ops() {
        let mut rng = 0x1234_5678_9ABC_DEFFu64;
        let mut next = move || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng
        };
        let mut board = Board::with_top_margin(10, 20, 20);
        for _ in 0..5_000 {
            match next() % 10 {
                0..=5 => {
                    let x = (next() % 10) as isize;
                    let y = (next() % 40) as isize;
                    let kind = match next() % 3 {
                        0 => CellKind::Some(PieceType::T),
                        1 => CellKind::Garbage,
                        _ => CellKind::None,
                    };
                    board.set(x, y, kind);
                }
                6..=7 => {
                    board.insert_garbage_lines((next() % 3) as usize + 1, (next() % 12) as usize);
                }
                _ => {
                    // Fill a random row to make clears reachable, then clear.
                    let y = (next() % 40) as isize;
                    fill_row(&mut board, y, PieceType::O);
                    board.clear_lines();
                }
            }
            // Invariant: every cell agrees across the planes.
            for y in 0..board.backing_rows() as isize {
                for x in 0..board.width() as isize {
                    assert_eq!(
                        board.get_cell_kind(x, y).is_some(),
                        board.bits().occupied(x, y),
                        "plane disagreement at ({x},{y})"
                    );
                }
            }
        }
    }

    #[test]
    fn set_and_get_round_trip_inside_visible_board() {
        let mut board = Board::new(10, 20);

        assert!(board.set(3, 4, CellKind::Some(PieceType::T)));
        assert_eq!(board.get_cell_kind(3, 4), CellKind::Some(PieceType::T));
    }

    #[test]
    fn is_empty_tracks_occupancy_including_the_buffer() {
        let mut board = Board::with_top_margin(10, 20, 20);
        assert!(board.is_empty(), "a fresh board is empty");

        // A filled cell up in the hidden buffer still counts as non-empty.
        assert!(board.set(4, 25, CellKind::Some(PieceType::I)));
        assert!(!board.is_empty(), "a buffer-zone cell makes it non-empty");

        // Clearing it back to None restores emptiness (a perfect clear).
        assert!(board.set(4, 25, CellKind::None));
        assert!(
            board.is_empty(),
            "clearing the only cell restores emptiness"
        );
    }

    #[test]
    fn horizontal_bounds_are_walls() {
        let board = Board::new(10, 20);

        assert_eq!(board.get_cell_kind(-1, 0), CellKind::Wall);
        assert_eq!(board.get_cell_kind(10, 0), CellKind::Wall);
    }

    #[test]
    fn negative_y_is_floor_collision() {
        let board = Board::new(10, 20);

        assert_eq!(board.get_cell_kind(0, -1), CellKind::Wall);
    }

    #[test]
    fn top_margin_accepts_hidden_spawn_cells() {
        let mut board = Board::with_top_margin(10, 20, 20);

        assert!(board.set(4, 25, CellKind::Some(PieceType::I)));
        assert_eq!(board.get_cell_kind(4, 25), CellKind::Some(PieceType::I));
    }

    #[test]
    fn clear_line_removes_row_and_drops_above_cells() {
        let mut board = Board::new(4, 4);
        fill_row(&mut board, 0, PieceType::I);
        assert!(board.set(1, 1, CellKind::Some(PieceType::T)));

        let cleared = board.clear_lines();

        assert_eq!(cleared, 1);
        assert_eq!(board.get_cell_kind(1, 0), CellKind::Some(PieceType::T));
        assert_eq!(board.get_cell_kind(1, 1), CellKind::None);
    }

    #[test]
    fn insert_garbage_shifts_stack_up_and_opens_a_hole() {
        let mut board = Board::new(4, 4);
        board.set(1, 0, CellKind::Some(PieceType::T)); // a cell on the floor

        let overflow = board.insert_garbage_lines(1, 2); // one row, hole at col 2

        assert!(!overflow);
        // The pre-existing cell rose by one row.
        assert_eq!(board.get_cell_kind(1, 1), CellKind::Some(PieceType::T));
        // New bottom row: full except the hole column.
        for x in 0..4 {
            let expected = if x == 2 {
                CellKind::None
            } else {
                CellKind::Garbage
            };
            assert_eq!(board.get_cell_kind(x, 0), expected, "col {x}");
        }
    }

    #[test]
    fn insert_garbage_reports_overflow_when_pushed_past_the_ceiling() {
        let mut board = Board::new(4, 4); // 4 rows, no buffer
        board.set(0, 3, CellKind::Some(PieceType::T)); // top visible row

        // Pushing up by 2 forces the top cell off the backing array.
        let overflow = board.insert_garbage_lines(2, 0);

        assert!(overflow);
        // No T cell survived anywhere; the bottom two rows are garbage, hole at col 0.
        assert!(
            board
                .cells()
                .iter()
                .all(|&(_, _, kind)| kind != CellKind::Some(PieceType::T))
        );
        assert_eq!(board.get_cell_kind(0, 0), CellKind::None);
        assert_eq!(board.get_cell_kind(1, 0), CellKind::Garbage);
        assert_eq!(board.get_cell_kind(1, 1), CellKind::Garbage);
    }

    #[test]
    fn clear_line_drops_cells_in_the_buffer_zone_above_visible_height() {
        // Regression: the compaction must cover the full backing array, not just
        // the visible height. A cell that locked in the buffer zone (y >= visible
        // height, legal per §16.4) above a cleared visible row has to fall like
        // any other — otherwise it is left floating above the skyline (§11.3).
        let mut board = Board::with_top_margin(4, 4, 4);
        fill_row(&mut board, 0, PieceType::I);
        assert!(board.set(0, 5, CellKind::Some(PieceType::S))); // buffer-zone cell

        assert_eq!(board.clear_lines(), 1);

        assert_eq!(board.get_cell_kind(0, 4), CellKind::Some(PieceType::S));
        assert_eq!(board.get_cell_kind(0, 5), CellKind::None);
    }

    #[test]
    fn buffer_zone_rows_clear_like_visible_rows() {
        // A row that fills entirely inside the hidden buffer clears like any
        // other (the guideline full-matrix rule).
        let mut board = Board::with_top_margin(4, 4, 4);
        fill_row(&mut board, 5, PieceType::I); // entirely in the buffer
        assert!(board.set(0, 6, CellKind::Some(PieceType::S)));

        assert_eq!(board.clear_lines(), 1);

        assert_eq!(
            board.get_cell_kind(0, 5),
            CellKind::Some(PieceType::S),
            "the cell above the cleared buffer row falls one row"
        );
        assert_eq!(board.get_cell_kind(0, 6), CellKind::None);
    }

    #[test]
    fn oversize_boards_are_rejected() {
        let r = std::panic::catch_unwind(|| Board::new(17, 20));
        assert!(r.is_err(), "width beyond the bit plane must panic");
        let r = std::panic::catch_unwind(|| Board::with_top_margin(10, 40, 30));
        assert!(r.is_err(), "backing rows beyond 64 must panic");
    }
}
