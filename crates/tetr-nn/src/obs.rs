//! The observation: what the net sees, defined exactly once.
//!
//! One encoding for one net. An observation is **two occupancy planes** (own
//! board and the opponent's, for the siamese conv tower) plus an
//! **85-dim feature vector**:
//!
//! | offset | len | segment |
//! |---|---|---|
//! | `0..7`   | 7  | active piece one-hot |
//! | `7..14`  | 7  | hold one-hot |
//! | `14..15` | 1  | hold empty flag |
//! | `15..50` | 35 | queue slots 0..5, one-hot each |
//! | `50..57` | 7  | bag draw-set multi-hot |
//! | `57..58` | 1  | combo |
//! | `58..59` | 1  | back-to-back active |
//! | `59..60` | 1  | pending garbage, total lines |
//! | `60..70` | 10 | pending hole lines per column |
//! | `70..71` | 1  | opp combo |
//! | `71..72` | 1  | opp back-to-back |
//! | `72..73` | 1  | opp pending total |
//! | `73..83` | 10 | opp pending hole lines per column |
//! | `83..84` | 1  | plies-until-rain ÷ rain period |
//! | `84..85` | 1  | fraction of the ply cap consumed |
//!
//! Piece indices use **declaration order** `I,J,L,O,S,T,Z` — the same order as
//! [`PieceType::all`] and the bag bitset, *not* the colour `render_index`.
//!
//! Only the own half (`0..70` + the own plane) varies per search leaf; the opp
//! half is a per-decision [`OppCtx`], frozen when the decision starts — the
//! same freezing contract pending garbage already uses.
//!
//! Segment offsets, packing, and the hash live here and nowhere else: the
//! shard writer stores these exact bytes ([`pack_plane`] + the feature f32s),
//! so a trainer reads *what was served* rather than re-deriving it.

use tetr_core::ai::SearchState;
use tetr_core::engine::PieceType;

/// Board plane height (rows): the engine's full backing board for the default
/// config (visible 20 + buffer 20). The whole height is encoded so a tall live
/// state (legal up to `y = 39`) is never silently truncated.
pub const BOARD_H: usize = 40;
/// Board plane width — the guideline 10-wide field.
pub const BOARD_W: usize = 10;
/// Flattened plane length.
pub const BOARD_LEN: usize = BOARD_H * BOARD_W;
/// A plane bit-packed for storage: `BOARD_LEN` bits.
pub const PACKED_PLANE: usize = BOARD_LEN / 8;

/// Queue slots the encoder reads (the revealed Next preview the net sees).
pub const QUEUE_SLOTS: usize = 5;

// Feature segment offsets (the module table). One definition; the tests pin
// contiguity.
const F_ACTIVE: usize = 0;
const F_HOLD: usize = F_ACTIVE + PieceType::LEN;
const F_HOLD_EMPTY: usize = F_HOLD + PieceType::LEN;
const F_QUEUE: usize = F_HOLD_EMPTY + 1;
const F_BAG: usize = F_QUEUE + QUEUE_SLOTS * PieceType::LEN;
const F_COMBO: usize = F_BAG + PieceType::LEN;
const F_B2B: usize = F_COMBO + 1;
const F_PENDING_TOTAL: usize = F_B2B + 1;
const F_PENDING_COLS: usize = F_PENDING_TOTAL + 1;
/// Length of the own segment (everything a solo state determines).
pub const OWN_FEATURES: usize = F_PENDING_COLS + BOARD_W;
const F_OPP_COMBO: usize = OWN_FEATURES;
const F_OPP_B2B: usize = F_OPP_COMBO + 1;
const F_OPP_PENDING_TOTAL: usize = F_OPP_B2B + 1;
const F_OPP_PENDING_COLS: usize = F_OPP_PENDING_TOTAL + 1;
const F_RAIN_FRAC: usize = F_OPP_PENDING_COLS + BOARD_W;
const F_CAP_FRAC: usize = F_RAIN_FRAC + 1;
/// Full feature-vector length (own 70 + opp 13 + clock 2).
pub const FEATURE_LEN: usize = F_CAP_FRAC + 1;

/// A stable `0..7` piece index in declaration order `I,J,L,O,S,T,Z` — identical
/// to [`PieceType::all`]'s order and the bag bit layout, so every one-hot and
/// the bag multi-hot share one indexing. Deliberately NOT `render_index` (the
/// colour order), which would silently permute the input channels.
pub fn piece_index(piece: PieceType) -> usize {
    match piece {
        PieceType::I => 0,
        PieceType::J => 1,
        PieceType::L => 2,
        PieceType::O => 3,
        PieceType::S => 4,
        PieceType::T => 5,
        PieceType::Z => 6,
    }
}

/// The opponent context a decision is evaluated under: the opposing seat after
/// its last completed lock, captured once per own decision and pinned for that
/// decision's whole search (plus the driver's venue clock).
#[derive(Clone, Debug, PartialEq)]
pub struct OppCtx {
    /// Opponent occupancy plane, same layout as an own plane (`y = 0` floor;
    /// locked cells only — a falling piece may still move, so it is not real).
    pub board: [f32; BOARD_LEN],
    /// Opponent combo counter.
    pub combo: u32,
    /// Opponent Back-to-Back chain active.
    pub b2b: bool,
    /// Opponent's incoming garbage, `(lines, hole_col)` oldest first.
    pub pending: Vec<(u32, usize)>,
    /// Plies until the next rain drop ÷ rain period (`0.0` when rain is off).
    pub rain_frac: f32,
    /// Fraction of the venue's hard ply cap already consumed.
    pub cap_frac: f32,
}

impl Default for OppCtx {
    /// A neutral context (empty board, no chain, no pressure, clock at zero) —
    /// the solo/bring-up stand-in, NOT a valid versus observation.
    fn default() -> Self {
        Self {
            board: [0.0; BOARD_LEN],
            combo: 0,
            b2b: false,
            pending: Vec::new(),
            rain_frac: 0.0,
            cap_frac: 0.0,
        }
    }
}

impl OppCtx {
    /// Build from the opposing engine's snapshot + the driver's venue clock.
    pub fn from_snapshot(
        snap: &tetr_core::engine::EngineSnapshot,
        rain_frac: f32,
        cap_frac: f32,
    ) -> Self {
        let mut board = [0.0f32; BOARD_LEN];
        for cell in &snap.board_cells {
            if cell.x >= 0
                && (cell.x as usize) < BOARD_W
                && cell.y >= 0
                && (cell.y as usize) < BOARD_H
            {
                board[cell.y as usize * BOARD_W + cell.x as usize] = 1.0;
            }
        }
        Self {
            board,
            combo: snap.combo,
            b2b: snap.back_to_back_active,
            pending: snap
                .pending_garbage
                .iter()
                .map(|b| (b.lines, b.hole_col))
                .collect(),
            rain_frac,
            cap_frac,
        }
    }
}

/// One encoded observation: the two conv planes + the feature vector. Per
/// search leaf only the own half varies; `opp_board` is the decision's frozen
/// [`OppCtx`] plane (a serving layer caches its embedding per decision).
#[derive(Clone, Debug, PartialEq)]
pub struct Obs {
    /// Own occupancy plane, `board[y * BOARD_W + x]`, `y = 0` floor.
    pub own_board: [f32; BOARD_LEN],
    /// The decision's frozen opponent plane.
    pub opp_board: [f32; BOARD_LEN],
    /// The feature vector — see the module table.
    pub features: [f32; FEATURE_LEN],
}

/// Encode a child afterstate under its decision's frozen opponent context.
///
/// Reads only public state: board occupancy, active/hold/queue, the bag
/// draw-set, combo/B2B, the pending-garbage queue — every batch of it, however
/// long the queue has spilled (what the net sees is never truncated; the shard
/// stores these encoded values, so it can't disagree).
pub fn encode(state: &SearchState, opp: &OppCtx) -> Obs {
    let mut own_board = [0.0f32; BOARD_LEN];
    // `occupied` reads false out of bounds, so a narrower test board or the
    // clipped top rows simply stay zero.
    for y in 0..BOARD_H {
        for x in 0..BOARD_W {
            if state.board.occupied(x as isize, y as isize) {
                own_board[y * BOARD_W + x] = 1.0;
            }
        }
    }

    let mut f = [0.0f32; FEATURE_LEN];
    f[F_ACTIVE + piece_index(state.active.piece_type())] = 1.0;
    match state.hold {
        Some(pt) => f[F_HOLD + piece_index(pt)] = 1.0,
        None => f[F_HOLD_EMPTY] = 1.0,
    }
    for (slot, &pt) in state.queue.iter().take(QUEUE_SLOTS).enumerate() {
        f[F_QUEUE + slot * PieceType::LEN + piece_index(pt)] = 1.0;
    }
    for pt in PieceType::all() {
        if state.bag.contains(pt) {
            f[F_BAG + piece_index(pt)] = 1.0;
        }
    }
    f[F_COMBO] = state.combo as f32;
    f[F_B2B] = if state.b2b { 1.0 } else { 0.0 };
    for batch in state.pending.iter() {
        f[F_PENDING_TOTAL] += batch.lines as f32;
        if batch.hole_col < BOARD_W {
            f[F_PENDING_COLS + batch.hole_col] += batch.lines as f32;
        }
    }

    f[F_OPP_COMBO] = opp.combo as f32;
    f[F_OPP_B2B] = if opp.b2b { 1.0 } else { 0.0 };
    for &(lines, hole_col) in &opp.pending {
        f[F_OPP_PENDING_TOTAL] += lines as f32;
        if hole_col < BOARD_W {
            f[F_OPP_PENDING_COLS + hole_col] += lines as f32;
        }
    }
    f[F_RAIN_FRAC] = opp.rain_frac;
    f[F_CAP_FRAC] = opp.cap_frac;

    Obs {
        own_board,
        opp_board: opp.board,
        features: f,
    }
}

/// Bit-pack an occupancy plane for storage (row-major, LSB-first within each
/// byte). Planes are exactly 0.0/1.0 by construction, so packing is lossless —
/// which is what lets a shard store *served* planes compactly.
pub fn pack_plane(plane: &[f32; BOARD_LEN]) -> [u8; PACKED_PLANE] {
    let mut out = [0u8; PACKED_PLANE];
    for (i, &v) in plane.iter().enumerate() {
        if v != 0.0 {
            out[i / 8] |= 1 << (i % 8); // LSB-first: cell i → bit (i%8) of byte i/8
        }
    }
    out
}

/// FNV-1a over bytes — the crate's ONE hash implementation (a hand-inlined
/// copy of this pattern once shipped a wrong prime; it exists exactly once
/// now). Used for shard payload checksums.
pub fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

#[cfg(test)]
mod tests {
    use super::*;
    use tetr_core::engine::{Engine, EngineConfig, InputFrame};

    /// A real, spawned `SearchState` via the public production path (Engine →
    /// snapshot → from_snapshot), so tests pin the true serve distribution.
    fn spawned(seed: u64) -> SearchState {
        let mut engine = Engine::new(EngineConfig::default(), seed);
        engine.step(InputFrame::default());
        SearchState::from_snapshot(&engine.snapshot()).expect("active piece present")
    }

    #[test]
    fn segments_are_contiguous_and_total_85() {
        assert_eq!(F_ACTIVE, 0);
        assert_eq!(F_HOLD, 7);
        assert_eq!(F_HOLD_EMPTY, 14);
        assert_eq!(F_QUEUE, 15);
        assert_eq!(F_BAG, 50);
        assert_eq!(F_COMBO, 57);
        assert_eq!(F_B2B, 58);
        assert_eq!(F_PENDING_TOTAL, 59);
        assert_eq!(F_PENDING_COLS, 60);
        assert_eq!(OWN_FEATURES, 70);
        assert_eq!(F_OPP_COMBO, 70);
        assert_eq!(F_RAIN_FRAC, 83);
        assert_eq!(F_CAP_FRAC, 84);
        assert_eq!(FEATURE_LEN, 85);
    }

    #[test]
    fn encode_reads_a_real_spawn() {
        let state = spawned(7);
        let obs = encode(&state, &OppCtx::default());
        // Exactly one active one-hot.
        let active: f32 = obs.features[F_ACTIVE..F_HOLD].iter().sum();
        assert_eq!(active, 1.0);
        // Fresh spawn: hold empty, five queue one-hots.
        assert_eq!(obs.features[F_HOLD_EMPTY], 1.0);
        let queue: f32 = obs.features[F_QUEUE..F_BAG].iter().sum();
        assert_eq!(queue, QUEUE_SLOTS as f32);
        // A fresh board has no occupied cells.
        assert!(obs.own_board.iter().all(|&v| v == 0.0));
        // Neutral opp: the whole opp/clock tail is zero.
        assert!(obs.features[OWN_FEATURES..].iter().all(|&v| v == 0.0));
    }

    #[test]
    fn opp_tail_lands_where_the_table_says() {
        let state = spawned(3);
        let opp = OppCtx {
            combo: 2,
            b2b: true,
            pending: vec![(3, 4), (2, 4), (1, 9)],
            rain_frac: 0.25,
            cap_frac: 0.5,
            ..OppCtx::default()
        };
        let obs = encode(&state, &opp);
        assert_eq!(obs.features[F_OPP_COMBO], 2.0);
        assert_eq!(obs.features[F_OPP_B2B], 1.0);
        assert_eq!(obs.features[F_OPP_PENDING_TOTAL], 6.0);
        assert_eq!(obs.features[F_OPP_PENDING_COLS + 4], 5.0);
        assert_eq!(obs.features[F_OPP_PENDING_COLS + 9], 1.0);
        assert_eq!(obs.features[F_RAIN_FRAC], 0.25);
        assert_eq!(obs.features[F_CAP_FRAC], 0.5);
    }

    #[test]
    fn pack_is_lsb_first() {
        // The exact byte layout is the cross-language contract: the Python
        // trainer reads these planes with np.unpackbits(bitorder="little").
        // Cells 0 and 9 set → byte 0 bit 0, byte 1 bit 1.
        let mut plane = [0.0f32; BOARD_LEN];
        plane[0] = 1.0;
        plane[9] = 1.0;
        let packed = pack_plane(&plane);
        assert_eq!(packed[0], 0b0000_0001);
        assert_eq!(packed[1], 0b0000_0010);
        assert!(packed[2..].iter().all(|&b| b == 0));
    }

    #[test]
    fn fnv_matches_the_standard_vectors() {
        // FNV-1a 64 test vectors (the exact constants a typo once broke).
        assert_eq!(fnv1a(b""), 0xcbf2_9ce4_8422_2325);
        assert_eq!(fnv1a(b"a"), 0xaf63_dc4c_8601_ec8c);
        assert_eq!(fnv1a(b"foobar"), 0x85944171f73967e8);
    }
}
