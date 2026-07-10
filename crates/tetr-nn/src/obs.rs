//! The observation: what the net sees, defined exactly once.
//!
//! An observation is **one occupancy plane** (your own board) plus a
//! **70-dim feature vector**:
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
//!
//! Bots are blind to the opponent's board by rule, and no opponent context was
//! ever wired into a production path — so there is no opponent input. (The
//! previous two-board observation carried an all-zero opponent plane and
//! constant-zero opponent features through every corpus and measurement; its
//! whitening interaction caused a real training defect. History in
//! `wayfinder/leapfrog/archive/`.)
//!
//! Piece indices use **declaration order** `I,J,L,O,S,T,Z` — the same order as
//! [`PieceType::all`] and the bag bitset, *not* the colour `render_index`.
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
/// Full feature-vector length.
pub const FEATURE_LEN: usize = F_PENDING_COLS + BOARD_W;

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

/// One encoded observation: the conv plane + the feature vector.
#[derive(Clone, Debug, PartialEq)]
pub struct Obs {
    /// Own occupancy plane, `board[y * BOARD_W + x]`, `y = 0` floor.
    pub board: [f32; BOARD_LEN],
    /// The feature vector — see the module table.
    pub features: [f32; FEATURE_LEN],
}

/// Encode a state.
///
/// Reads only public state: board occupancy, active/hold/queue, the bag
/// draw-set, combo/B2B, the pending-garbage queue — every batch of it, however
/// long the queue has spilled (what the net sees is never truncated; the shard
/// stores these encoded values, so it can't disagree).
pub fn encode(state: &SearchState) -> Obs {
    let mut board = [0.0f32; BOARD_LEN];
    // `occupied` reads false out of bounds, so a narrower test board or the
    // clipped top rows simply stay zero.
    for y in 0..BOARD_H {
        for x in 0..BOARD_W {
            if state.board.occupied(x as isize, y as isize) {
                board[y * BOARD_W + x] = 1.0;
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

    Obs { board, features: f }
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
    fn segments_are_contiguous_and_total_70() {
        assert_eq!(F_ACTIVE, 0);
        assert_eq!(F_HOLD, 7);
        assert_eq!(F_HOLD_EMPTY, 14);
        assert_eq!(F_QUEUE, 15);
        assert_eq!(F_BAG, 50);
        assert_eq!(F_COMBO, 57);
        assert_eq!(F_B2B, 58);
        assert_eq!(F_PENDING_TOTAL, 59);
        assert_eq!(F_PENDING_COLS, 60);
        assert_eq!(FEATURE_LEN, 70);
    }

    #[test]
    fn encode_reads_a_real_spawn() {
        let state = spawned(7);
        let obs = encode(&state);
        // Exactly one active one-hot.
        let active: f32 = obs.features[F_ACTIVE..F_HOLD].iter().sum();
        assert_eq!(active, 1.0);
        // Fresh spawn: hold empty, five queue one-hots.
        assert_eq!(obs.features[F_HOLD_EMPTY], 1.0);
        let queue: f32 = obs.features[F_QUEUE..F_BAG].iter().sum();
        assert_eq!(queue, QUEUE_SLOTS as f32);
        // A fresh board has no occupied cells.
        assert!(obs.board.iter().all(|&v| v == 0.0));
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
