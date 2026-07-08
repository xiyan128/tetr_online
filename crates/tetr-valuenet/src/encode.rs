//! The single-board observation the deployed value net sees.
//!
//! One occupancy plane (the player's own board) + a 70-dim feature vector. This
//! is a *frozen* encoding: it must reproduce exactly what the shipped weights
//! (`conv_rb1` et al.) were trained on, so the layout below is a contract, not a
//! design surface.
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
//! Piece indices use declaration order `I,J,L,O,S,T,Z` (the bag bit layout),
//! not the colour render order. (This is the same layout the research crate's
//! two-board encoder carries as its first 70 "own" features — kept separate
//! here so the deploy net's frozen encoding can't drift when research evolves.)

use tetr_core::ai::SearchState;
use tetr_core::engine::PieceType;

/// Board plane height (the engine's full backing board for the default config).
pub const BOARD_H: usize = 40;
/// Board plane width — the guideline 10-wide field.
pub const BOARD_W: usize = 10;
/// Flattened plane length.
pub const BOARD_LEN: usize = BOARD_H * BOARD_W;
/// Queue slots the encoder reads.
pub const QUEUE_SLOTS: usize = 5;

const F_ACTIVE: usize = 0;
const F_HOLD: usize = F_ACTIVE + PieceType::LEN;
const F_HOLD_EMPTY: usize = F_HOLD + PieceType::LEN;
const F_QUEUE: usize = F_HOLD_EMPTY + 1;
const F_BAG: usize = F_QUEUE + QUEUE_SLOTS * PieceType::LEN;
const F_COMBO: usize = F_BAG + PieceType::LEN;
const F_B2B: usize = F_COMBO + 1;
const F_PENDING_TOTAL: usize = F_B2B + 1;
const F_PENDING_COLS: usize = F_PENDING_TOTAL + 1;
/// Feature-vector length (70).
pub const FEATURE_LEN: usize = F_PENDING_COLS + BOARD_W;

/// A stable `0..7` piece index in declaration order `I,J,L,O,S,T,Z`.
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

/// One encoded position: the board plane + the feature vector.
#[derive(Clone, Debug, PartialEq)]
pub struct Obs {
    /// Occupancy plane, `board[y * BOARD_W + x]`, `y = 0` floor.
    pub board: [f32; BOARD_LEN],
    /// The feature vector — see the module table.
    pub features: [f32; FEATURE_LEN],
}

impl Default for Obs {
    fn default() -> Self {
        Self {
            board: [0.0; BOARD_LEN],
            features: [0.0; FEATURE_LEN],
        }
    }
}

/// Encode a [`SearchState`] into the net's observation. Reads only public state
/// (board occupancy, active/hold/queue, the bag draw-set, combo/B2B, and the
/// pending-garbage queue).
pub fn encode(state: &SearchState) -> Obs {
    let mut obs = Obs::default();
    // `occupied` reads false out of bounds, so a narrower test board or the
    // clipped top rows simply stay zero.
    for y in 0..BOARD_H {
        for x in 0..BOARD_W {
            if state.board.occupied(x as isize, y as isize) {
                obs.board[y * BOARD_W + x] = 1.0;
            }
        }
    }

    let f = &mut obs.features;
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
    obs
}

#[cfg(test)]
mod tests {
    use super::*;
    use tetr_core::engine::{Engine, EngineConfig, InputFrame};

    fn spawned(seed: u64) -> SearchState {
        let mut engine = Engine::new(EngineConfig::default(), seed);
        engine.step(InputFrame::default());
        SearchState::from_snapshot(&engine.snapshot()).expect("active piece present")
    }

    #[test]
    fn offsets_are_contiguous_and_total_70() {
        assert_eq!((F_ACTIVE, F_HOLD, F_HOLD_EMPTY, F_QUEUE), (0, 7, 14, 15));
        assert_eq!((F_BAG, F_COMBO, F_B2B), (50, 57, 58));
        assert_eq!((F_PENDING_TOTAL, F_PENDING_COLS, FEATURE_LEN), (59, 60, 70));
    }

    #[test]
    fn encodes_a_real_spawn() {
        let obs = encode(&spawned(7));
        assert_eq!(obs.features[F_ACTIVE..F_HOLD].iter().sum::<f32>(), 1.0);
        assert_eq!(obs.features[F_HOLD_EMPTY], 1.0);
        assert_eq!(
            obs.features[F_QUEUE..F_BAG].iter().sum::<f32>(),
            QUEUE_SLOTS as f32
        );
        assert!(obs.board.iter().all(|&v| v == 0.0));
    }
}
