//! A faithful port of **Cold Clear 2**'s `freestyle` evaluator onto our
//! [`Evaluator`] trait, so CC2's *evaluation function* can play on our engine and
//! search. This makes a head-to-head against our own eval fair **by construction**
//! — same board, same garbage, same beam — unlike the TBP bridge, where CC2 has no
//! garbage message and a re-sync cripples it (see `cc2_baseline.rs`).
//!
//! Source: `cold-clear-2/src/bot/freestyle.rs` + weights from `src/default.json`.
//!
//! # Fidelity
//!
//! The board **Value** (holes, cell coveredness, tetris-well depth, height tiers,
//! row transitions, T-slot cutouts) is ported verbatim from CC2's column-bitboard
//! math. The per-move **Reward** (clears, spins, B2B, perfect clear, wasted-T) is
//! ported faithfully. `combo_attack` and `has_back_to_back` are now wired through
//! [`EvalContext`] (the running combo / B2B chain) and applied at their sites. One
//! term is still omitted:
//!
//! - `softdrop` — needs the soft-drop distance of the placement.
//!
//! B2B *reward* uses the same "credit every B2B-eligible clear" approximation as
//! [`LinearEvaluator`](super::LinearEvaluator) (the true chain lives in the search's
//! [`SearchState`](crate::ai::SearchState)). The T-slot `cutout_count` (needs
//! bag/reserve) is approximated as a single cutout. These gaps affect *offense
//! bookkeeping*, not the board-shaping that defines CC2's style.

use super::{EvalContext, Evaluator, Reward, Value};
use crate::engine::{Board, CellKind, LockOutcome, PieceType, TSpinKind};

/// Cold Clear 2 `freestyle` weights (`src/bot/freestyle.rs::Weights`), kept as `f32`
/// exactly as CC2 stores them.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Cc2Weights {
    pub cell_coveredness: f32,
    pub max_cell_covered_height: u32,
    pub holes: f32,
    pub row_transitions: f32,
    pub height: f32,
    pub height_upper_half: f32,
    pub height_upper_quarter: f32,
    pub tetris_well_depth: f32,
    pub tslot: [f32; 4],
    pub has_back_to_back: f32,
    pub wasted_t: f32,
    pub softdrop: f32,
    pub normal_clears: [f32; 5],
    pub mini_spin_clears: [f32; 3],
    pub spin_clears: [f32; 4],
    pub back_to_back_clear: f32,
    pub combo_attack: f32,
    pub perfect_clear: f32,
    pub perfect_clear_override: bool,
}

impl Cc2Weights {
    /// CC2's shipped `default.json` weights, verbatim.
    pub const DEFAULT: Cc2Weights = Cc2Weights {
        cell_coveredness: -0.2,
        max_cell_covered_height: 6,
        holes: -1.5,
        row_transitions: -0.2,
        height: -0.4,
        height_upper_half: -1.5,
        height_upper_quarter: -5.0,
        tetris_well_depth: 0.3,
        tslot: [0.1, 1.5, 2.0, 4.0],
        has_back_to_back: 0.5,
        wasted_t: -1.5,
        softdrop: -0.2,
        normal_clears: [0.0, -2.0, -1.5, -1.0, 3.5],
        mini_spin_clears: [0.0, -1.5, -1.0],
        spin_clears: [0.0, 1.0, 4.0, 6.0],
        back_to_back_clear: 1.0,
        combo_attack: 1.5,
        perfect_clear: 15.0,
        perfect_clear_override: true,
    };

    /// Number of tunable board-Value weights exposed for hillclimbing.
    pub const BOARD_PARAM_COUNT: usize = 11;

    /// The board-Value weights as a flat vector, for hillclimbing. Order:
    /// `[cell_coveredness, holes, row_transitions, height, height_upper_half,
    /// height_upper_quarter, tetris_well_depth, tslot0, tslot1, tslot2, tslot3]`.
    /// These define board shaping (the digging/T-slot style the downstack fitness
    /// measures); the reward weights and the integer/bool fields are left fixed.
    pub fn board_params(&self) -> [f32; Self::BOARD_PARAM_COUNT] {
        [
            self.cell_coveredness,
            self.holes,
            self.row_transitions,
            self.height,
            self.height_upper_half,
            self.height_upper_quarter,
            self.tetris_well_depth,
            self.tslot[0],
            self.tslot[1],
            self.tslot[2],
            self.tslot[3],
        ]
    }

    /// The attack profile: [`DEFAULT`](Self::DEFAULT) with the board weights replaced
    /// by the `cc2-app-climb` result (warm-climbed for attack-per-piece, 2026-06).
    /// This is the brain the shipped beam / best-first attack bots play; it lives here
    /// — beside [`DT20`](super::BoardWeights::DT20) and friends — so the game registry
    /// and the research harness read the same numbers.
    pub fn attack_tuned() -> Self {
        const ATTACK_BOARD_PARAMS: [f32; Cc2Weights::BOARD_PARAM_COUNT] = [
            -0.003_447_473,
            -1.5,
            -0.2,
            -0.362_030_36,
            -1.5,
            -5.0,
            0.347_263_3,
            0.1,
            1.5,
            4.465_080_7,
            4.0,
        ];
        Self::DEFAULT.with_board_params(&ATTACK_BOARD_PARAMS)
    }

    /// Return a copy with the board-Value weights replaced by `p` (see
    /// [`board_params`](Self::board_params) for the order).
    pub fn with_board_params(mut self, p: &[f32; Self::BOARD_PARAM_COUNT]) -> Self {
        self.cell_coveredness = p[0];
        self.holes = p[1];
        self.row_transitions = p[2];
        self.height = p[3];
        self.height_upper_half = p[4];
        self.height_upper_quarter = p[5];
        self.tetris_well_depth = p[6];
        self.tslot = [p[7], p[8], p[9], p[10]];
        self
    }
}

impl Default for Cc2Weights {
    fn default() -> Self {
        Self::DEFAULT
    }
}

/// Fixed-point scale: CC2 evaluates in `f32`, our [`Value`]/[`Reward`] are `i32`.
/// Both halves use the same factor, so `Value + Reward` still composes correctly.
const SCALE: f32 = 256.0;

/// The Cold Clear 2 freestyle evaluator, ported. Drop-in for
/// [`LinearEvaluator`](super::LinearEvaluator) behind `&dyn Evaluator`.
#[derive(Clone, Copy, Debug, Default)]
pub struct Cc2Evaluator {
    weights: Cc2Weights,
}

impl Cc2Evaluator {
    pub fn new(weights: Cc2Weights) -> Self {
        Self { weights }
    }

    /// CC2's board-Value terms over the column bitboard (post-clear board).
    fn board_value(&self, cols: &[u64]) -> f32 {
        let w = &self.weights;
        let mut eval = 0.0f32;

        // --- T-slot cutouts -------------------------------------------------
        // CC2 credits up to `cutout_count` (= bag-has-T + reserve-is-T + bag<=3)
        // speculative T-spin cutouts. The trait passes no bag/reserve, so we credit
        // ONE — enough to reward having a T-slot ready, which is the style signal.
        {
            // Detect on `cols` directly (read-only); allocate the mutated copy only
            // when a slot exists — most boards have none, so this skips a per-evaluate
            // heap allocation in the search's hot path.
            if let Some((sx, sy)) =
                well_known_tslot_left(cols).or_else(|| well_known_tslot_right(cols))
            {
                let mut after = cols.to_vec();
                place_t_south(&mut after, sx, sy);
                let clears = line_clears(&after).count_ones() as usize;
                eval += w.tslot[clears.min(3)];
            }
        }

        // --- holes + cell coveredness (one pass; share the underneath mask) ------
        let mut holes = 0u32;
        let mut coveredness = 0u32;
        for &c in cols {
            let height = 64 - c.leading_zeros();
            let hole_mask = !c & underneath_mask(height);
            holes += hole_mask.count_ones();
            let mut h = hole_mask;
            while h != 0 {
                let y = h.trailing_zeros();
                coveredness += (height - y).min(w.max_cell_covered_height);
                h &= !(1u64 << y);
            }
        }
        eval += w.holes * holes as f32;
        eval += w.cell_coveredness * coveredness as f32;

        // --- tetris well depth ---------------------------------------------
        let (well_col, well_height) = cols
            .iter()
            .enumerate()
            .map(|(i, &c)| (i, 64 - c.leading_zeros()))
            .min_by_key(|&(_, h)| h)
            .unwrap();
        let full_except_well = cols
            .iter()
            .enumerate()
            .filter(|&(i, _)| i != well_col)
            .map(|(_, &c)| c)
            .fold(!0u64, |a, b| a & b);
        // `well_height` can be 0..=64; a 64-bit shift is UB, so guard the full column.
        let well_depth = if well_height >= 64 {
            0
        } else {
            (full_except_well >> well_height).trailing_ones()
        };
        eval += well_depth as f32 * w.tetris_well_depth;

        // --- height tiers ---------------------------------------------------
        let highest = cols.iter().map(|&c| 64 - c.leading_zeros()).max().unwrap();
        eval += w.height * highest as f32;
        if highest > 10 {
            eval += w.height_upper_half * (highest - 10) as f32;
        }
        if highest > 15 {
            eval += w.height_upper_quarter * (highest - 15) as f32;
        }

        // --- row transitions (CC2's exact 64-bit formula) ------------------
        let mut row_transitions = (!0u64 ^ cols[0]).count_ones();
        row_transitions += (!0u64 ^ cols[cols.len() - 1]).count_ones();
        for cs in cols.windows(2) {
            row_transitions += (cs[0] ^ cs[1]).count_ones();
        }
        eval += row_transitions as f32 * w.row_transitions;

        eval
    }

    /// CC2's per-move Reward terms (clears, spins, B2B, perfect clear, wasted-T).
    fn placement_reward(
        &self,
        lock: &LockOutcome,
        is_empty: bool,
        t_spin: Option<TSpinKind>,
        combo: u32,
    ) -> f32 {
        let w = &self.weights;
        let lines = lock.cleared_rows.len();
        let perfect_clear = lines > 0 && is_empty;
        let mut reward = 0.0f32;

        if perfect_clear {
            reward += w.perfect_clear;
        }
        if !perfect_clear || !w.perfect_clear_override {
            // True chain (info.back_to_back) is unavailable here; like
            // LinearEvaluator, credit every B2B-eligible clear. The search's
            // SearchState carries the real chain. A zero-line spin is not a clear,
            // so it earns no clear bonus (`back_to_back_clear` is a *clear* weight).
            let b2b_eligible = matches!(
                (t_spin, lines),
                (Some(TSpinKind::Full | TSpinKind::Mini), 1..) | (None, 4)
            );
            if b2b_eligible {
                reward += w.back_to_back_clear;
            }
            reward += match t_spin {
                None => w.normal_clears[lines.min(4)],
                Some(TSpinKind::Mini) => w.mini_spin_clears[lines.min(2)],
                Some(TSpinKind::Full) => w.spin_clears[lines.min(3)],
            };
            // Combo attack (CC2): `combo_attack × floor((combo-1)/2)`, using the
            // search-path combo now supplied via EvalContext.
            reward += w.combo_attack * (combo.saturating_sub(1) / 2) as f32;
        }

        // wasted-T: a T placed without a T-spin double+ is "wasted".
        if placed_piece(lock) == Some(PieceType::T)
            && (lines < 2 || !matches!(t_spin, Some(TSpinKind::Full)))
        {
            reward += w.wasted_t;
        }
        // `softdrop` (per-move soft-drop distance) is still omitted — our movegen does
        // not model it; `has_back_to_back` is applied in `evaluate` from `ctx.b2b`.

        reward
    }

    /// Shared scoring core over the column bitboard: static board `Value` (+ the B2B
    /// bonus) and the per-move `Reward`. Both `evaluate` (engine `Board`) and
    /// `evaluate_cols` (search `BitBoard`) feed it their columns, so the two paths are
    /// bit-identical by construction. `is_empty` (perfect-clear) is read off the columns.
    fn score(
        &self,
        cols: &[u64],
        lock: &LockOutcome,
        t_spin: Option<TSpinKind>,
        ctx: EvalContext,
    ) -> (Value, Reward) {
        let mut value = self.board_value(cols);
        if ctx.b2b {
            value += self.weights.has_back_to_back;
        }
        let is_empty = cols.iter().all(|&c| c == 0);
        let reward = self.placement_reward(lock, is_empty, t_spin, ctx.combo);
        (
            Value((value * SCALE).round() as i32),
            Reward((reward * SCALE).round() as i32),
        )
    }
}

impl Evaluator for Cc2Evaluator {
    fn evaluate(
        &self,
        lock: &LockOutcome,
        board: &Board,
        t_spin: Option<TSpinKind>,
        ctx: EvalContext,
    ) -> (Value, Reward) {
        self.score(&board.column_bits(), lock, t_spin, ctx)
    }

    /// Fast bitboard path: the columns ARE the input — no `column_bits()` scan, no Array2D.
    fn evaluate_cols(
        &self,
        lock: &LockOutcome,
        board: crate::engine::ColumnView,
        t_spin: Option<TSpinKind>,
        ctx: EvalContext,
    ) -> (Value, Reward) {
        self.score(board.columns(), lock, t_spin, ctx)
    }
}

/// Bit mask of all rows strictly below `height` (`(1<<height)-1`, guarded so a
/// full 64-high column does not overflow the shift).
fn underneath_mask(height: u32) -> u64 {
    if height >= 64 {
        !0
    } else if height == 0 {
        0
    } else {
        (1u64 << height) - 1
    }
}

/// CC2's `Board::occupied`: out of bounds (walls / floor / above row 40) reads as
/// occupied.
fn occupied(cols: &[u64], x: i32, y: i32) -> bool {
    if x < 0 || x as usize >= cols.len() || !(0..40).contains(&y) {
        return true;
    }
    cols[x as usize] & (1u64 << y) != 0
}

/// Rows where every column is filled (`fold(!0, &)`).
fn line_clears(cols: &[u64]) -> u64 {
    cols.iter().fold(!0u64, |a, &b| a & b)
}

/// Set the four cells of a South-facing T at center `(x, y)` (CC2 cell offsets).
fn place_t_south(cols: &mut [u64], x: i32, y: i32) {
    for (dx, dy) in [(1, 0), (0, 0), (-1, 0), (0, -1)] {
        let (cx, cy) = (x + dx, y + dy);
        if cx >= 0 && (cx as usize) < cols.len() && (0..64).contains(&cy) {
            cols[cx as usize] |= 1u64 << cy;
        }
    }
}

/// CC2's `well_known_tslot_left`: a T-spin-double slot opening to the left, as a
/// South-T center `(x, y)`.
fn well_known_tslot_left(cols: &[u64]) -> Option<(i32, i32)> {
    for (x, win) in cols.windows(3).enumerate() {
        let y = (64 - win[0].leading_zeros()) as i32;
        if (64 - win[1].leading_zeros()) as i32 >= y {
            continue;
        }
        let xc = x as i32;
        if !occupied(cols, xc + 2, y - 1) {
            continue;
        }
        if occupied(cols, xc + 2, y) {
            continue;
        }
        if !occupied(cols, xc + 2, y + 1) {
            continue;
        }
        return Some((xc + 1, y));
    }
    None
}

/// CC2's `well_known_tslot_right`: the mirror of [`well_known_tslot_left`].
fn well_known_tslot_right(cols: &[u64]) -> Option<(i32, i32)> {
    for (x, win) in cols.windows(3).enumerate() {
        let y = (64 - win[2].leading_zeros()) as i32;
        if (64 - win[1].leading_zeros()) as i32 >= y {
            continue;
        }
        let xc = x as i32;
        if !occupied(cols, xc, y - 1) {
            continue;
        }
        if occupied(cols, xc, y) {
            continue;
        }
        if !occupied(cols, xc, y + 1) {
            continue;
        }
        return Some((xc + 1, y));
    }
    None
}

/// The piece type of a just-locked placement (all locked cells share it).
fn placed_piece(lock: &LockOutcome) -> Option<PieceType> {
    lock.cells_locked.first().and_then(|(_, _, k)| match k {
        CellKind::Some(pt) => Some(*pt),
        _ => None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::PieceType;

    fn no_clear_lock(piece: PieceType) -> LockOutcome {
        LockOutcome {
            cells_locked: vec![(0, 0, CellKind::Some(piece))],
            cleared_rows: Vec::new(),
            top_y_after_lock: Some(0),
        }
    }

    // `board_params`/`with_board_params` expose only the 11 board-Value weights (the
    // reward + scalar fields stay fixed); they are hand-indexed, so they must be exact
    // inverses or the CC2 hillclimb tunes the wrong slots. See the matching round-trip
    // tests for `BoardWeights`/`RewardWeights` in `weights.rs`.
    #[test]
    fn cc2_board_params_round_trip() {
        let w = Cc2Weights::DEFAULT;
        assert_eq!(w.board_params().len(), Cc2Weights::BOARD_PARAM_COUNT);
        assert_eq!(w.with_board_params(&w.board_params()), w);
    }

    #[test]
    fn cc2_with_board_params_is_positional() {
        // Distinct value per slot ⇒ a swapped index is caught; non-board fields
        // (e.g. the reward `perfect_clear`) must be left untouched.
        let p: [f32; Cc2Weights::BOARD_PARAM_COUNT] = std::array::from_fn(|i| (i as f32 + 1.0) * 2.0);
        let w = Cc2Weights::DEFAULT.with_board_params(&p);
        assert_eq!(w.board_params(), p);
        assert_eq!(w.perfect_clear, Cc2Weights::DEFAULT.perfect_clear);
    }

    #[test]
    fn empty_board_is_deterministic_and_no_reward() {
        // No cells → no holes, no height, well depth 0, but row_transitions counts
        // the empty edge columns (CC2's exact behaviour). Just assert it runs and is
        // finite / deterministic.
        let eval = Cc2Evaluator::default();
        let board = Board::new(10, 20);
        let (v1, r1) = eval.evaluate(&no_clear_lock(PieceType::O), &board, None, EvalContext::default());
        let (v2, r2) = eval.evaluate(&no_clear_lock(PieceType::O), &board, None, EvalContext::default());
        assert_eq!((v1, r1), (v2, r2), "evaluation is deterministic");
        assert_eq!(r1, Reward(0), "a no-clear non-T lock earns no reward");
    }

    #[test]
    fn holes_are_penalised() {
        // A column with a covered hole must score worse than a clean one.
        let eval = Cc2Evaluator::default();
        let mut clean = Board::new(10, 20);
        clean.set(0, 0, CellKind::Some(PieceType::I));
        let mut holey = Board::new(10, 20);
        holey.set(0, 1, CellKind::Some(PieceType::I)); // cell at y=1 with empty y=0 below

        let (clean_v, _) = eval.evaluate(&no_clear_lock(PieceType::O), &clean, None, EvalContext::default());
        let (holey_v, _) = eval.evaluate(&no_clear_lock(PieceType::O), &holey, None, EvalContext::default());
        assert!(
            holey_v < clean_v,
            "a covered hole must reduce Value: holey {holey_v:?} vs clean {clean_v:?}"
        );
    }

    #[test]
    fn tetris_clear_outrewards_single() {
        let eval = Cc2Evaluator::default();
        let mut board = Board::new(10, 20);
        board.set(0, 0, CellKind::Some(PieceType::O)); // not a perfect clear

        let tetris = LockOutcome {
            cells_locked: vec![(0, 0, CellKind::Some(PieceType::I))],
            cleared_rows: vec![0, 1, 2, 3],
            top_y_after_lock: None,
        };
        let single = LockOutcome {
            cells_locked: vec![(0, 0, CellKind::Some(PieceType::I))],
            cleared_rows: vec![0],
            top_y_after_lock: None,
        };
        let (_, tetris_r) = eval.evaluate(&tetris, &board, None, EvalContext::default());
        let (_, single_r) = eval.evaluate(&single, &board, None, EvalContext::default());
        // normal_clears[4]=3.5 (+ b2b 1.0) vs normal_clears[1]=-2.0 → Tetris wins.
        assert!(
            tetris_r > single_r,
            "Tetris {tetris_r:?} must out-reward a single {single_r:?}"
        );
        assert!(tetris_r.0 > 0, "a Tetris is a positive reward");
        assert!(single_r.0 < 0, "a single is penalised (CC2 normal_clears[1] < 0)");
    }

    #[test]
    fn wasted_t_is_penalised() {
        let eval = Cc2Evaluator::default();
        let board = Board::new(10, 20);
        // A T that clears nothing: wasted.
        let (_, r) = eval.evaluate(&no_clear_lock(PieceType::T), &board, None, EvalContext::default());
        // wasted_t = -1.5 (× SCALE).
        assert_eq!(r, Reward((-1.5 * SCALE).round() as i32));
    }

    #[test]
    fn object_safe_and_finite() {
        let eval = Cc2Evaluator::default();
        let dyn_eval: &dyn Evaluator = &eval;
        let mut board = Board::new(10, 20);
        for x in 0..9 {
            board.set(x, 0, CellKind::Some(PieceType::I));
        }
        let (v, _r) = dyn_eval.evaluate(&no_clear_lock(PieceType::O), &board, None, EvalContext::default());
        assert!(v.0.abs() < 1_000_000, "value stays in a sane range: {v:?}");
    }

    #[test]
    fn combo_attack_scales_with_chain() {
        // The CC2 reward adds combo_attack * floor((combo - 1) / 2) to a clearing
        // placement, using the search-path combo the planners supply via EvalContext.
        // So combo 0/1/2 add nothing and the reward steps up at combo 3, 5, ... (the
        // staircase is floor((combo-1)/2), NOT floor(combo/2)). This is the only place
        // EvalContext.combo reaches the shipped CC2 evaluator, and was previously
        // untested (every other cc2 test uses EvalContext::default()).
        let eval = Cc2Evaluator::default();
        let mut board = Board::new(10, 20);
        board.set(0, 0, CellKind::Some(PieceType::O)); // not a perfect clear
        let clear = LockOutcome {
            cells_locked: vec![(0, 0, CellKind::Some(PieceType::I))],
            cleared_rows: vec![0],
            top_y_after_lock: None,
        };
        let reward = |combo: u32| eval.evaluate(&clear, &board, None, EvalContext { combo, b2b: false }).1 .0;

        // floor((combo - 1) / 2) == 0 for combo 0, 1, 2 (and combo 0 must not underflow
        // the u32 subtraction).
        assert_eq!(reward(0), reward(1), "combo 0 and 1 add the same combo attack (0)");
        assert_eq!(reward(1), reward(2), "combo 2 still adds 0 -- floor((combo-1)/2), not combo/2");
        // Steps up at 3 (floor(2/2) = 1), flat at 4, steps again at 5 (floor(4/2) = 2).
        assert!(reward(3) > reward(2), "combo 3 adds the first combo-attack step");
        assert_eq!(reward(3), reward(4), "the staircase is flat between odd combos");
        assert!(reward(5) > reward(3), "combo 5 adds a second step");
    }

    #[test]
    fn back_to_back_adds_to_value() {
        // ctx.b2b adds has_back_to_back to the static Value (not the Reward); combo does
        // not affect Value. Both are the planner-supplied chain reaching CC2 -- the
        // ctx.b2b branch was never taken under test before.
        let eval = Cc2Evaluator::default();
        let mut board = Board::new(10, 20);
        board.set(0, 0, CellKind::Some(PieceType::O));
        let lock = no_clear_lock(PieceType::O);
        let value = |ctx| eval.evaluate(&lock, &board, None, ctx).0 .0;

        let base = value(EvalContext { combo: 0, b2b: false });
        assert!(value(EvalContext { combo: 0, b2b: true }) > base, "b2b raises Value");
        assert_eq!(value(EvalContext { combo: 5, b2b: false }), base, "combo does not affect Value");
    }

    #[test]
    fn no_clear_spin_earns_no_b2b_bonus() {
        // `back_to_back_clear` is a *clear* weight: a T spun into a slot without
        // clearing must earn only the wasted-T penalty, never the B2B clear bonus.
        let eval = Cc2Evaluator::default();
        let board = Board::new(10, 20);
        let (_, r) = eval.evaluate(
            &no_clear_lock(PieceType::T),
            &board,
            Some(TSpinKind::Full),
            EvalContext::default(),
        );
        assert_eq!(r, Reward((-1.5 * SCALE).round() as i32), "wasted_t only");
    }

    #[test]
    fn cc2_evaluate_cols_matches_evaluate() {
        // The bit-identical contract on the override (the per-impl differential the
        // trait doc promises), exercised on a board with holes, a well, and a T-slot.
        let eval = Cc2Evaluator::default();
        let mut board = Board::new(10, 20);
        for x in 0..9 {
            board.set(x, 0, CellKind::Some(PieceType::I));
        }
        board.set(0, 2, CellKind::Some(PieceType::O)); // covered hole at (0, 1)
        let clear_lock = LockOutcome {
            cells_locked: vec![(9, 0, CellKind::Some(PieceType::T))],
            cleared_rows: vec![0],
            top_y_after_lock: Some(2),
        };

        for (lock, t_spin) in [
            (no_clear_lock(PieceType::T), None),
            (clear_lock, Some(TSpinKind::Full)),
        ] {
            let ctx = EvalContext { combo: 3, b2b: true };
            let bb = crate::engine::BitBoard::from_board(&board);
            assert_eq!(
                eval.evaluate_cols(&lock, bb.view(), t_spin, ctx),
                eval.evaluate(&lock, &board, t_spin, ctx),
            );
        }
    }

    /// Columns from filled-cell lists, for driving the T-slot detectors directly.
    fn cols_from(cells: &[&[u32]]) -> Vec<u64> {
        cells
            .iter()
            .map(|rows| rows.iter().fold(0u64, |c, &y| c | (1u64 << y)))
            .collect()
    }

    #[test]
    fn tslot_left_detects_a_tsd_notch() {
        // col0 filled to height 2 (y = surface 2), col1 lower (the slot), col2 shaped
        // occupied(y-1) / open(y) / occupied(y+1) — the left-opening TSD notch the
        // detector returns as a South-T center at (1, 2).
        let cols = cols_from(&[&[0, 1], &[0], &[0, 1, 3]]);
        assert_eq!(well_known_tslot_left(&cols), Some((1, 2)));
        assert_eq!(well_known_tslot_right(&cols), None);
    }

    #[test]
    fn tslot_right_detects_the_mirror_notch() {
        let cols = cols_from(&[&[0, 1, 3], &[0], &[0, 1]]);
        assert_eq!(well_known_tslot_right(&cols), Some((1, 2)));
        assert_eq!(well_known_tslot_left(&cols), None);
    }

    #[test]
    fn tslot_detectors_ignore_a_flat_or_plain_well_board() {
        // Flat surface: no notch anywhere.
        assert_eq!(well_known_tslot_left(&cols_from(&[&[0], &[0], &[0]])), None);
        assert_eq!(well_known_tslot_right(&cols_from(&[&[0], &[0], &[0]])), None);
        // A plain 1-wide well with no lid is a Tetris well, not a T-slot.
        let well = cols_from(&[&[0, 1], &[], &[0, 1]]);
        assert_eq!(well_known_tslot_left(&well), None);
        assert_eq!(well_known_tslot_right(&well), None);
    }
}
