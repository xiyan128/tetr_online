//! Tunable weight vectors for the linear board evaluator.
//!
//! The evaluator scores a board as a *linear weighted sum* of hand-engineered
//! features (the Dellacherie / BCTS family — see [`super::features`]). Splitting
//! the numbers out here keeps the feature math (what is measured) separate from
//! the policy (how much each measurement is worth), so weights stay tunable
//! without touching the extraction code.
//!
//! # Two weight groups, mirroring the `(Value, Reward)` seam
//!
//! [`Weights`] holds two conceptually distinct groups (Cold Clear's
//! `transient`/`acc` split, finding [3]):
//!
//! - **Board weights** — applied to the *static* board features to produce a
//!   [`Value`](super::Value): the quality of a resting position independent of how
//!   it was reached (holes, transitions, wells, …).
//! - **Reward weights** — applied to the *per-move* payoff to produce a
//!   [`Reward`](super::Reward): what the placement just earned (line clears,
//!   T-spins, Back-to-Back). Rewards sum along a search path.
//!
//! # The DT-20 default is an *initialization*, not gospel
//!
//! [`Weights::DT20`] seeds the board weights with the Dellacherie–Thiery 9-feature
//! CBMPI-optimized vector for the 10×20 board (finding [2]). **Caveat (research
//! §74.5):** DT-20 is a *learned, maximize-convention* policy — e.g. its `holes`
//! weight is *positive* (`+2.03`) because higher score = better there. This crate
//! evaluates with a *higher-Value-is-better* convention too, so the signs are kept
//! verbatim, but the published feature *semantics* differ subtly from ours
//! (landing height, eroded cells). Treat every number as a starting point for
//! tuning, never as a guideline-Tetris-correct constant.
//!
//! Reward weights are seeded from Cold Clear's master config (findings [4],[5]),
//! which deliberately rewards Tetrises / T-spins / B2B far above low clears and
//! *penalizes* singles/doubles to force downstacking and B2B-chain preservation.

/// Board-quality weights: one coefficient per static board feature.
///
/// Field order intentionally matches [`BoardFeatures`](super::features::BoardFeatures)
/// so [`BoardWeights::dot`] can pair them up positionally. The first six are
/// Dellacherie's canonical set; [`hole_depth`](Self::hole_depth) and
/// [`rows_with_holes`](Self::rows_with_holes) are the BCTS-8 extension and default
/// to their DT-20 values.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BoardWeights {
    /// Height at which the last piece came to rest (penalize building tall).
    pub landing_height: f32,
    /// Cells of the just-placed piece that were cleared (reward useful placements).
    pub eroded_piece_cells: f32,
    /// Filled⇄empty alternations scanning each row (penalize jagged rows).
    pub row_transitions: f32,
    /// Filled⇄empty alternations scanning each column (penalize jagged columns).
    pub column_transitions: f32,
    /// Empty cells with a filled cell somewhere above them (penalize holes).
    pub holes: f32,
    /// Cumulative well depth (triangular sum over each well's depth).
    pub board_wells: f32,
    /// BCTS-8: filled cells stacked directly above each hole (penalize deep burial).
    pub hole_depth: f32,
    /// BCTS-8: number of distinct rows that contain at least one hole.
    pub rows_with_holes: f32,
}

impl BoardWeights {
    /// Dellacherie–Thiery DT-20 board weights for the 10×20 board (finding [2]).
    ///
    /// Used as the [`Weights::DT20`] board group. See the module docs for the
    /// sign-convention caveat. (DT-20's 9th "diversity" feature is omitted here:
    /// the Dellacherie-6 + BCTS-2 set this crate ships does not yet extract it.)
    pub const DT20: Self = Self {
        landing_height: -2.68,
        eroded_piece_cells: 1.38,
        row_transitions: -2.41,
        column_transitions: -6.32,
        holes: 2.03,
        board_wells: -2.71,
        hole_depth: -0.43,
        rows_with_holes: -9.48,
    };

    /// The static-board contribution to a board's Value: the dot product of these
    /// weights with `features`, rounded to the nearest integer.
    ///
    /// Rounding keeps [`Value`](super::Value) an `i32` (so rewards — also integers
    /// — add cleanly along a search path) while the weights stay real-valued for
    /// tuning.
    pub fn dot(&self, features: &super::features::BoardFeatures) -> i32 {
        let sum = self.landing_height * features.landing_height as f32
            + self.eroded_piece_cells * features.eroded_piece_cells as f32
            + self.row_transitions * features.row_transitions as f32
            + self.column_transitions * features.column_transitions as f32
            + self.holes * features.holes as f32
            + self.board_wells * features.board_wells as f32
            + self.hole_depth * features.hole_depth as f32
            + self.rows_with_holes * features.rows_with_holes as f32;
        sum.round() as i32
    }
}

/// Per-move payoff weights: how much each line-clear / spin outcome is worth.
///
/// Seeded from Cold Clear's master reward config (findings [4],[5]). The negative
/// single/double/triple weights are intentional: they push the bot to downstack
/// and hold out for Tetrises / T-spins, preserving the Back-to-Back chain.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RewardWeights {
    /// Clearing exactly one line (penalized — wastes the row).
    pub clear1: f32,
    /// Clearing exactly two lines (penalized).
    pub clear2: f32,
    /// Clearing exactly three lines (mildly penalized).
    pub clear3: f32,
    /// Clearing four lines — a Tetris (strongly rewarded).
    pub clear4: f32,
    /// Mini T-spin clearing one line (penalized — a wasted T).
    pub mini_tspin: f32,
    /// Full T-spin single.
    pub tspin1: f32,
    /// Full T-spin double.
    pub tspin2: f32,
    /// Full T-spin triple.
    pub tspin3: f32,
    /// Bonus added when a clear continues a Back-to-Back chain.
    pub b2b_clear: f32,
    /// Bonus for a perfect clear (board fully emptied).
    pub perfect_clear: f32,
}

impl RewardWeights {
    /// Cold Clear master reward weights (findings [4],[5]).
    pub const COLD_CLEAR: Self = Self {
        clear1: -143.0,
        clear2: -100.0,
        clear3: -58.0,
        clear4: 390.0,
        mini_tspin: -158.0,
        tspin1: 121.0,
        tspin2: 410.0,
        tspin3: 602.0,
        b2b_clear: 104.0,
        perfect_clear: 999.0,
    };
}

/// The full tunable weight set: a board-quality group and a per-move reward group.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Weights {
    /// Coefficients for the static board features → [`Value`](super::Value).
    pub board: BoardWeights,
    /// Coefficients for the per-move outcome → [`Reward`](super::Reward).
    pub reward: RewardWeights,
}

impl Weights {
    /// The shipped default: DT-20 board weights + Cold Clear reward weights.
    ///
    /// A reasonable, citable starting point — *not* a tuned-for-this-engine
    /// optimum. See the module docs for why every number here is provisional.
    pub const DT20: Self = Self {
        board: BoardWeights::DT20,
        reward: RewardWeights::COLD_CLEAR,
    };
}

impl Default for Weights {
    fn default() -> Self {
        Self::DT20
    }
}
