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
//! The shipped default reward weights are a **survival** profile
//! ([`RewardWeights::SURVIVAL`]) that pays the Tier-1 greedy planner to clear lines
//! *now* — a 1-ply search has no lookahead to defer them. Cold Clear's master
//! reward config ([`RewardWeights::COLD_CLEAR`], findings [4],[5]) — which
//! *penalizes* small clears to force downstacking and preserve the B2B chain — is
//! kept as the [`Weights::DOWNSTACK`] profile for a future multi-ply Tier-2 beam,
//! where deferring clears actually pays off. Pairing Cold Clear's downstacking
//! rewards with a 1-ply greedy buries the bot (it never cashes the downstack in),
//! so it is deliberately NOT the default.

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
    /// Tetris-well readiness (offense): rows ready to clear via an I-piece in the
    /// lowest column. Positive ⇒ reward building toward a Tetris. See
    /// [`BoardFeatures::tetris_well`](super::features::BoardFeatures::tetris_well).
    pub tetris_well: f32,
    /// Combo-readiness (offense): rows one cell from clearing. Positive ⇒ reward
    /// building a combo machine. See
    /// [`BoardFeatures::near_full_rows`](super::features::BoardFeatures::near_full_rows).
    pub near_full_rows: f32,
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
        // Offense feature, neutral by default: DT-20 is a survival vector, so this
        // adds nothing until autoresearch tunes it positive to build Tetrises.
        tetris_well: 0.0,
        // Combo-readiness, neutral by default (tuned positive for combo offense).
        near_full_rows: 0.0,
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
            + self.rows_with_holes * features.rows_with_holes as f32
            + self.tetris_well * features.tetris_well as f32
            + self.near_full_rows * features.near_full_rows as f32;
        sum.round() as i32
    }

    /// Number of tunable coefficients (one per board feature).
    pub const PARAM_COUNT: usize = 10;

    /// The weights as a flat vector in [`BoardFeatures`](super::features::BoardFeatures)
    /// order, for hillclimbing: `[landing_height, eroded_piece_cells, row_transitions,
    /// column_transitions, holes, board_wells, hole_depth, rows_with_holes,
    /// tetris_well, near_full_rows]`.
    pub fn params(&self) -> [f32; Self::PARAM_COUNT] {
        [
            self.landing_height,
            self.eroded_piece_cells,
            self.row_transitions,
            self.column_transitions,
            self.holes,
            self.board_wells,
            self.hole_depth,
            self.rows_with_holes,
            self.tetris_well,
            self.near_full_rows,
        ]
    }

    /// Build a weight set from a flat vector (see [`params`](Self::params) for order).
    pub fn from_params(p: &[f32; Self::PARAM_COUNT]) -> Self {
        Self {
            landing_height: p[0],
            eroded_piece_cells: p[1],
            row_transitions: p[2],
            column_transitions: p[3],
            holes: p[4],
            board_wells: p[5],
            hole_depth: p[6],
            rows_with_holes: p[7],
            tetris_well: p[8],
            near_full_rows: p[9],
        }
    }
}

/// Per-move payoff weights: how much each line-clear / spin outcome is worth.
///
/// Two profiles ship: [`SURVIVAL`](Self::SURVIVAL) (the Tier-1 greedy default —
/// every clear positive) and [`COLD_CLEAR`](Self::COLD_CLEAR) (a downstacking
/// profile for a future multi-ply beam, where small clears are penalized to hold
/// out for Tetrises / T-spins). The *sign* of each weight therefore depends on the
/// profile; the field docs below describe what each measures, not its sign.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RewardWeights {
    /// Payoff for clearing exactly one line.
    pub clear1: f32,
    /// Payoff for clearing exactly two lines.
    pub clear2: f32,
    /// Payoff for clearing exactly three lines.
    pub clear3: f32,
    /// Payoff for clearing four lines — a Tetris.
    pub clear4: f32,
    /// Payoff for a mini T-spin clearing a line.
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
    /// Scales the **actual guideline attack** this clear sends — combo, Back-to-Back,
    /// spin, and perfect-clear bonuses all included, via
    /// [`attack_lines`](crate::engine::attack_lines) evaluated over the search-path
    /// [`EvalContext`](super::EvalContext). The direct APP lever: where the abstract
    /// `clearN` / `tspinN` weights are a hand-tuned *proxy* for attack value, this
    /// rewards the garbage a placement *actually* produces, so a multi-ply search
    /// values escalating combos and sustained B2B chains (the multipliers that
    /// weight-tuning alone, blind to chain state, cannot reach). Default `0.0` ⇒ the
    /// shipped survival profile is unchanged; the attack sprint tunes it up.
    pub attack: f32,
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
        attack: 0.0,
    };

    /// Survival reward weights for the **Tier-1 greedy** planner — the shipped
    /// default.
    ///
    /// Unlike [`COLD_CLEAR`](Self::COLD_CLEAR) — whose negative single/double/triple
    /// weights assume a deep beam that *defers* clears to build Tetrises — a 1-ply
    /// greedy has no lookahead to cash a downstack in later, so it must be paid to
    /// clear *now* or it buries itself and tops out. These weights reward every
    /// clear, rising with lines (Tetris best) and with T-spins above same-line
    /// normal clears, while the board weights stay in charge of keeping the stack
    /// clean. Empirically the greedy bot then survives indefinitely instead of
    /// topping out in ~40-126 pieces.
    pub const SURVIVAL: Self = Self {
        clear1: 80.0,
        clear2: 200.0,
        clear3: 360.0,
        clear4: 640.0,
        mini_tspin: 60.0,
        tspin1: 240.0,
        tspin2: 480.0,
        tspin3: 720.0,
        b2b_clear: 80.0,
        perfect_clear: 1600.0,
        attack: 0.0,
    };

    /// Number of tunable reward coefficients.
    pub const PARAM_COUNT: usize = 11;

    /// The reward weights as a flat vector, for hillclimbing: `[clear1, clear2,
    /// clear3, clear4, mini_tspin, tspin1, tspin2, tspin3, b2b_clear, perfect_clear,
    /// attack]`.
    pub fn params(&self) -> [f32; Self::PARAM_COUNT] {
        [
            self.clear1,
            self.clear2,
            self.clear3,
            self.clear4,
            self.mini_tspin,
            self.tspin1,
            self.tspin2,
            self.tspin3,
            self.b2b_clear,
            self.perfect_clear,
            self.attack,
        ]
    }

    /// Build a reward set from a flat vector (see [`params`](Self::params) for order).
    pub fn from_params(p: &[f32; Self::PARAM_COUNT]) -> Self {
        Self {
            clear1: p[0],
            clear2: p[1],
            clear3: p[2],
            clear4: p[3],
            mini_tspin: p[4],
            tspin1: p[5],
            tspin2: p[6],
            tspin3: p[7],
            b2b_clear: p[8],
            perfect_clear: p[9],
            attack: p[10],
        }
    }
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
    /// The shipped **Tier-1 default**: DT-20 board weights + the survival reward
    /// profile. The board group keeps the stack clean and low; the reward group
    /// pays the greedy planner to clear lines (see [`RewardWeights::SURVIVAL`]).
    /// Citable starting points, not a tuned-for-this-engine optimum; the
    /// research harness is the tuning venue.
    pub const SURVIVAL: Self = Self {
        board: BoardWeights::DT20,
        reward: RewardWeights::SURVIVAL,
    };
}

impl Default for Weights {
    /// The Tier-1 survival default ([`Weights::SURVIVAL`]).
    fn default() -> Self {
        Self::SURVIVAL
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // The hill-climbers tune these weights as flat `f32` vectors, so `params()` and
    // `from_params()` must be exact inverses and `PARAM_COUNT` must equal the vector
    // width. These are hand-indexed (a field added to the struct but not to both
    // methods, or listed in a different order, is the classic desync), so the round
    // trip is the contract that catches it. Until a `#[derive(Tunable)]` makes the
    // mapping unforgeable, these tests are the lock.

    #[test]
    fn board_weights_round_trip_through_params() {
        let w = BoardWeights::DT20;
        assert_eq!(w.params().len(), BoardWeights::PARAM_COUNT);
        assert_eq!(BoardWeights::from_params(&w.params()), w);
    }

    #[test]
    fn board_weights_from_params_is_positional() {
        // Distinct value per slot ⇒ any index swap between `params`/`from_params`
        // changes the result and fails the comparison.
        let p: [f32; BoardWeights::PARAM_COUNT] = std::array::from_fn(|i| (i as f32 + 1.0) * 1.5);
        assert_eq!(BoardWeights::from_params(&p).params(), p);
    }

    #[test]
    fn reward_weights_round_trip_through_params() {
        for w in [RewardWeights::SURVIVAL, RewardWeights::COLD_CLEAR] {
            assert_eq!(w.params().len(), RewardWeights::PARAM_COUNT);
            assert_eq!(RewardWeights::from_params(&w.params()), w);
        }
    }

    #[test]
    fn reward_weights_from_params_is_positional() {
        let p: [f32; RewardWeights::PARAM_COUNT] = std::array::from_fn(|i| (i as f32 + 1.0) * 7.0);
        assert_eq!(RewardWeights::from_params(&p).params(), p);
    }
}
