//! The board evaluator: the Cold Clear `(Value, Reward)` seam (AI3.2).
//!
//! A placement search needs to score candidate placements. Following Cold Clear
//! (research finding [3]), scoring splits into two distinct quantities:
//!
//! - [`Value`] — the *static* quality of the resulting board: holes, transitions,
//!   wells, height. A property of a position, independent of how it was reached.
//! - [`Reward`] — the *per-move* payoff of the placement that produced it: line
//!   clears, T-spins, Back-to-Back, perfect clears. A property of a transition.
//!
//! The split is load-bearing, not cosmetic. Because [`Reward`] adds into [`Value`]
//! (`impl Add<Reward> for Value`), rewards accumulate along a search path: a
//! multi-ply search sums the rewards of every move on a branch and adds them to
//! the leaf board's static [`Value`], all through one [`Evaluator`]. So the *same*
//! evaluator serves the greedy one-piece Tier-1 search (AI3.3) and a future
//! multi-ply Tier-2 search with no rework.
//!
//! # The trait
//!
//! [`Evaluator`] is object-safe (used as `&dyn Evaluator` by the planner seam) and
//! `Send + Sync` (the search may run off-thread on native). The shipped
//! implementation, [`LinearEvaluator`], is a linear weighted sum of the
//! Dellacherie / BCTS features (see [`features`]) with the tunable weights of
//! [`weights`].
//!
//! # Determinism
//!
//! Pure Rust, no Bevy, no RNG, no clock — like [`crate::engine`]. Evaluation is a
//! deterministic function of its inputs, so the same board + lock always scores
//! identically. Any randomness the AI needs (tie-breaking, error injection) lives
//! behind the controller's own seeded RNG, never here.

pub mod features;
pub mod weights;

use std::ops::Add;

use crate::engine::{Board, LockOutcome, TSpinKind};

pub use features::BoardFeatures;
pub use weights::{BoardWeights, RewardWeights, Weights};

/// The static quality of a board position. Higher is better.
///
/// Produced from the board's [`BoardFeatures`] by an [`Evaluator`]. Carries an
/// `i32` so it composes cleanly with the integer [`Reward`] along a search path.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct Value(pub i32);

/// The per-move payoff of a placement (line clears, spins, B2B). Higher is better.
///
/// Summed along a search path and folded into a leaf's [`Value`] via
/// [`Add<Reward> for Value`](Value::add). Implements [`Add`] with itself so a
/// branch's rewards accumulate before meeting the board Value.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct Reward(pub i32);

impl Add<Reward> for Value {
    type Output = Value;

    /// Fold a move's [`Reward`] into a board [`Value`]: this is what lets a search
    /// accumulate path rewards into the leaf's static score.
    fn add(self, reward: Reward) -> Value {
        Value(self.0 + reward.0)
    }
}

impl Add for Reward {
    type Output = Reward;

    /// Accumulate rewards along a multi-move branch before they meet the Value.
    fn add(self, other: Reward) -> Reward {
        Reward(self.0 + other.0)
    }
}

/// Scores a board placement as a `(Value, Reward)` pair.
///
/// Object-safe (`&dyn Evaluator`) and thread-safe so the planner can run a search
/// off-thread on native targets. See the [module docs](self) for the meaning of
/// the split.
pub trait Evaluator: Send + Sync {
    /// Score the board that results from a placement.
    ///
    /// - `lock`: the [`LockOutcome`] of the placement (what cleared, where the
    ///   piece rested) — drives the per-move [`Reward`] and the two per-move board
    ///   features (landing height, eroded cells).
    /// - `board`: the board *after* the placement's line clears — drives the
    ///   static board [`Value`].
    /// - `t_spin`: the T-spin classification of the placement, if any, from
    ///   [`classify_t_spin`](crate::engine::classify_t_spin) — refines the
    ///   [`Reward`] for spins.
    fn evaluate(
        &self,
        lock: &LockOutcome,
        board: &Board,
        t_spin: Option<TSpinKind>,
    ) -> (Value, Reward);
}

/// The shipped Tier-1 evaluator: a linear weighted sum of the Dellacherie / BCTS
/// features, with tunable [`Weights`].
///
/// Construct with [`LinearEvaluator::default`] for the DT-20 + Cold Clear default
/// weights, or [`LinearEvaluator::new`] to supply a tuned set.
#[derive(Clone, Copy, Debug, Default)]
pub struct LinearEvaluator {
    weights: Weights,
}

impl LinearEvaluator {
    /// A linear evaluator with the given weights.
    pub fn new(weights: Weights) -> Self {
        Self { weights }
    }

    /// The weights this evaluator scores with.
    pub fn weights(&self) -> &Weights {
        &self.weights
    }

    /// The per-move [`Reward`] for a placement: the weighted line-clear / spin /
    /// Back-to-Back payoff.
    ///
    /// Classifies the placement into exactly one clear category from `lock` +
    /// `t_spin`, weights it, then adds the B2B bonus and the perfect-clear bonus.
    ///
    /// **Note on Back-to-Back:** the engine's running B2B *chain* state is not
    /// passed to [`Evaluator::evaluate`] (the trait signature is board + lock +
    /// t-spin). So the B2B bonus here is applied to every B2B-*eligible* clear (a
    /// Tetris or any full T-spin line clear), rewarding placements that *sustain* a
    /// chain. Modeling the precise "only when the previous clear was also B2B"
    /// rule is deferred to the search layer, which carries `b2b` in its
    /// [`SearchState`](crate::ai::SearchState).
    fn reward(&self, lock: &LockOutcome, board: &Board, t_spin: Option<TSpinKind>) -> Reward {
        let w = &self.weights.reward;
        let lines = lock.cleared_rows.len();

        // The placement falls into exactly one scoring category.
        let (base, b2b_eligible) = match (t_spin, lines) {
            (Some(TSpinKind::Mini), 1 | 2) => (w.mini_tspin, true),
            (Some(TSpinKind::Full), 1) => (w.tspin1, true),
            (Some(TSpinKind::Full), 2) => (w.tspin2, true),
            (Some(TSpinKind::Full), 3) => (w.tspin3, true),
            // A T-spin that cleared no lines scores nothing on its own (the board
            // Value still reflects the resulting shape).
            (Some(_), _) => (0.0, false),
            (None, 1) => (w.clear1, false),
            (None, 2) => (w.clear2, false),
            (None, 3) => (w.clear3, false),
            (None, 4) => (w.clear4, true),
            (None, _) => (0.0, false), // no lines cleared
        };

        let mut total = base;
        if b2b_eligible {
            total += w.b2b_clear;
        }
        if lines > 0 && is_perfect_clear(board) {
            total += w.perfect_clear;
        }
        Reward(total.round() as i32)
    }
}

impl Evaluator for LinearEvaluator {
    fn evaluate(
        &self,
        lock: &LockOutcome,
        board: &Board,
        t_spin: Option<TSpinKind>,
    ) -> (Value, Reward) {
        let features = BoardFeatures::extract(board, lock);
        let value = Value(self.weights.board.dot(&features));
        (value, self.reward(lock, board, t_spin))
    }
}

/// Whether the board is completely empty (a perfect clear). Cheap: the engine's
/// `cells()` lists only occupied cells.
fn is_perfect_clear(board: &Board) -> bool {
    board.cells().is_empty()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{CellKind, PieceType};

    fn no_clear_lock() -> LockOutcome {
        LockOutcome {
            cells_locked: vec![(0, 0, CellKind::Some(PieceType::O))],
            cleared_rows: Vec::new(),
            top_y_after_lock: Some(0),
        }
    }

    #[test]
    fn value_plus_reward_sums_the_scalars() {
        assert_eq!(Value(10) + Reward(5), Value(15));
        assert_eq!(Value(10) + Reward(-3), Value(7));
    }

    #[test]
    fn rewards_accumulate_along_a_path_then_meet_value() {
        // A two-move branch: rewards add, then fold into the leaf board Value.
        let path = Reward(100) + Reward(-30) + Reward(50);
        assert_eq!(path, Reward(120));
        assert_eq!(Value(7) + path, Value(127));
    }

    #[test]
    fn default_evaluator_uses_dt20_and_cold_clear_weights() {
        let eval = LinearEvaluator::default();
        assert_eq!(eval.weights().board, BoardWeights::DT20);
        assert_eq!(eval.weights().reward, RewardWeights::COLD_CLEAR);
    }

    #[test]
    fn value_is_the_weighted_feature_dot_on_a_known_board() {
        // The same composite board pinned in features.rs:
        //   y2: X . .   y1: X . .   y0: X . X
        // features there: holes 0, hole_depth 0, rows_with_holes 0,
        //   row_transitions 6, column_transitions 3, board_wells 1,
        //   landing_height 0 (no move), eroded_piece_cells 0.
        let mut board = Board::new(3, 4);
        board.set(0, 0, CellKind::Some(PieceType::O));
        board.set(0, 1, CellKind::Some(PieceType::O));
        board.set(0, 2, CellKind::Some(PieceType::O));
        board.set(2, 0, CellKind::Some(PieceType::O));

        // Score with a *known* simple weight set so the arithmetic is exact:
        // value = row_transitions*(-1) + column_transitions*(-2) + board_wells*(-3)
        //       = 6*-1 + 3*-2 + 1*-3 = -6 -6 -3 = -15.
        let weights = Weights {
            board: BoardWeights {
                landing_height: 0.0,
                eroded_piece_cells: 0.0,
                row_transitions: -1.0,
                column_transitions: -2.0,
                holes: 0.0,
                board_wells: -3.0,
                hole_depth: 0.0,
                rows_with_holes: 0.0,
            },
            reward: RewardWeights::COLD_CLEAR,
        };
        let eval = LinearEvaluator::new(weights);
        let (value, reward) = eval.evaluate(&no_clear_lock(), &board, None);
        assert_eq!(value, Value(-15));
        assert_eq!(reward, Reward(0), "no lines cleared => no reward");
    }

    #[test]
    fn reward_tetris_adds_b2b_bonus() {
        let eval = LinearEvaluator::default();
        let lock = LockOutcome {
            cells_locked: vec![(0, 0, CellKind::Some(PieceType::I))],
            cleared_rows: vec![0, 1, 2, 3],
            top_y_after_lock: None,
        };
        // Not a perfect clear: leave a stray cell on the resulting board.
        let mut board = Board::new(4, 6);
        board.set(0, 0, CellKind::Some(PieceType::O));
        let (_v, reward) = eval.evaluate(&lock, &board, None);
        // clear4 (390) + b2b_clear (104) = 494.
        assert_eq!(reward, Reward(494));
    }

    #[test]
    fn reward_single_is_penalized_and_no_b2b() {
        let eval = LinearEvaluator::default();
        let lock = LockOutcome {
            cells_locked: vec![(0, 0, CellKind::Some(PieceType::O))],
            cleared_rows: vec![0],
            top_y_after_lock: None,
        };
        let mut board = Board::new(4, 6);
        board.set(0, 0, CellKind::Some(PieceType::O)); // not a perfect clear
        let (_v, reward) = eval.evaluate(&lock, &board, None);
        assert_eq!(reward, Reward(-143)); // clear1, no b2b, no PC
    }

    #[test]
    fn reward_t_spin_double_uses_tspin2_and_b2b() {
        let eval = LinearEvaluator::default();
        let lock = LockOutcome {
            cells_locked: vec![(0, 0, CellKind::Some(PieceType::T))],
            cleared_rows: vec![0, 1],
            top_y_after_lock: None,
        };
        let mut board = Board::new(4, 6);
        board.set(3, 0, CellKind::Some(PieceType::O)); // not a perfect clear
        let (_v, reward) = eval.evaluate(&lock, &board, Some(TSpinKind::Full));
        // tspin2 (410) + b2b (104) = 514.
        assert_eq!(reward, Reward(514));
    }

    #[test]
    fn reward_perfect_clear_stacks_on_top() {
        let eval = LinearEvaluator::default();
        let lock = LockOutcome {
            cells_locked: vec![(0, 0, CellKind::Some(PieceType::I))],
            cleared_rows: vec![0, 1, 2, 3],
            top_y_after_lock: None,
        };
        let board = Board::new(4, 6); // empty => perfect clear
        let (_v, reward) = eval.evaluate(&lock, &board, None);
        // clear4 (390) + b2b (104) + perfect_clear (999) = 1493.
        assert_eq!(reward, Reward(1493));
    }

    #[test]
    fn reward_mini_t_spin_is_penalized() {
        let eval = LinearEvaluator::default();
        let lock = LockOutcome {
            cells_locked: vec![(0, 0, CellKind::Some(PieceType::T))],
            cleared_rows: vec![0],
            top_y_after_lock: None,
        };
        let mut board = Board::new(4, 6);
        board.set(3, 0, CellKind::Some(PieceType::O));
        let (_v, reward) = eval.evaluate(&lock, &board, Some(TSpinKind::Mini));
        // mini_tspin (-158) + b2b (104) = -54.
        assert_eq!(reward, Reward(-54));
    }

    #[test]
    fn evaluator_is_object_safe() {
        // Compiles only if the trait is object-safe; exercises the &dyn path the
        // planner will use.
        let eval = LinearEvaluator::default();
        let dyn_eval: &dyn Evaluator = &eval;
        let board = Board::new(4, 6);
        let (_v, _r) = dyn_eval.evaluate(&no_clear_lock(), &board, None);
    }
}
