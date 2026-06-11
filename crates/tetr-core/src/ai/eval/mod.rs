//! The board evaluator: the Cold Clear `(Value, Reward)` seam.
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
//! evaluator serves the greedy one-piece Tier-1 search and a future
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

pub mod cc2;
pub mod features;
pub mod weights;

use std::ops::Add;

use crate::engine::{
    attack_lines, qualifies_for_back_to_back, Board, EngineScoreAction, LockOutcome, TSpinKind,
};

pub use cc2::{Cc2Evaluator, Cc2Weights};
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

/// Search-path context for one evaluation: the running chain state *before* the
/// placement being scored.
///
/// The `(lock, board, t_spin)` triple describes a single transition in isolation, so
/// it cannot express **combo** or **Back-to-Back** — both of which depend on the path
/// taken to reach the placement, and both of which are major attack multipliers. The
/// search already tracks them in [`SearchState`](crate::ai::SearchState); this struct
/// carries them into the evaluator so an attack-aware eval (and the value net) can
/// value combo / B2B continuation. [`Default`] is the neutral context (no combo, no
/// B2B chain) used for one-off scoring and tests — under it an evaluator must reduce
/// to its chain-agnostic behavior.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct EvalContext {
    /// Combo chain length before this placement (`0` = no active combo).
    pub combo: u32,
    /// Whether a Back-to-Back chain was active before this placement.
    pub b2b: bool,
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
        ctx: EvalContext,
    ) -> (Value, Reward);

    /// Score a placement whose resulting board is given as a [`ColumnView`] — the
    /// bitboard search's hot path, avoiding the `Array2D`. The default reconstructs a
    /// dense [`Board`] and defers to [`evaluate`](Self::evaluate) (correct, allocating);
    /// both shipped evaluators override it to read the columns directly. An override
    /// **must** be bit-identical to the default — pinned per impl by an
    /// `evaluate_cols`-vs-`evaluate` differential test.
    ///
    /// [`ColumnView`]: crate::engine::ColumnView
    fn evaluate_cols(
        &self,
        lock: &LockOutcome,
        board: crate::engine::ColumnView,
        t_spin: Option<TSpinKind>,
        ctx: EvalContext,
    ) -> (Value, Reward) {
        self.evaluate(lock, &board.to_board(), t_spin, ctx)
    }
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

    /// Shared scoring core over the column bitboard: both trait paths feed it their
    /// columns, so the dense and bitboard scores are bit-identical by construction
    /// (the same shape as `Cc2Evaluator::score`).
    fn score(
        &self,
        cols: &[u64],
        lock: &LockOutcome,
        t_spin: Option<TSpinKind>,
        ctx: EvalContext,
    ) -> (Value, Reward) {
        let features = BoardFeatures::extract_cols(cols, lock);
        let value = Value(self.weights.board.dot(&features));
        let board_is_empty = cols.iter().all(|&c| c == 0);
        (
            value,
            reward_for(&self.weights.reward, lock, board_is_empty, t_spin, ctx),
        )
    }
}

impl Evaluator for LinearEvaluator {
    /// `ctx` (combo / B2B chain) feeds the reward's attack term ([`RewardWeights::attack`]):
    /// the board [`Value`] is chain-independent (a resting position is what it is), but
    /// the per-move [`Reward`] now values the actual garbage a clear sends under the
    /// search-path chain. At the default `attack == 0.0` the reward is unchanged.
    fn evaluate(
        &self,
        lock: &LockOutcome,
        board: &Board,
        t_spin: Option<TSpinKind>,
        ctx: EvalContext,
    ) -> (Value, Reward) {
        self.score(&board.column_bits(), lock, t_spin, ctx)
    }

    /// Fast bitboard path: the columns ARE the input — no dense reconstruction.
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

/// The per-move [`Reward`] for a placement under the given [`RewardWeights`].
///
/// Shared by [`LinearEvaluator`] and any external evaluator (e.g. a learned value
/// net) that wants the same principled clear / spin / Back-to-Back payoff while
/// supplying its own board [`Value`]. This is the seam that lets a learned
/// evaluator replace *only* the static board score and keep the engine-faithful
/// reward math.
///
/// Classifies the placement into exactly one clear category from `lock` + `t_spin`,
/// weights it, then adds the B2B bonus and the perfect-clear bonus.
///
/// **Two B2B paths:** the *abstract* `b2b_clear` bonus is applied to every
/// B2B-*eligible* clear (a Tetris or full/mini T-spin line clear), rewarding
/// placements that *can* sustain a chain regardless of history. The *attack* term
/// (`w.attack`), by contrast, uses `ctx.b2b` for the precise "only when the
/// previous clear was also B2B" continuation rule — that's why the chain context
/// now flows in. With `w.attack == 0.0` (the shipped default) only the abstract
/// path is active, matching the prior chain-agnostic behavior exactly.
pub fn compute_reward(
    weights: &RewardWeights,
    lock: &LockOutcome,
    board: &Board,
    t_spin: Option<TSpinKind>,
    ctx: EvalContext,
) -> Reward {
    reward_for(weights, lock, board.is_empty(), t_spin, ctx)
}

/// [`compute_reward`]'s core, with the perfect-clear input reduced to the one bit it
/// actually reads — `board_is_empty` — so the bitboard scoring paths can feed it from
/// the columns without materialising a dense [`Board`].
fn reward_for(
    weights: &RewardWeights,
    lock: &LockOutcome,
    board_is_empty: bool,
    t_spin: Option<TSpinKind>,
    ctx: EvalContext,
) -> Reward {
    let w = weights;
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

    let perfect = lines > 0 && board_is_empty;

    let mut total = base;
    if b2b_eligible {
        total += w.b2b_clear;
    }
    if perfect {
        total += w.perfect_clear;
    }

    // Attack-aware term (the APP lever): the garbage this clear actually sends under
    // the search-path chain — combo count and B2B *continuation* both from `ctx` —
    // via the engine's guideline table, scaled by `w.attack`. At the shipped default
    // `w.attack == 0.0` this adds nothing, so the reward stays chain-agnostic and the
    // survival profile is byte-for-byte unchanged.
    if lines > 0 {
        let action = EngineScoreAction::from_lock_result(t_spin, lines);
        // Continuation uses the ENGINE's qualifying rule (`qualifies_for_back_to_back`,
        // the same predicate `ScoreState::lock_result` applies) rather than the local
        // `b2b_eligible` bonus table above. Since the Mini-Double row was unified
        // across the rule tables the two happen to coincide, but the engine predicate
        // stays the source of truth here — this term claims engine-exact attack.
        let b2b_continue = ctx.b2b && qualifies_for_back_to_back(t_spin, lines);
        let attack = attack_lines(action, b2b_continue, ctx.combo, perfect);
        total += w.attack * attack as f32;
    }

    Reward(total.round() as i32)
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

    /// An evaluator on the Cold Clear *downstacking* reward profile, for pinning
    /// that profile's reward math now that the shipped default is the survival
    /// profile.
    fn downstack_eval() -> LinearEvaluator {
        LinearEvaluator::new(Weights {
            board: BoardWeights::DT20,
            reward: RewardWeights::COLD_CLEAR,
        })
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
    fn default_evaluator_uses_dt20_board_and_survival_reward() {
        let eval = LinearEvaluator::default();
        assert_eq!(eval.weights().board, BoardWeights::DT20);
        assert_eq!(eval.weights().reward, RewardWeights::SURVIVAL);
    }

    #[test]
    fn default_survival_profile_rewards_line_clears_positively() {
        // The fix's core property: the shipped (Tier-1 greedy) default must PAY for
        // clearing lines so the 1-ply bot cashes them in instead of stacking into a
        // top-out. (The Cold Clear downstack profile — `downstack_eval` — does the
        // opposite on purpose.)
        let eval = LinearEvaluator::default();
        let mut board = Board::new(4, 6);
        board.set(0, 0, CellKind::Some(PieceType::O)); // not a perfect clear
        for lines in 1..=4usize {
            let lock = LockOutcome {
                cells_locked: vec![(0, 0, CellKind::Some(PieceType::I))],
                cleared_rows: (0..lines as isize).collect(),
                top_y_after_lock: None,
            };
            let (_v, reward) = eval.evaluate(&lock, &board, None, EvalContext::default());
            assert!(
                reward.0 > 0,
                "survival default must reward a {lines}-line clear, got {reward:?}"
            );
        }
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
                tetris_well: 0.0,
                near_full_rows: 0.0,
            },
            reward: RewardWeights::COLD_CLEAR,
        };
        let eval = LinearEvaluator::new(weights);
        let (value, reward) = eval.evaluate(&no_clear_lock(), &board, None, EvalContext::default());
        assert_eq!(value, Value(-15));
        assert_eq!(reward, Reward(0), "no lines cleared => no reward");
    }

    #[test]
    fn reward_tetris_adds_b2b_bonus() {
        let eval = downstack_eval();
        let lock = LockOutcome {
            cells_locked: vec![(0, 0, CellKind::Some(PieceType::I))],
            cleared_rows: vec![0, 1, 2, 3],
            top_y_after_lock: None,
        };
        // Not a perfect clear: leave a stray cell on the resulting board.
        let mut board = Board::new(4, 6);
        board.set(0, 0, CellKind::Some(PieceType::O));
        let (_v, reward) = eval.evaluate(&lock, &board, None, EvalContext::default());
        // clear4 (390) + b2b_clear (104) = 494.
        assert_eq!(reward, Reward(494));
    }

    #[test]
    fn reward_single_is_penalized_and_no_b2b() {
        let eval = downstack_eval();
        let lock = LockOutcome {
            cells_locked: vec![(0, 0, CellKind::Some(PieceType::O))],
            cleared_rows: vec![0],
            top_y_after_lock: None,
        };
        let mut board = Board::new(4, 6);
        board.set(0, 0, CellKind::Some(PieceType::O)); // not a perfect clear
        let (_v, reward) = eval.evaluate(&lock, &board, None, EvalContext::default());
        assert_eq!(reward, Reward(-143)); // clear1, no b2b, no PC
    }

    #[test]
    fn reward_t_spin_double_uses_tspin2_and_b2b() {
        let eval = downstack_eval();
        let lock = LockOutcome {
            cells_locked: vec![(0, 0, CellKind::Some(PieceType::T))],
            cleared_rows: vec![0, 1],
            top_y_after_lock: None,
        };
        let mut board = Board::new(4, 6);
        board.set(3, 0, CellKind::Some(PieceType::O)); // not a perfect clear
        let (_v, reward) =
            eval.evaluate(&lock, &board, Some(TSpinKind::Full), EvalContext::default());
        // tspin2 (410) + b2b (104) = 514.
        assert_eq!(reward, Reward(514));
    }

    #[test]
    fn reward_perfect_clear_stacks_on_top() {
        let eval = downstack_eval();
        let lock = LockOutcome {
            cells_locked: vec![(0, 0, CellKind::Some(PieceType::I))],
            cleared_rows: vec![0, 1, 2, 3],
            top_y_after_lock: None,
        };
        let board = Board::new(4, 6); // empty => perfect clear
        let (_v, reward) = eval.evaluate(&lock, &board, None, EvalContext::default());
        // clear4 (390) + b2b (104) + perfect_clear (999) = 1493.
        assert_eq!(reward, Reward(1493));
    }

    #[test]
    fn reward_mini_t_spin_is_penalized() {
        let eval = downstack_eval();
        let lock = LockOutcome {
            cells_locked: vec![(0, 0, CellKind::Some(PieceType::T))],
            cleared_rows: vec![0],
            top_y_after_lock: None,
        };
        let mut board = Board::new(4, 6);
        board.set(3, 0, CellKind::Some(PieceType::O));
        let (_v, reward) =
            eval.evaluate(&lock, &board, Some(TSpinKind::Mini), EvalContext::default());
        // mini_tspin (-158) + b2b (104) = -54.
        assert_eq!(reward, Reward(-54));
    }

    #[test]
    fn attack_term_zero_is_chain_agnostic() {
        // At the shipped default `attack == 0.0`, the reward must be independent of the
        // chain context — the property that keeps every shipped bot byte-for-byte
        // unchanged by the EvalContext wiring.
        let lock = LockOutcome {
            cells_locked: vec![(0, 0, CellKind::Some(PieceType::I))],
            cleared_rows: vec![0, 1, 2, 3],
            top_y_after_lock: None,
        };
        let mut board = Board::new(4, 6);
        board.set(0, 0, CellKind::Some(PieceType::O)); // not a perfect clear
        let neutral = compute_reward(
            &RewardWeights::SURVIVAL,
            &lock,
            &board,
            None,
            EvalContext::default(),
        );
        let with_chain = compute_reward(
            &RewardWeights::SURVIVAL,
            &lock,
            &board,
            None,
            EvalContext {
                combo: 5,
                b2b: true,
            },
        );
        assert_eq!(
            neutral, with_chain,
            "attack=0 ⇒ reward independent of combo/B2B"
        );
    }

    #[test]
    fn attack_term_scales_guideline_attack_under_chain() {
        // A Tetris continuing a B2B chain at combo index 3 sends, by the guideline
        // table, 4 (Tetris) + 1 (B2B) + 1 (COMBO_TABLE[3]) = 6 lines. The attack term
        // adds `w.attack * 6` over the otherwise-identical abstract reward.
        let lock = LockOutcome {
            cells_locked: vec![(0, 0, CellKind::Some(PieceType::I))],
            cleared_rows: vec![0, 1, 2, 3],
            top_y_after_lock: None,
        };
        let mut board = Board::new(4, 6);
        board.set(0, 0, CellKind::Some(PieceType::O)); // not a perfect clear
        let ctx = EvalContext {
            combo: 3,
            b2b: true,
        };
        let mut with_attack = RewardWeights::SURVIVAL;
        with_attack.attack = 10.0;
        let base = compute_reward(&RewardWeights::SURVIVAL, &lock, &board, None, ctx);
        let scaled = compute_reward(&with_attack, &lock, &board, None, ctx);
        assert_eq!(
            scaled.0 - base.0,
            60,
            "delta = w.attack(10) * attack_lines(6)"
        );
    }

    #[test]
    fn linear_evaluate_cols_matches_evaluate() {
        // The bit-identical contract on the override (the per-impl differential the
        // trait doc promises): scoring through the ColumnView fast path must equal
        // scoring the equivalent dense board, including reward and perfect-clear.
        let eval = LinearEvaluator::default();

        let mut stacked = Board::new(4, 8);
        stacked.set(0, 0, CellKind::Some(PieceType::O));
        stacked.set(0, 2, CellKind::Some(PieceType::O)); // a covered hole at (0, 1)
        stacked.set(2, 0, CellKind::Some(PieceType::O));
        let clear_lock = LockOutcome {
            cells_locked: vec![(1, 0, CellKind::Some(PieceType::I))],
            cleared_rows: vec![0],
            top_y_after_lock: Some(1),
        };
        let empty = Board::new(4, 8); // post-clear empty => perfect-clear path

        for (board, lock, t_spin) in [
            (&stacked, &no_clear_lock(), None),
            (&stacked, &clear_lock, None),
            (&empty, &clear_lock, Some(TSpinKind::Full)),
        ] {
            let ctx = EvalContext {
                combo: 2,
                b2b: true,
            };
            let bb = crate::engine::BitBoard::from_board(board);
            assert_eq!(
                eval.evaluate_cols(lock, bb.view(), t_spin, ctx),
                eval.evaluate(lock, board, t_spin, ctx),
            );
        }
    }

    #[test]
    fn mini_double_attack_continues_b2b_like_the_engine() {
        // The attack term's continuation rule is the ENGINE's
        // `qualifies_for_back_to_back`. With the Mini-Double row unified across the
        // rule tables, a mini t-spin double under an active chain prices the +1 B2B
        // attack line — guideline attack: mini double 1 + B2B 1 = 2 vs 1 unchained,
        // so at w.attack = 10 the chained reward is exactly +10 higher.
        let lock = LockOutcome {
            cells_locked: vec![(0, 0, CellKind::Some(PieceType::T))],
            cleared_rows: vec![0, 1],
            top_y_after_lock: None,
        };
        let mut board = Board::new(4, 6);
        board.set(0, 0, CellKind::Some(PieceType::O)); // not a perfect clear
        let mut w = RewardWeights::SURVIVAL;
        w.attack = 10.0;
        let spin = Some(TSpinKind::Mini);
        let chained = compute_reward(
            &w,
            &lock,
            &board,
            spin,
            EvalContext {
                combo: 0,
                b2b: true,
            },
        );
        let fresh = compute_reward(
            &w,
            &lock,
            &board,
            spin,
            EvalContext {
                combo: 0,
                b2b: false,
            },
        );
        assert_eq!(
            chained.0 - fresh.0,
            10,
            "a chained mini double prices exactly one extra attack line"
        );
    }

    #[test]
    fn evaluator_is_object_safe() {
        // Compiles only if the trait is object-safe; exercises the &dyn path the
        // planner will use.
        let eval = LinearEvaluator::default();
        let dyn_eval: &dyn Evaluator = &eval;
        let board = Board::new(4, 6);
        let (_v, _r) = dyn_eval.evaluate(&no_clear_lock(), &board, None, EvalContext::default());
    }

    #[test]
    fn default_batch_matches_scalar() {
        // The default `evaluate_batch` must be bit-identical to mapping
        // `evaluate` over the same inputs (the seam the beam relies on so that a
        // depth-1 batched search == the scalar greedy search).
        let eval = LinearEvaluator::default();

        // Three distinct (LockOutcome, Board, t_spin) cases: a no-clear O lock, a
        // Tetris on a board with a stray cell, and a full T-spin double.
        let mut board_a = Board::new(4, 6);
        board_a.set(0, 0, CellKind::Some(PieceType::O));
        let lock_a = no_clear_lock();
        let t_a = None;

        let mut board_b = Board::new(4, 6);
        board_b.set(0, 0, CellKind::Some(PieceType::O));
        let lock_b = LockOutcome {
            cells_locked: vec![(0, 0, CellKind::Some(PieceType::I))],
            cleared_rows: vec![0, 1, 2, 3],
            top_y_after_lock: None,
        };
        let t_b = None;

        let mut board_c = Board::new(4, 6);
        board_c.set(3, 0, CellKind::Some(PieceType::O));
        let lock_c = LockOutcome {
            cells_locked: vec![(0, 0, CellKind::Some(PieceType::T))],
            cleared_rows: vec![0, 1],
            top_y_after_lock: None,
        };
        let t_c = Some(TSpinKind::Full);

        let ctx = EvalContext::default();
        let bb_a = crate::engine::BitBoard::from_board(&board_a);
        let bb_b = crate::engine::BitBoard::from_board(&board_b);
        let bb_c = crate::engine::BitBoard::from_board(&board_c);
        let inputs: Vec<(
            &LockOutcome,
            crate::engine::ColumnView,
            Option<TSpinKind>,
            EvalContext,
        )> = vec![
            (&lock_a, bb_a.view(), t_a, ctx),
            (&lock_b, bb_b.view(), t_b, ctx),
            (&lock_c, bb_c.view(), t_c, ctx),
        ];

        // The cols fast path must equal scalar `evaluate` on the equivalent
        // dense boards — the differential that keeps the override honest.
        let cols: Vec<_> = inputs
            .iter()
            .map(|(l, b, t, ctx)| eval.evaluate_cols(l, *b, *t, *ctx))
            .collect();
        let scalar = vec![
            eval.evaluate(&lock_a, &board_a, t_a, ctx),
            eval.evaluate(&lock_b, &board_b, t_b, ctx),
            eval.evaluate(&lock_c, &board_c, t_c, ctx),
        ];

        assert_eq!(cols, scalar);
    }
}
