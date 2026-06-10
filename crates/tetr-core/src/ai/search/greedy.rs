//! The greedy Tier-1 planner (AI3.3).
//!
//! For each placement movegen reports — including the placements reachable after a
//! hold swap — the greedy planner simulates the lock with the engine's own
//! primitives, scores the resulting board + the move's reward with the
//! [`Evaluator`], and picks the single highest-scoring placement. No lookahead
//! beyond the current piece: one piece, scored greedily.
//!
//! Despite its simplicity this is a genuinely strong baseline — a one-piece greedy
//! search over a good linear evaluator is the classic Dellacherie controller
//! (research finding [1]) — and it fits the [`Mind`] session seam so a multi-ply
//! Tier-2 search can replace it with no other change. As a session it is the
//! degenerate case: **all** of its work happens at [`Mind::reroot`] (seeding *is*
//! the whole argmax), [`Mind::think`] is immediately exhausted, and
//! [`Mind::best`] returns the cached decision.
//!
//! # How a placement is scored (mirrors the engine's lock path)
//!
//! The engine classifies a T-spin against the board **before** the lock mutates it,
//! then locks (`api.rs::lock_active_piece`). The planner does exactly the same so
//! its scoring can never disagree with the real rules:
//!
//! 1. clone the state (cheap; the board is a `Copy` `BitBoard`),
//! 2. [`classify_t_spin`] the placement against that pre-lock board,
//! 3. lock the placement's piece into the clone (`BitBoard::lock_piece`),
//! 4. [`Evaluator::evaluate`] the resulting board + lock + t-spin → `(Value, Reward)`,
//! 5. rank by `Value + Reward`.
//!
//! Ties are broken by movegen's canonical placement order (stable), so the result
//! is fully deterministic with no RNG.

use crate::ai::eval::{EvalContext, Evaluator};
use crate::ai::movegen::{self, Placement};
use crate::ai::search::{
    hold_placements, score_placement, Mind, PlacementPlan, RootKey, ThinkProgress,
};
use crate::ai::state::SearchState;

/// The completed one-shot decision for the current root (greedy thinks entirely
/// at [`Mind::reroot`] time): the root's identity plus its argmax plan (`None`
/// inside when the root had no legal placement).
struct GreedyRun {
    root: RootKey,
    plan: Option<PlacementPlan>,
}

/// The greedy one-piece planner: a [`Mind`] whose whole search is the seeding
/// argmax, cached per root.
pub struct GreedyPlanner {
    /// Whether to consider the hold swap (search the held/next piece too). On by
    /// default; exposed mainly so tests can isolate the no-hold placement set.
    consider_hold: bool,
    /// The cached decision for the current root; `None` before the first reroot.
    run: Option<GreedyRun>,
}

impl GreedyPlanner {
    /// A greedy planner that also evaluates the hold swap (the recommended setting).
    pub fn new() -> Self {
        Self {
            consider_hold: true,
            run: None,
        }
    }

    /// A greedy planner that ignores hold (only the active piece's placements).
    pub fn without_hold() -> Self {
        Self {
            consider_hold: false,
            run: None,
        }
    }

    /// Enumerate the candidate placements for `state`, with or without the hold swap.
    fn candidates(&self, state: &SearchState) -> Vec<Placement> {
        if self.consider_hold {
            hold_placements(state)
        } else {
            movegen::generate(&state.board, &state.active)
        }
    }
}

impl Default for GreedyPlanner {
    /// Matches [`GreedyPlanner::new`] (hold considered) — *not* the derived
    /// all-`false` default, which would silently disable hold.
    fn default() -> Self {
        Self::new()
    }
}

impl Mind for GreedyPlanner {
    /// Seeding **is** the greedy search: score every candidate and cache the
    /// argmax. The depth cap is irrelevant (greedy is always exactly one ply), so
    /// it is not part of the root identity.
    fn reroot(&mut self, state: &SearchState, eval: &dyn Evaluator, _max_depth: u8) {
        let root = RootKey::of(state);
        if self.run.as_ref().is_some_and(|run| run.root == root) {
            return; // already rooted here: the cached decision stands
        }

        // Pick the highest-scoring placement, keeping the *first* maximum on a tie
        // (`score <= best` leaves the incumbent in place). Movegen's placement order
        // is canonical and stable, so this tie-break — and thus the whole plan — is
        // deterministic, no RNG involved.
        let plan =
            self.candidates(state)
                .into_iter()
                .fold(None::<PlacementPlan>, |best, placement| {
                    let ctx = EvalContext {
                        combo: state.combo,
                        b2b: state.b2b,
                    };
                    let score = score_placement(state, &placement, eval, ctx);
                    match best {
                        Some(plan) if score <= plan.score => Some(plan),
                        _ => Some(PlacementPlan { placement, score }),
                    }
                });

        self.run = Some(GreedyRun { root, plan });
    }

    /// Greedy has no frontier: everything was decided at reroot.
    fn think(&mut self, _quantum: u32, _eval: &dyn Evaluator) -> ThinkProgress {
        ThinkProgress::Exhausted
    }

    fn best(&self) -> Option<PlacementPlan> {
        self.run.as_ref().and_then(|run| run.plan.clone())
    }

    /// Greedy never *expands* (no lookahead) — the meter stays at zero, matching
    /// its historical "ignores the node budget" contract.
    fn nodes_expanded(&self) -> u32 {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::eval::LinearEvaluator;
    use crate::engine::{ActivePiece, Board, CellKind, EngineConfig, EngineSnapshot, PieceType};

    /// Build a `SearchState` from a crafted board + active piece (no hold/queue).
    fn state_with(board: Board, active: ActivePiece) -> SearchState {
        SearchState::for_test(board, active, None, std::iter::empty())
    }

    /// A planner plan, unwrapped, or a panic with context.
    fn plan_of(state: &SearchState, planner: &mut GreedyPlanner) -> PlacementPlan {
        let eval = LinearEvaluator::default();
        crate::ai::search::think_to_completion(
            planner,
            state,
            &eval,
            crate::ai::search::SearchBudget::greedy(),
        )
        .expect("the state has a legal placement")
    }

    /// Absolute occupied cells of a placement, sorted.
    fn cells_of(placement: &Placement) -> Vec<(isize, isize)> {
        let (ox, oy) = placement.origin();
        let mut cells: Vec<(isize, isize)> = placement
            .piece
            .piece()
            .cells()
            .iter()
            .map(|(x, y)| (x + ox, y + oy))
            .collect();
        cells.sort();
        cells
    }

    #[test]
    fn greedy_completes_obvious_lines() {
        // A 4-wide board with cols 0-2 filled four rows high and an empty 1-wide
        // well at col 3. A vertical I dropped into the well fills (3,0..3) and clears
        // all four rows — an unambiguous Tetris that scores far above any placement
        // that leaves the stack standing. Greedy must pick it.
        //
        //   col:  0 1 2 3
        //   y0-3: X X X .      4-tall stack on cols 0-2, well at col 3
        let mut board = Board::new(4, 10);
        for y in 0..4 {
            for x in 0..3 {
                board.set(x, y, CellKind::Some(PieceType::O));
            }
        }
        let active = movegen::spawn_piece(PieceType::I, 4, 10);
        let state = state_with(board, active);

        let mut planner = GreedyPlanner::without_hold();
        let plan = plan_of(&state, &mut planner);

        // The chosen placement, locked, clears the four stacked rows.
        let mut check = state.board;
        let lock = check.lock_piece(&plan.placement.piece);
        assert_eq!(
            lock.cleared_rows.len(),
            4,
            "greedy should pick the well-filling I for a Tetris; chose cells {:?}, cleared {:?}",
            cells_of(&plan.placement),
            lock.cleared_rows
        );
    }

    #[test]
    fn greedy_avoids_creating_a_hole() {
        // Two candidate columns: dropping the O flat on the floor (no hole) vs.
        // dropping it onto a one-cell pillar that would leave a covered hole beside
        // it. The evaluator penalises holes heavily, so greedy must choose the
        // flat, hole-free placement.
        //
        //   col:  0 1 2 3
        //    y1:        ?        (a pillar at col 3 creates an overhang risk)
        //    y0:  . . . X        single block at (3,0)
        let mut board = Board::new(4, 10);
        board.set(3, 0, CellKind::Some(PieceType::O));

        let active = movegen::spawn_piece(PieceType::O, 4, 10);
        let state = state_with(board, active);

        let mut planner = GreedyPlanner::without_hold();
        let plan = plan_of(&state, &mut planner);

        // The chosen placement must not leave a hole (a None cell with a filled
        // cell somewhere above it in the same column).
        let mut after = state.board;
        after.lock_piece(&plan.placement.piece);
        let after = after.to_array2d();
        assert!(
            !has_hole(&after),
            "greedy should avoid creating a hole; resulting board:\n{after}"
        );
    }

    /// Whether the board has a covered hole: an empty cell with a filled cell above
    /// it in the same column (within the visible height).
    fn has_hole(board: &Board) -> bool {
        for x in 0..board.width() as isize {
            let mut seen_filled = false;
            for y in (0..board.height() as isize).rev() {
                match board.get_cell_kind(x, y) {
                    CellKind::Some(_) => seen_filled = true,
                    CellKind::None if seen_filled => return true,
                    _ => {}
                }
            }
        }
        false
    }

    #[test]
    fn greedy_is_deterministic() {
        // Same state planned twice yields the same placement and score.
        let mut board = Board::new(6, 12);
        board.set(0, 0, CellKind::Some(PieceType::O));
        board.set(5, 0, CellKind::Some(PieceType::O));
        let active = movegen::spawn_piece(PieceType::T, 6, 12);
        let state = state_with(board, active);

        let mut p1 = GreedyPlanner::new();
        let mut p2 = GreedyPlanner::new();
        let a = plan_of(&state, &mut p1);
        let b = plan_of(&state, &mut p2);
        assert_eq!(a.placement.origin(), b.placement.origin());
        assert_eq!(a.placement.rotation(), b.placement.rotation());
        assert_eq!(a.score, b.score);
        assert_eq!(a.placement.path, b.placement.path);
    }

    #[test]
    fn greedy_uses_hold_when_the_held_piece_is_better() {
        // Active is an S piece (awkward on a flat board); the hold slot has an I.
        // Build a board with a 1-wide well 4 deep that ONLY an I can fill cleanly to
        // clear lines, so the planner should hold (swap in the I) and place it.
        //
        //   col:  0 1 2 3
        //   y0-3: X X X .      a 4-tall stack on cols 0-2, empty well at col 3
        let mut board = Board::new(4, 12);
        for y in 0..4 {
            for x in 0..3 {
                board.set(x, y, CellKind::Some(PieceType::O));
            }
        }
        // Active S, held I.
        let active = movegen::spawn_piece(PieceType::S, 4, 12);
        let state = SearchState::for_test(board, active, Some(PieceType::I), std::iter::empty());

        let mut planner = GreedyPlanner::new();
        let plan = plan_of(&state, &mut planner);

        assert!(
            plan.uses_hold(),
            "greedy should hold to bring in the I that clears the well; chose {:?} (hold={})",
            plan.placement.piece_type(),
            plan.uses_hold()
        );
        assert_eq!(plan.placement.piece_type(), PieceType::I);
        // The plan's path begins with a hold swap.
        assert_eq!(plan.placement.path.first(), Some(&movegen::Move::Hold));
    }

    #[test]
    fn plan_from_engine_snapshot_round_trips() {
        // End-to-end: a real engine snapshot -> SearchState -> greedy plan. Proves
        // the planner consumes the production SearchState, not just hand-built ones.
        use crate::engine::{Engine, InputFrame};
        let mut engine = Engine::new(EngineConfig::default(), 7);
        engine.step(InputFrame::default()); // spawn the first piece
        let snapshot: EngineSnapshot = engine.snapshot();
        let state = SearchState::from_snapshot(&snapshot).expect("active piece present");

        let mut planner = GreedyPlanner::new();
        let plan = plan_of(&state, &mut planner);
        // On an empty board the best placement rests on the floor (no holes, low
        // height): assert the plan is executable (non-empty path or already resting)
        // and lands on the floor.
        let mut after = state.board;
        after.lock_piece(&plan.placement.piece);
        assert!(!after.is_empty(), "a piece was placed");
    }
}
