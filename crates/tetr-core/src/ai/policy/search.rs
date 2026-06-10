//! [`SearchPolicy`]: the search-based AI brain (any [`Mind`] paradigm — greedy,
//! beam, best-first — behind the same seam).
//!
//! It wraps a [`Mind`] + [`Evaluator`] and owns the **deliberate-error model**:
//! it plans the best placement, then with probability `imperfection` substitutes a
//! plausible near-best — a top-N softmax over candidate placements — so a
//! handicapped bot misplays like a human rather than flailing. This error model
//! lived in the controller; moving it into the policy is what makes the controller
//! model-agnostic (a neural policy would degrade via sampling temperature instead,
//! behind the same [`Policy::decide`] contract).
//!
//! # Determinism
//!
//! `imperfection` sampling draws from the policy's own seeded [`StdRng`]. With
//! `imperfection == 0` the policy is fully deterministic (always the planner's
//! best, tie-broken by movegen's canonical order). It never reads a clock or the
//! engine's RNG.

use rand::rngs::StdRng;
use rand::seq::IndexedRandom;
use rand::{RngExt, SeedableRng};

use crate::ai::eval::{EvalContext, Evaluator, LinearEvaluator};
use crate::ai::movegen;
use crate::ai::policy::{Decision, Observation, Policy};
use crate::ai::search::{
    score_placement, think_to_completion, GreedyPlanner, Mind, PlacementPlan, SearchBudget,
};

/// How many of the top placements the imperfection softmax samples from. A small
/// window keeps a "mistake" *plausible* (a near-best alternative) rather than
/// catastrophic. (Was `Handicap::ERROR_SAMPLE_WINDOW`; the error model lives here
/// now.)
const ERROR_SAMPLE_WINDOW: usize = 4;

/// A search brain: a [`Mind`] over an [`Evaluator`] under a [`SearchBudget`],
/// with a tunable `imperfection` that degrades it into a beatable opponent.
///
/// Owns the mind + evaluator because the mind carries an in-flight search between
/// calls (`&mut self`), so it cannot be borrowed across the runner's poll
/// boundary — the policy is its home.
pub struct SearchPolicy {
    mind: Box<dyn Mind>,
    evaluator: Box<dyn Evaluator>,
    budget: SearchBudget,
    /// `0.0..=1.0`: probability of substituting a softmax-sampled near-best
    /// placement for the planner's best — the deliberate-error handicap.
    imperfection: f32,
    /// The policy's own seeded RNG — imperfection sampling only. Never the engine's.
    rng: StdRng,
}

impl SearchPolicy {
    /// Build a search policy from an explicit mind + evaluator + budget. `seed`
    /// seeds the imperfection RNG (the determinism handle).
    pub fn new(
        mind: Box<dyn Mind>,
        evaluator: Box<dyn Evaluator>,
        budget: SearchBudget,
        imperfection: f32,
        seed: u64,
    ) -> Self {
        Self {
            mind,
            evaluator,
            budget,
            imperfection,
            rng: StdRng::seed_from_u64(seed),
        }
    }

    /// The shipped Tier-1 brain: a greedy planner over the default linear
    /// evaluator, with the given `imperfection` and RNG `seed`.
    pub fn greedy(imperfection: f32, seed: u64) -> Self {
        Self::new(
            Box::new(GreedyPlanner::new()),
            Box::new(LinearEvaluator::default()),
            SearchBudget::greedy(),
            imperfection,
            seed,
        )
    }

    /// Run the mind to its final best placement in one blocking call (the
    /// direct-drive venue — see [`think_to_completion`]).
    fn plan_best(&mut self, obs: &Observation) -> Option<PlacementPlan> {
        think_to_completion(
            self.mind.as_mut(),
            obs,
            self.evaluator.as_ref(),
            self.budget,
        )
    }

    /// Apply the imperfection handicap to the planner's `best`.
    ///
    /// With probability `imperfection`, re-score the candidate placements and
    /// softmax-sample one from the top window (a higher rate raises the temperature,
    /// flattening toward worse picks); otherwise return `best` unchanged. Scores
    /// with this policy's own evaluator so the sampled near-best is near-best under
    /// the metric the planner optimized. Uses `self.rng` only.
    fn apply_imperfection(&mut self, obs: &Observation, best: PlacementPlan) -> PlacementPlan {
        let rate = self.imperfection.clamp(0.0, 1.0);
        if rate <= 0.0 || !self.rng.random_bool(f64::from(rate)) {
            return best;
        }

        let mut scored = score_candidates(obs, self.evaluator.as_ref());
        if scored.len() <= 1 {
            return best; // nothing to substitute
        }
        // Highest score first; a *stable* sort keeps movegen's canonical order for
        // ties, so the sample set is reproducible before the (seeded) softmax draw.
        scored.sort_by_key(|c| std::cmp::Reverse(c.score));
        scored.truncate(ERROR_SAMPLE_WINDOW);

        let top = scored[0].score;
        let temperature = 1.0 + f64::from(rate) * 8.0;
        scored
            .choose_weighted(&mut self.rng, |c| {
                let delta = f64::from(c.score - top); // <= 0
                (delta / temperature).exp() // in (0, 1]
            })
            .cloned()
            .unwrap_or(best)
    }
}

impl Policy for SearchPolicy {
    fn decide(&mut self, obs: &Observation) -> Decision {
        match self.plan_best(obs) {
            Some(best) => Decision::Place(self.apply_imperfection(obs, best).placement),
            None => Decision::None,
        }
    }
}

/// Score every candidate placement for `obs` with `eval` (the same scorer the
/// planner ranks with — see [`score_placement`]), keeping all of them for the
/// imperfection softmax. Moved here from the controller along with the error model.
fn score_candidates(obs: &Observation, eval: &dyn Evaluator) -> Vec<PlacementPlan> {
    let candidates = movegen::generate_with_hold(
        &obs.board,
        &obs.active,
        obs.hold,
        obs.queue.first().copied(),
        |piece_type| movegen::spawn_piece(piece_type, obs.board.width(), obs.board.height()),
    );
    candidates
        .into_iter()
        .map(|placement| {
            // Score with the live chain state — the Observation IS the SearchState, so it
            // carries combo/B2B. This must match the planner's basis: a chain-sensitive
            // eval (e.g. CC2's combo_attack / B2B value) would otherwise rank candidates
            // here on a chain-stripped score and the imperfection sample would diverge
            // from the policy it is meant to perturb.
            let ctx = EvalContext {
                combo: obs.combo,
                b2b: obs.b2b,
            };
            let score = score_placement(obs, &placement, eval, ctx);
            PlacementPlan { placement, score }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{Board, CellKind, PieceType};

    /// A search state with an unambiguous best (vertical I clears a Tetris in the
    /// col-3 well).
    fn tetris_state() -> Observation {
        let mut board = Board::new(4, 10);
        for y in 0..4 {
            for x in 0..3 {
                board.set(x, y, CellKind::Some(PieceType::O));
            }
        }
        let active = movegen::spawn_piece(PieceType::I, 4, 10);
        crate::ai::state::SearchState::for_test(board, active, None, std::iter::empty())
    }

    fn placed(decision: Decision) -> crate::ai::movegen::Placement {
        match decision {
            Decision::Place(p) => p,
            Decision::None => panic!("expected a placement"),
        }
    }

    #[test]
    fn flawless_policy_is_deterministic_and_optimal() {
        // imperfection 0 ⇒ always the planner's best, no RNG path. Two policies with
        // *different* seeds must still agree (the seed is unused at imperfection 0).
        let mut a = SearchPolicy::greedy(0.0, 1);
        let mut b = SearchPolicy::greedy(0.0, 999);
        let pa = placed(a.decide(&tetris_state()));
        let pb = placed(b.decide(&tetris_state()));
        assert_eq!(pa.origin(), pb.origin());
        assert_eq!(pa.path, pb.path);
    }

    #[test]
    fn same_seed_same_decisions_under_imperfection() {
        // With imperfection > 0 the choice is randomized but seeded: same seed ⇒
        // identical decision sequence over repeated pieces.
        let state = tetris_state();
        let mut a = SearchPolicy::greedy(0.9, 42);
        let mut b = SearchPolicy::greedy(0.9, 42);
        for _ in 0..20 {
            let da = placed(a.decide(&state));
            let db = placed(b.decide(&state));
            assert_eq!(da.origin(), db.origin());
            assert_eq!(da.path, db.path);
        }
    }

    #[test]
    fn empty_board_still_decides() {
        let active = movegen::spawn_piece(PieceType::T, 6, 12);
        let state = crate::ai::state::SearchState::for_test(
            Board::new(6, 12),
            active,
            None,
            std::iter::empty(),
        );
        let mut policy = SearchPolicy::greedy(0.0, 7);
        let _ = placed(policy.decide(&state));
    }
}
