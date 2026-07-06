//! The bridge into the search: [`NetEvaluator`] implements tetr-core's
//! [`Evaluator`] over the net.
//!
//! Score composition (the model's exported contract):
//!
//! ```text
//! Value  = z_scale · z_hat(leaf)          — the net's estimate of the future
//! Reward = attack_w · attack_reward(leaf) — the engine-truth present payoff
//! ```
//!
//! The opponent context is **frozen per decision**: the driver calls
//! [`NetEvaluator::set_opponent`] once when a decision starts (the same
//! freezing contract pending garbage uses), which embeds the opponent plane
//! once; every leaf of that decision's search then reuses the cached
//! embedding — the siamese saving that makes a two-board value cost about one
//! board per leaf. Scoring goes through [`Evaluator::evaluate_leaves`], so a
//! sibling group is one batched forward.
//!
//! One evaluator per game/worker thread is the intended shape (internal state
//! is mutex-held for trait object safety, not for cross-thread sharing).

use std::path::Path;
use std::sync::Mutex;

use tetr_core::ai::eval::{EvalContext, Evaluator, Leaf, Reward, Value, attack_reward};
use tetr_core::engine::{Board, LockOutcome, TSpinKind};

use crate::net::{BoardEmb, LoadError, Net, Scratch};
use crate::obs::{BOARD_LEN, FEATURE_LEN, Obs, OppCtx, encode};

/// The net behind tetr-core's [`Evaluator`] seam.
pub struct NetEvaluator {
    net: Net,
    /// The frozen decision context (opponent + its cached embedding) and the
    /// forward scratch, under ONE lock. The intended shape is one evaluator per
    /// worker thread; the lock exists only for the interior mutability the
    /// `&self` [`Evaluator`] methods need, and keeping it a single lock makes a
    /// lock-ordering deadlock impossible even if one is ever shared.
    inner: Mutex<Inner>,
}

/// The mutable serving state guarded by [`NetEvaluator::inner`].
struct Inner {
    opp: OppCtx,
    opp_emb: BoardEmb,
    scratch: Scratch,
}

impl NetEvaluator {
    /// Wrap a loaded net with a neutral (solo) opponent context.
    pub fn new(net: Net) -> Self {
        let mut scratch = Scratch::default();
        let opp = OppCtx::default();
        let opp_emb = net
            .embed_boards(&[&opp.board], &mut scratch)
            .pop()
            .expect("one plane in, one embedding out");
        Self {
            net,
            inner: Mutex::new(Inner {
                opp,
                opp_emb,
                scratch,
            }),
        }
    }

    /// Load a model dir and wrap it.
    pub fn load(dir: impl AsRef<Path>) -> Result<Self, LoadError> {
        Ok(Self::new(Net::load(dir)?))
    }

    /// Freeze the opponent context for the coming decision: embeds the plane
    /// once; every leaf scored until the next call reuses the embedding.
    pub fn set_opponent(&self, ctx: OppCtx) {
        let mut g = self.inner.lock().expect("net evaluator");
        let emb = self
            .net
            .embed_boards(&[&ctx.board], &mut g.scratch)
            .pop()
            .expect("one plane in, one embedding out");
        g.opp = ctx;
        g.opp_emb = emb;
    }

    /// The exported leaf contract (z_scale, attack_w).
    pub fn contract(&self) -> crate::net::Contract {
        self.net.contract
    }

    fn compose(&self, z_hat: f32, atk: Reward) -> (Value, Reward) {
        let c = self.net.contract;
        (
            Value((c.z_scale * z_hat).round() as i32),
            Reward((c.attack_w * atk.0 as f32).round() as i32),
        )
    }
}

impl Evaluator for NetEvaluator {
    /// Not a serving path. The net conditions on the full observation (queue,
    /// hold, bag, incoming garbage), which a bare [`Board`] cannot supply — so a
    /// net is deliberately NOT a [`board_only`](Evaluator::board_only) evaluator
    /// and is only ever scored through
    /// [`evaluate_leaves`](Evaluator::evaluate_leaves), which carries the
    /// [`SearchState`](tetr_core::ai::SearchState). Routing a net through the
    /// single-board path is a wiring bug; fail loudly rather than return a
    /// silently degraded score.
    fn evaluate(
        &self,
        _lock: &LockOutcome,
        _board: &Board,
        _t_spin: Option<TSpinKind>,
        _ctx: EvalContext,
    ) -> (Value, Reward) {
        panic!(
            "NetEvaluator scores via evaluate_leaves (it needs the full SearchState); \
             it cannot score a bare Board"
        )
    }

    /// The serving contract: encode every leaf under the frozen opponent
    /// context, run ONE batched forward, compose each score. `board_only` stays
    /// false (the default) — the net reads the queue, bag, and pending state, so
    /// beam speculation must fan per continuation rather than share one score.
    fn evaluate_leaves(&self, leaves: &[Leaf<'_>]) -> Vec<(Value, Reward)> {
        if leaves.is_empty() {
            return Vec::new();
        }
        let mut guard = self.inner.lock().expect("net evaluator");
        let inner = &mut *guard;
        let obs: Vec<Obs> = leaves.iter().map(|l| encode(l.state, &inner.opp)).collect();
        let items: Vec<(&[f32; BOARD_LEN], &[f32; FEATURE_LEN])> =
            obs.iter().map(|o| (&o.own_board, &o.features)).collect();
        let heads = self.net.forward(&items, &inner.opp_emb, &mut inner.scratch);
        heads
            .iter()
            .zip(leaves)
            .map(|(h, leaf)| self.compose(h.z_hat(), attack_reward(leaf)))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::obs::BOARD_W;
    use tetr_core::ai::search::{BeamPlanner, SearchBudget, think_to_completion};
    use tetr_core::ai::state::SearchState;
    use tetr_core::engine::{Engine, EngineConfig, InputFrame};

    fn fixture() -> NetEvaluator {
        NetEvaluator::load(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/round0"
        ))
        .expect("fixture model loads")
    }

    fn spawned(seed: u64) -> SearchState {
        let mut engine = Engine::new(EngineConfig::default(), seed);
        engine.step(InputFrame::default());
        SearchState::from_snapshot(&engine.snapshot()).expect("active piece present")
    }

    #[test]
    fn beam_drives_the_net_deterministically() {
        let state = spawned(11);
        let eval = fixture();
        let mut a = BeamPlanner::new(8);
        let mut b = BeamPlanner::new(8);
        let pa = think_to_completion(&mut a, &state, &eval, SearchBudget::beam(3)).unwrap();
        let pb = think_to_completion(&mut b, &state, &eval, SearchBudget::beam(3)).unwrap();
        assert_eq!(pa.placement.origin(), pb.placement.origin());
        assert_eq!(pa.placement.rotation(), pb.placement.rotation());
        assert_eq!(pa.score, pb.score);
    }

    #[test]
    fn the_opponent_actually_changes_the_value() {
        // A two-board net that ignores its opponent half would be a silent
        // regression to solo — pin that a pressured opponent context moves
        // scores.
        let state = spawned(11);
        let eval = fixture();
        let mut planner = BeamPlanner::new(8);
        let calm = think_to_completion(&mut planner, &state, &eval, SearchBudget::beam(2)).unwrap();

        let mut pressured = OppCtx::default();
        for x in 0..BOARD_W {
            for y in 0..18 {
                if x != 4 {
                    pressured.board[y * BOARD_W + x] = 1.0;
                }
            }
        }
        pressured.combo = 5;
        pressured.b2b = true;
        pressured.pending = vec![(4, 2)];
        eval.set_opponent(pressured);
        let mut planner2 = BeamPlanner::new(8);
        let tense =
            think_to_completion(&mut planner2, &state, &eval, SearchBudget::beam(2)).unwrap();
        assert_ne!(
            calm.score, tense.score,
            "opponent context must reach the value head"
        );
    }
}
