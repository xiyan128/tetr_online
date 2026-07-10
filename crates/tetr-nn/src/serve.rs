//! The bridge into the search: [`NetEvaluator`] implements tetr-core's
//! [`Evaluator`] over the net.
//!
//! The score is pure: `Value = z_scale · z_hat(leaf)`, `Reward = 0` — the
//! net's win-probability estimate quantized to the beam's integer domain, with
//! no hand-tuned attack or board terms anywhere in the composition. Scoring
//! goes through [`Evaluator::evaluate_leaves`], so a sibling group is one
//! batched forward.
//!
//! One evaluator per game/worker thread is the intended shape (internal state
//! is mutex-held for trait object safety, not for cross-thread sharing).

use std::path::Path;
use std::sync::Mutex;

use tetr_core::ai::eval::{EvalContext, Evaluator, Leaf, Reward, Value};
use tetr_core::engine::{Board, LockOutcome, TSpinKind};

use crate::net::{LoadError, Net, Scratch};
use crate::obs::{BOARD_LEN, FEATURE_LEN, Obs, encode};

/// The net behind tetr-core's [`Evaluator`] seam.
pub struct NetEvaluator {
    net: Net,
    /// Forward scratch. The lock exists only for the interior mutability the
    /// `&self` [`Evaluator`] methods need.
    scratch: Mutex<Scratch>,
}

impl NetEvaluator {
    /// Wrap a loaded net.
    pub fn new(net: Net) -> Self {
        Self {
            net,
            scratch: Mutex::new(Scratch::default()),
        }
    }

    /// Load a model dir and wrap it.
    pub fn load(dir: impl AsRef<Path>) -> Result<Self, LoadError> {
        Ok(Self::new(Net::load(dir)?))
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

    /// The serving contract: encode every leaf, run ONE batched forward,
    /// quantize each z_hat. `board_only` stays false (the default) — the net
    /// reads the queue, bag, and pending state, so beam speculation must fan
    /// per continuation rather than share one score.
    fn evaluate_leaves(&self, leaves: &[Leaf<'_>]) -> Vec<(Value, Reward)> {
        if leaves.is_empty() {
            return Vec::new();
        }
        let mut scratch = self.scratch.lock().expect("net evaluator");
        let obs: Vec<Obs> = leaves.iter().map(|l| encode(l.state)).collect();
        let items: Vec<(&[f32; BOARD_LEN], &[f32; FEATURE_LEN])> =
            obs.iter().map(|o| (&o.board, &o.features)).collect();
        let z_scale = self.net.contract.z_scale;
        self.net
            .forward(&items, &mut scratch)
            .iter()
            .map(|h| (Value((z_scale * h.z_hat()).round() as i32), Reward(0)))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tetr_core::ai::search::{BeamPlanner, SearchBudget, think_to_completion};
    use tetr_core::ai::state::SearchState;
    use tetr_core::engine::{Engine, EngineConfig, InputFrame};

    fn fixture() -> NetEvaluator {
        NetEvaluator::load(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/pyref"))
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
}
