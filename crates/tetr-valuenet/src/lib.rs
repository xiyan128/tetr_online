//! The game's deployed learned evaluator.
//!
//! A pure-Rust single-board **value** net (the wasm-ready ship kernel) behind
//! tetr-core's [`Evaluator`] seam. Each afterstate scores as a *composed* leaf:
//!
//! - **Value** = the learned net's board estimate (de-standardised, rounded),
//! - **Reward** = the per-move payoff from the hand-tuned [`Cc2Evaluator`].
//!
//! So the net replaces only the static board judgement; the immediate attack of
//! a move stays engine-exact via CC2. That is the composition the shipped
//! weights were trained and tuned under, reproduced here.
//!
//! This crate is deliberately separate from `tetr-nn` (the two-board *research*
//! net): different architecture, different job, and frozen to reproduce the
//! deployed weights rather than to evolve. When the research campaign produces a
//! two-board net worth deploying, the game moves onto `tetr-nn` and this crate
//! retires.

pub mod encode;
pub mod net;

use std::sync::Mutex;

use tetr_core::ai::eval::{Cc2Evaluator, Cc2Weights, EvalContext, Evaluator, Leaf, Reward, Value};
use tetr_core::engine::{Board, LockOutcome, TSpinKind};

pub use net::{LoadError, ValueNet};

/// The deployed evaluator: the value net + the CC2 reward source + forward
/// scratch, behind one lock (one evaluator per game/worker; the lock is for the
/// `&self` trait methods' interior mutability, not cross-thread sharing).
pub struct DeployNet {
    net: ValueNet,
    cc2: Cc2Evaluator,
    scratch: Mutex<net::Scratch>,
}

impl DeployNet {
    /// Wrap a loaded value net (CC2 attack-tuned reward, matching the trained
    /// composition).
    pub fn new(net: ValueNet) -> Self {
        let scratch = Mutex::new(net.scratch());
        Self {
            net,
            cc2: Cc2Evaluator::new(Cc2Weights::attack_tuned()),
            scratch,
        }
    }

    /// Load a model dir (`value_net.safetensors` + `config.json`) and wrap it.
    pub fn load(dir: impl AsRef<std::path::Path>) -> Result<Self, LoadError> {
        Ok(Self::new(ValueNet::load(dir)?))
    }
}

impl Evaluator for DeployNet {
    /// Not a serving path. The net conditions on the full observation (queue,
    /// hold, bag, pending garbage), which a bare [`Board`] cannot supply — so
    /// this evaluator is never `board_only` and is scored only through
    /// [`evaluate_leaves`](Evaluator::evaluate_leaves), which carries the
    /// `SearchState`. Routing it through the single-board path is a wiring bug.
    fn evaluate(
        &self,
        _lock: &LockOutcome,
        _board: &Board,
        _t_spin: Option<TSpinKind>,
        _ctx: EvalContext,
    ) -> (Value, Reward) {
        panic!(
            "DeployNet scores via evaluate_leaves (it needs the full SearchState); \
             it cannot score a bare Board"
        )
    }

    /// Score each leaf: the learned board [`Value`] plus the CC2 per-move
    /// [`Reward`]. `board_only` stays false (the default) — the net reads the
    /// queue/hold/bag/pending, so beam speculation fans per continuation.
    fn evaluate_leaves(&self, leaves: &[Leaf<'_>]) -> Vec<(Value, Reward)> {
        let mut scratch = self.scratch.lock().expect("value-net scratch");
        leaves
            .iter()
            .map(|leaf| {
                let obs = encode::encode(leaf.state);
                let value = self.net.forward_into(&obs, &mut scratch);
                // The reward half is the hand evaluator's, exactly as trained;
                // it reads the post-clear board for perfect-clear detection.
                let reward = self
                    .cc2
                    .evaluate_cols(leaf.lock, leaf.state.board.view(), leaf.t_spin, leaf.ctx)
                    .1;
                (Value(value.round() as i32), reward)
            })
            .collect()
    }
}
