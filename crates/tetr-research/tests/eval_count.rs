//! How many net forwards does ONE w8d5 decision cost? The net is the whole
//! datagen bottleneck, so evals/decision × decisions/game × µs/eval = the
//! wall. This counts leaves scored for a real spawned decision, for a
//! board_only=false evaluator (the NET's path — every speculative bag
//! continuation scored separately) vs board_only=true (CC2's path — one eval
//! shared across the ≤7 continuations). The ratio is the speculation penalty
//! the net pays that CC2 does not.
//!
//! Run: cargo test --release -p tetr-research --test eval_count -- --ignored --nocapture

use std::sync::atomic::{AtomicUsize, Ordering};
use tetr_core::ai::eval::{EvalContext, Evaluator, Leaf, Reward, Value};
use tetr_core::ai::search::{BeamPlanner, SearchBudget, think_to_completion};
use tetr_core::ai::state::SearchState;
use tetr_core::engine::{Board, Engine, EngineConfig, InputFrame, LockOutcome, TSpinKind};

struct Counter {
    leaves: AtomicUsize,
    calls: AtomicUsize,
    board_only: bool,
}

impl Evaluator for Counter {
    fn evaluate(
        &self,
        _l: &LockOutcome,
        _b: &Board,
        _t: Option<TSpinKind>,
        _c: EvalContext,
    ) -> (Value, Reward) {
        self.leaves.fetch_add(1, Ordering::Relaxed);
        self.calls.fetch_add(1, Ordering::Relaxed);
        (Value(0), Reward(0))
    }
    fn board_only(&self) -> bool {
        self.board_only
    }
    fn evaluate_leaves(&self, leaves: &[Leaf<'_>]) -> Vec<(Value, Reward)> {
        // Mimic the net: one batched call per sibling group.
        self.calls.fetch_add(1, Ordering::Relaxed);
        self.leaves.fetch_add(leaves.len(), Ordering::Relaxed);
        vec![(Value(0), Reward(0)); leaves.len()]
    }
}

fn spawned(seed: u64) -> SearchState {
    let mut e = Engine::new(EngineConfig::default(), seed);
    e.step(InputFrame::default());
    SearchState::from_snapshot(&e.snapshot()).expect("active piece")
}

#[test]
#[ignore]
fn evals_per_w8d5_decision() {
    eprintln!("\nleaves scored per w8d5 decision (avg over 20 spawned states):");
    for &(tag, board_only) in &[
        ("net path (board_only=false)", false),
        ("cc2 path (board_only=true)", true),
    ] {
        let (mut tot_leaves, mut tot_calls) = (0usize, 0usize);
        for seed in 0..20u64 {
            let state = spawned(seed * 7 + 1);
            let ev = Counter {
                leaves: AtomicUsize::new(0),
                calls: AtomicUsize::new(0),
                board_only,
            };
            let mut beam = BeamPlanner::new(8);
            think_to_completion(&mut beam, &state, &ev, SearchBudget::beam(5));
            tot_leaves += ev.leaves.load(Ordering::Relaxed);
            tot_calls += ev.calls.load(Ordering::Relaxed);
        }
        eprintln!(
            "  {tag:<30}  {:>7.0} leaves/decision  ({:>6.0} forward-calls)",
            tot_leaves as f64 / 20.0,
            tot_calls as f64 / 20.0
        );
    }
    eprintln!(
        "  (net/cc2 ratio = the speculation-sharing penalty the queue-conditioned\n   \
         net pays. At ~150µs/eval and ~300 decisions/game, leaves/decision*300\n   \
         *150µs = the datagen wall per game.)"
    );
}
