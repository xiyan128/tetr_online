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
fn evals_per_decision() {
    // Configs to compare: the gate/datagen default and the cheaper-search
    // candidate. Override with TETR_BENCH_WD="w8d5,w4d3,w6d4" to sweep others.
    let wds = std::env::var("TETR_BENCH_WD").unwrap_or_else(|_| "w8d5,w4d3".into());
    eprintln!("\nleaves scored per decision (net path, avg over 20 spawned states):");
    let mut base: Option<f64> = None;
    for wd in wds.split(',') {
        let (w, d) = wd[1..].split_once('d').expect("wWdD");
        let (w, d): (usize, u8) = (w.parse().unwrap(), d.parse().unwrap());
        let mut tot = 0usize;
        for seed in 0..20u64 {
            let state = spawned(seed * 7 + 1);
            let ev = Counter {
                leaves: AtomicUsize::new(0),
                calls: AtomicUsize::new(0),
                board_only: false,
            };
            let mut beam = BeamPlanner::new(w);
            think_to_completion(&mut beam, &state, &ev, SearchBudget::beam(d));
            tot += ev.leaves.load(Ordering::Relaxed);
        }
        let per = tot as f64 / 20.0;
        let b = *base.get_or_insert(per);
        eprintln!(
            "  {wd:<6}  {per:>7.0} leaves/decision   ({:.2}x cheaper than {})",
            b / per,
            wds.split(',').next().unwrap()
        );
    }
    eprintln!(
        "  (leaves/decision * ~300 decisions/game * ~150µs/eval = the datagen wall/game.\n   \
         The cheaper-search ratio bounds the w4d3 A/B's datagen speedup.)"
    );
}
