//! End-to-end: the deployed evaluator actually drives a beam decision through
//! the `evaluate_leaves` seam (encode → forward → compose → rank), not just
//! constructs. This is the path the game exercises every move.

use tetr_core::ai::search::{BeamPlanner, SearchBudget, think_to_completion};
use tetr_core::ai::state::SearchState;
use tetr_core::engine::{Engine, EngineConfig, InputFrame};
use tetr_valuenet::DeployNet;

const MODEL: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../models/conv_rb1");

fn spawned(seed: u64) -> SearchState {
    let mut engine = Engine::new(EngineConfig::default(), seed);
    engine.step(InputFrame::default());
    SearchState::from_snapshot(&engine.snapshot()).expect("active piece present")
}

#[test]
fn deploy_net_drives_a_beam_deterministically() {
    let eval = DeployNet::load(MODEL).expect("conv_rb1 loads");
    let state = spawned(7);
    let mut a = BeamPlanner::new(16);
    let mut b = BeamPlanner::new(16);
    let pa = think_to_completion(&mut a, &state, &eval, SearchBudget::beam(2)).expect("a plan");
    let pb = think_to_completion(&mut b, &state, &eval, SearchBudget::beam(2)).expect("a plan");
    // Deterministic: the learned evaluator is a pure function of the state.
    assert_eq!(pa.placement.origin(), pb.placement.origin());
    assert_eq!(pa.placement.rotation(), pb.placement.rotation());
    assert_eq!(pa.score, pb.score);
}
