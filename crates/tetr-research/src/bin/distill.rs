//! `distill` — Rust bootstrap trainer (no Python needed).
//!
//! Trains the tetr-nn [`ValueNet`] to reproduce the DT-20 linear evaluator's board
//! `Value` (`value = DT20 · features`) via Burn autodiff, then exports the weights
//! to `crates/tetr-nn/assets/value_net.safetensors`. This gives an NN that starts
//! at ~baseline parity — a sane launch point for `/autoresearch` to climb from —
//! and exercises the full train → safetensors → Burn-inference pipeline.
//!
//! The richer model-dev path (real gameplay data, RL / self-play) is the JAX
//! script `training/train_value_net.py`; this is the dependency-free bootstrap.
//!
//! Run: `cargo run --release -p tetr-research --bin distill`

use std::path::PathBuf;

use burn::backend::{Autodiff, NdArray};
use burn::module::AutodiffModule;
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::tensor::{ElementConversion, Tensor, TensorData};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

use tetr_core::ai::eval::{BoardFeatures, BoardWeights};
use tetr_nn::{features_to_input, ValueNet, ValueNetConfig, NUM_FEATURES};

/// Autodiff-wrapped CPU backend for training.
type AB = Autodiff<NdArray<f32>>;

/// Upper bounds for synthetic feature sampling (matches the JAX script). Distilling
/// a *linear* target over this domain teaches the MLP the DT-20 value function.
const FEATURE_MAX: [f32; NUM_FEATURES] = [20.0, 16.0, 80.0, 40.0, 80.0, 80.0, 80.0, 20.0];

/// Sample a batch of (normalized input, DT-20 target) pairs.
fn sample_batch(rng: &mut StdRng, n: usize) -> (Vec<f32>, Vec<f32>) {
    let w = BoardWeights::DT20;
    let mut inputs = Vec::with_capacity(n * NUM_FEATURES);
    let mut targets = Vec::with_capacity(n);
    for _ in 0..n {
        let raw: [f32; NUM_FEATURES] =
            std::array::from_fn(|i| rng.random_range(0.0..=FEATURE_MAX[i]));
        let feats = BoardFeatures {
            landing_height: raw[0] as i32,
            eroded_piece_cells: raw[1] as i32,
            row_transitions: raw[2] as i32,
            column_transitions: raw[3] as i32,
            holes: raw[4] as i32,
            board_wells: raw[5] as i32,
            hole_depth: raw[6] as i32,
            rows_with_holes: raw[7] as i32,
            // The NN distills the 8-feature DT-20 target; `tetris_well` / `near_full_rows`
            // (DT-20 weight 0.0) are not part of that input space, so they sample to 0.
            tetris_well: 0,
            near_full_rows: 0,
        };
        // The exact value the baseline evaluator would assign.
        targets.push(w.dot(&feats) as f32);
        inputs.extend_from_slice(&features_to_input(&feats));
    }
    (inputs, targets)
}

fn main() {
    let device = Default::default();
    let mut rng = StdRng::seed_from_u64(0);

    let mut model: ValueNet<AB> = ValueNetConfig::default().init::<AB>(&device);
    let mut optim = AdamConfig::new().init::<AB, ValueNet<AB>>();

    let lr = 1e-3;
    let steps = 4000usize;
    let batch = 1024usize;

    for step in 0..steps {
        let (xs, ys) = sample_batch(&mut rng, batch);
        let x = Tensor::<AB, 2>::from_data(TensorData::new(xs, [batch, NUM_FEATURES]), &device);
        let y = Tensor::<AB, 2>::from_data(TensorData::new(ys, [batch, 1]), &device);

        let pred = model.forward(x);
        let diff = pred - y;
        let loss = (diff.clone() * diff).mean();

        if step % 500 == 0 || step == steps - 1 {
            let mse: f32 = loss.clone().into_scalar().elem();
            println!("step {step:5}  mse {mse:.3}");
        }

        let grads = loss.backward();
        let grads = GradientsParams::from_grads(grads, &model);
        model = optim.step(lr, model, grads);
    }

    // Drop autodiff for export; weights are backend-agnostic.
    let model = model.valid();
    let bytes = model
        .to_safetensors_bytes()
        .expect("serialize value net to safetensors");

    let out = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../tetr-nn/assets/value_net.safetensors");
    std::fs::create_dir_all(out.parent().unwrap()).unwrap();
    std::fs::write(&out, &bytes).unwrap();
    println!("wrote {} ({} bytes)", out.display(), bytes.len());
}
