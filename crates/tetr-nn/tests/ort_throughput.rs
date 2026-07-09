//! CoreML backend throughput (feature coreml). Run:
//!   cargo test --release -p tetr-nn --features coreml --test ort_throughput -- --ignored --nocapture
#![cfg(feature = "coreml")]

use std::time::Instant;
use tetr_nn::obs::{BOARD_LEN, FEATURE_LEN, Obs};
use tetr_nn::ort_backend::OrtBackend;

#[test]
#[ignore]
fn ort_throughput_sweep() {
    let dir =
        std::env::var("ORT_MODEL_DIR").expect("ORT_MODEL_DIR=<model dir with net_leaf_b*.onnx>");
    let backend = OrtBackend::load(&dir).expect("backend loads");
    let obs = Obs {
        own_board: [0.0; BOARD_LEN],
        opp_board: [0.0; BOARD_LEN],
        features: [0.0; FEATURE_LEN],
    };
    for &n in &[12usize, 34, 68, 128, 480] {
        let batch: Vec<&Obs> = (0..n).map(|_| &obs).collect();
        for _ in 0..3 {
            std::hint::black_box(backend.forward_batch(&batch));
        }
        let t0 = Instant::now();
        let mut iters = 0u64;
        while t0.elapsed().as_secs_f64() < 0.5 {
            std::hint::black_box(backend.forward_batch(&batch));
            iters += 1;
        }
        let eps = iters as f64 * n as f64 / t0.elapsed().as_secs_f64();
        println!("batch {n:>4}: {eps:>10.0} evals/s");
    }
}
