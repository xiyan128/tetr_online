//! Parity: the pure-Rust *forward* reproduces the deployed `conv_rb1` weights.
//!
//! `conv_rb1/golden.safetensors` carries PyTorch-produced cases —
//! (`board` [400], `features` [70]) → expected `value` — so matching them proves
//! this forward reproduces the trained model, not just itself. It pins the
//! forward given an already-encoded observation; the `encode` step (SearchState
//! → board+features) is a separate faithful port, exercised end-to-end by
//! `drives_beam.rs`. Weights and golden are committed under `models/conv_rb1`,
//! so this runs from a clean checkout.

use tetr_valuenet::ValueNet;
use tetr_valuenet::encode::{BOARD_LEN, FEATURE_LEN, Obs};

const MODEL: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../models/conv_rb1");

fn f32s(t: safetensors::tensor::TensorView<'_>) -> Vec<f32> {
    t.data()
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

#[test]
fn forward_matches_conv_rb1_golden() {
    let net = ValueNet::load(MODEL).expect("conv_rb1 loads");
    let buf = std::fs::read(format!("{MODEL}/golden.safetensors")).unwrap();
    let st = safetensors::SafeTensors::deserialize(&buf).unwrap();

    // The golden is a batch: board [N,1,40,10], features [N,70], value [N].
    let board = f32s(st.tensor("board").unwrap());
    let features = f32s(st.tensor("features").unwrap());
    let expected = f32s(st.tensor("value").unwrap());
    let n = expected.len();
    assert_eq!(board.len(), n * BOARD_LEN);
    assert_eq!(features.len(), n * FEATURE_LEN);
    assert!(n >= 16, "expected a real golden batch, got {n}");

    for i in 0..n {
        let mut obs = Obs::default();
        obs.board
            .copy_from_slice(&board[i * BOARD_LEN..(i + 1) * BOARD_LEN]);
        obs.features
            .copy_from_slice(&features[i * FEATURE_LEN..(i + 1) * FEATURE_LEN]);
        let got = net.forward(&obs);
        // Values live on the training target's scale (mean ≈ -9347, std ≈ 3494),
        // so a few units of float-rounding slack is < 1e-3 relative.
        assert!(
            (got - expected[i]).abs() < 2.0,
            "conv_rb1 case {i}: got {got}, expected {}",
            expected[i]
        );
    }
}
