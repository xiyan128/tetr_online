//! Cross-language parity: the Rust forward against PyTorch-dumped golden
//! vectors on the `pyref` fixture — a small net emitted by *our own* `python/`
//! package (`uv run python -m tetrnn.regen_pyref`). Matching it proves the two
//! forwards agree on a model we can regenerate from source.

use tetr_nn::net::{Net, Scratch};
use tetr_nn::obs::{BOARD_LEN, FEATURE_LEN, Obs};

/// Load a fixture dir, forward every golden case, assert agreement to 1e-4.
fn check_fixture(dir: &str) {
    let net = Net::load(dir).expect("fixture model loads");
    let golden: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(format!("{dir}/golden_v2.json")).unwrap())
            .unwrap();

    let mut s = Scratch::default();
    let cases = golden["cases"].as_array().expect("cases array");
    assert_eq!(cases.len(), 16, "fixture shape drifted");
    for (i, case) in cases.iter().enumerate() {
        let arr = |k: &str| -> Vec<f32> {
            case[k]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap() as f32)
                .collect()
        };
        let mut obs = Obs {
            board: [0.0; BOARD_LEN],
            features: [0.0; FEATURE_LEN],
        };
        obs.board.copy_from_slice(&arr("board"));
        obs.features.copy_from_slice(&arr("features"));
        let expected = arr("out");

        let heads = net.forward_obs(&obs, &mut s);
        for (j, (g, e)) in heads.wdl.iter().zip(&expected).enumerate() {
            assert!(
                (g - e).abs() < 1e-4,
                "{dir} case {i} out[{j}]: rust {g} vs pytorch {e}"
            );
        }
    }
}

#[test]
fn forward_matches_our_python_package() {
    check_fixture(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/pyref"));
}

#[test]
fn batched_forward_matches_one_by_one() {
    // The forward is batched by design; batching must be invisible in values.
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/pyref");
    let net = Net::load(dir).expect("fixture model loads");
    let golden: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(format!("{dir}/golden_v2.json")).unwrap())
            .unwrap();
    let cases = golden["cases"].as_array().unwrap();

    let mut s = Scratch::default();
    let planes: Vec<[f32; BOARD_LEN]> = cases
        .iter()
        .map(|c| {
            let v: Vec<f32> = c["board"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap() as f32)
                .collect();
            let mut p = [0.0; BOARD_LEN];
            p.copy_from_slice(&v);
            p
        })
        .collect();
    let feats: Vec<[f32; FEATURE_LEN]> = cases
        .iter()
        .map(|c| {
            let v: Vec<f32> = c["features"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap() as f32)
                .collect();
            let mut f = [0.0; FEATURE_LEN];
            f.copy_from_slice(&v);
            f
        })
        .collect();

    let items: Vec<(&[f32; BOARD_LEN], &[f32; FEATURE_LEN])> =
        planes.iter().zip(feats.iter()).collect();
    let batched = net.forward(&items, &mut s);
    for (i, item) in items.iter().enumerate() {
        let single = net.forward(&[*item], &mut s);
        for (a, b) in [
            (batched[i].wdl[0], single[0].wdl[0]),
            (batched[i].wdl[1], single[0].wdl[1]),
            (batched[i].wdl[2], single[0].wdl[2]),
        ] {
            // BLAS blocks GEMMs differently at different m, so batch size
            // legitimately moves the last ulps (~3e-7 relative, measured).
            // Search determinism needs same-SHAPE reproducibility (a sibling
            // group is a deterministic batch), not bit-invariance across
            // shapes — so this asserts agreement, not identity.
            assert!(
                (a - b).abs() < 1e-3 + 1e-5 * b.abs(),
                "case {i}: batch {a} vs single {b}"
            );
        }
    }
}

/// Ad-hoc parity probe: point TETR_GOLDEN_DIR at any model dir carrying a
/// golden_v2.json (e.g. a trained net + REAL-observation goldens dumped by
/// tetrnn.goldens) and run with --ignored.
#[test]
#[ignore]
fn env_dir_fixture() {
    let dir = std::env::var("TETR_GOLDEN_DIR").expect("set TETR_GOLDEN_DIR");
    check_fixture(&dir);
}
