//! Cross-language parity: the Rust forward against PyTorch-dumped golden
//! vectors, on two fixtures:
//!
//! - `round0` — the reference campaign's *trained* net. Matching it proves the
//!   Rust forward reproduces that exporting model, not just itself.
//! - `pyref` — a small net emitted by *our own* `python/` package
//!   (`uv run python -m tetrnn.regen_pyref`). Matching it proves the two
//!   forwards agree on a model we can regenerate from source, closing the loop
//!   the inherited black-box fixture leaves open.

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
            own_board: [0.0; BOARD_LEN],
            opp_board: [0.0; BOARD_LEN],
            features: [0.0; FEATURE_LEN],
        };
        obs.own_board.copy_from_slice(&arr("own"));
        obs.opp_board.copy_from_slice(&arr("opp"));
        obs.features.copy_from_slice(&arr("features"));
        let expected = arr("out");

        let heads = net.forward_obs(&obs, &mut s);
        let got = [
            heads.wdl[0],
            heads.wdl[1],
            heads.wdl[2],
            heads.policy,
            heads.aux,
        ];
        for (j, (g, e)) in got.iter().zip(&expected).enumerate() {
            assert!(
                (g - e).abs() < 1e-4,
                "{dir} case {i} out[{j}]: rust {g} vs pytorch {e}"
            );
        }

        // Slot-head parity, when the fixture carries it. Coverage gap history:
        // forward_slots shipped WITHOUT a golden (2026-07-09) while the slot
        // vehicle collapsed in play — this is the check that would have
        // localized (or exonerated) a Rust-side slot defect immediately.
        if !case["slots"].is_null() {
            let expected_slots = arr("slots");
            let opp_emb = net
                .embed_boards(&[&obs.opp_board], &mut s)
                .pop()
                .expect("one embedding");
            let slots = net
                .forward_slots(&[(&obs.own_board, &obs.features)], &opp_emb, &mut s)
                .pop()
                .expect("one slot row");
            for (j, (g, e)) in slots.iter().zip(&expected_slots).enumerate() {
                assert!(
                    (g - e).abs() < 1e-3,
                    "{dir} case {i} slots[{j}]: rust {g} vs pytorch {e}"
                );
            }
        }
    }
}

#[test]
fn forward_matches_trained_reference_net() {
    check_fixture(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/round0"
    ));
}

#[test]
fn forward_matches_our_python_package() {
    check_fixture(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/pyref"));
}

#[test]
fn batched_forward_matches_one_by_one() {
    // The forward is batched by design; batching must be invisible in values.
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/round0");
    let net = Net::load(dir).expect("fixture model loads");
    let golden: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(format!("{dir}/golden_v2.json")).unwrap())
            .unwrap();
    let cases = golden["cases"].as_array().unwrap();

    // All 16 cases share one opp plane batch-of-16 vs 16 batches-of-1, under
    // the FIRST case's opponent embedding (broadcast semantics).
    let mut s = Scratch::default();
    let planes: Vec<[f32; BOARD_LEN]> = cases
        .iter()
        .map(|c| {
            let v: Vec<f32> = c["own"]
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
    let opp_plane = {
        let v: Vec<f32> = cases[0]["opp"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap() as f32)
            .collect();
        let mut p = [0.0; BOARD_LEN];
        p.copy_from_slice(&v);
        p
    };

    let opp = net
        .embed_boards(&[&opp_plane], &mut s)
        .pop()
        .expect("one embedding");
    let items: Vec<(&[f32; BOARD_LEN], &[f32; FEATURE_LEN])> =
        planes.iter().zip(feats.iter()).collect();
    let batched = net.forward(&items, &opp, &mut s);
    for (i, item) in items.iter().enumerate() {
        let single = net.forward(&[*item], &opp, &mut s);
        for (a, b) in [
            (batched[i].wdl[0], single[0].wdl[0]),
            (batched[i].wdl[1], single[0].wdl[1]),
            (batched[i].wdl[2], single[0].wdl[2]),
            (batched[i].policy, single[0].policy),
            (batched[i].aux, single[0].aux),
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
