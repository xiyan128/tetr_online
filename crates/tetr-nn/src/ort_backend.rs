//! The CoreML (ANE/GPU) leaf-batch backend — feature `coreml`, macOS.
//!
//! Measured on the two-board net (2026-07-09, contended lower bounds): CoreML
//! MLProgram fixed-batch graphs run 30k evals/s @ batch 68 and 116.7k @ 480 vs
//! the BLAS forward's ~6.9k — fixed graphs beat the dynamic graph ~5× (CoreML
//! compiles a static plan), so this backend holds one session per BUCKET size
//! and pads sibling batches up to the nearest bucket. Parity: the exported
//! graph matches the torch forward to ~1e-6 relative (whitening baked in — raw
//! observations go straight in).
//!
//! The guided filter's per-node parent forward stays on the BLAS path (batch-1
//! CoreML pays per-call overhead for nothing); this backend serves the LEAF
//! batches, which dominate datagen cost.

use std::path::Path;
use std::sync::Mutex;

use ort::ep::CoreMLExecutionProvider;
use ort::session::Session;
use ort::value::Tensor;

use crate::net::Contract;
use crate::obs::{BOARD_LEN, FEATURE_LEN, Obs};

/// Bucket sizes — must match the graphs `export_onnx.py` writes
/// (`net_leaf_b{N}.onnx`). Sibling groups pad up to the nearest bucket;
/// larger batches split.
pub const BUCKETS: [usize; 4] = [34, 68, 128, 480];

/// One CoreML session per bucket + the model's leaf contract.
pub struct OrtBackend {
    inner: Mutex<Vec<(usize, Session)>>,
    pub contract: Contract,
}

impl OrtBackend {
    pub fn load(dir: impl AsRef<Path>) -> Result<Self, String> {
        let dir = dir.as_ref();
        let cfg: serde_json::Value = serde_json::from_slice(
            &std::fs::read(dir.join("config.json")).map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.to_string())?;
        let contract = Contract {
            z_scale: cfg["contract"]["z_scale"].as_f64().unwrap_or(10_000.0) as f32,
            attack_w: cfg["contract"]["attack_w"].as_f64().unwrap_or(100.0) as f32,
        };
        let mut sessions = Vec::new();
        for &n in &BUCKETS {
            let path = dir.join(format!("net_leaf_b{n}.onnx"));
            if !path.exists() {
                continue; // smaller export sets are fine; padding covers gaps
            }
            let session = Session::builder()
                .map_err(|e| e.to_string())?
                .with_execution_providers([CoreMLExecutionProvider::default()
                    .with_model_format(ort::ep::coreml::ModelFormat::MLProgram)
                    .build()])
                .map_err(|e| e.to_string())?
                .commit_from_file(&path)
                .map_err(|e| format!("load {}: {e}", path.display()))?;
            sessions.push((n, session));
        }
        if sessions.is_empty() {
            return Err(format!(
                "{}: no net_leaf_b*.onnx graphs (run tetrnn.export_onnx)",
                dir.display()
            ));
        }
        Ok(Self {
            inner: Mutex::new(sessions),
            contract,
        })
    }

    /// Forward a batch of observations, returning the raw 5-head rows.
    /// Batches pad up to the nearest bucket (padded rows discarded); batches
    /// larger than the biggest bucket split into chunks.
    pub fn forward_batch(&self, obss: &[&Obs]) -> Vec<[f32; 5]> {
        let mut out = Vec::with_capacity(obss.len());
        let mut sessions = self.inner.lock().expect("ort sessions lock");
        let max_bucket = sessions.last().map(|(n, _)| *n).expect("nonempty");
        for chunk in obss.chunks(max_bucket) {
            let n = chunk.len();
            let bucket_idx = sessions
                .iter()
                .position(|(b, _)| *b >= n)
                .unwrap_or(sessions.len() - 1);
            let (b, session) = &mut sessions[bucket_idx];
            let b = *b;
            let mut own = vec![0f32; b * BOARD_LEN];
            let mut opp = vec![0f32; b * BOARD_LEN];
            let mut feats = vec![0f32; b * FEATURE_LEN];
            for (i, o) in chunk.iter().enumerate() {
                own[i * BOARD_LEN..(i + 1) * BOARD_LEN].copy_from_slice(&o.own_board);
                opp[i * BOARD_LEN..(i + 1) * BOARD_LEN].copy_from_slice(&o.opp_board);
                feats[i * FEATURE_LEN..(i + 1) * FEATURE_LEN].copy_from_slice(&o.features);
            }
            let own_t = Tensor::from_array(([b, 1usize, 40, 10], own)).expect("own tensor");
            let opp_t = Tensor::from_array(([b, 1usize, 40, 10], opp)).expect("opp tensor");
            let feat_t = Tensor::from_array(([b, FEATURE_LEN], feats)).expect("feat tensor");
            let outputs = session
                .run(ort::inputs!["own" => own_t, "opp" => opp_t, "feats" => feat_t])
                .expect("ort run");
            let (_shape, v) = outputs["out"]
                .try_extract_tensor::<f32>()
                .expect("extract heads");
            for i in 0..n {
                let mut row = [0f32; 5];
                row.copy_from_slice(&v[i * 5..(i + 1) * 5]);
                out.push(row);
            }
        }
        out
    }
}

use tetr_core::ai::eval::{EvalContext, Evaluator, Leaf, Reward, Value, attack_reward};
use tetr_core::engine::{Board, LockOutcome, TSpinKind};

use crate::obs::{OppCtx, encode};

/// [`Evaluator`] over the CoreML backend — the drop-in accelerated twin of
/// [`crate::serve::NetEvaluator`]: same encode, same opponent-blind `OppCtx`,
/// same compose (`z_scale·ẑ + attack_w·attack`), ~5-11× the BLAS forward.
pub struct OrtNetEvaluator {
    backend: OrtBackend,
    opp: OppCtx,
}

impl OrtNetEvaluator {
    pub fn load(dir: impl AsRef<Path>) -> Result<Self, String> {
        Ok(Self {
            backend: OrtBackend::load(dir)?,
            opp: OppCtx::default(),
        })
    }

    fn z_hat(row: &[f32; 5]) -> f32 {
        let m = row[..3].iter().cloned().fold(f32::MIN, f32::max);
        let e: [f32; 3] = std::array::from_fn(|i| (row[i] - m).exp());
        (e[0] - e[2]) / (e[0] + e[1] + e[2])
    }
}

impl Evaluator for OrtNetEvaluator {
    fn evaluate(
        &self,
        _lock: &LockOutcome,
        _board: &Board,
        _t_spin: Option<TSpinKind>,
        _ctx: EvalContext,
    ) -> (Value, Reward) {
        panic!("OrtNetEvaluator scores via evaluate_leaves (needs the full SearchState)")
    }

    fn evaluate_leaves(&self, leaves: &[Leaf<'_>]) -> Vec<(Value, Reward)> {
        if leaves.is_empty() {
            return Vec::new();
        }
        let obs: Vec<_> = leaves.iter().map(|l| encode(l.state, &self.opp)).collect();
        let refs: Vec<&crate::obs::Obs> = obs.iter().collect();
        let rows = self.backend.forward_batch(&refs);
        let c = &self.backend.contract;
        rows.iter()
            .zip(leaves)
            .map(|(row, leaf)| {
                let atk = attack_reward(leaf);
                (
                    Value((c.z_scale * Self::z_hat(row)).round() as i32),
                    Reward((c.attack_w * atk.0 as f32).round() as i32),
                )
            })
            .collect()
    }
}
