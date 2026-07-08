//! The pure-Rust value-net forward — the wasm-ready ship kernel (no BLAS, no
//! candle). A shrunk conv tower over the board plane, concatenated with an
//! embedding of the feature vector, through two dense heads to one scalar value
//! (de-standardised to the training target's scale).
//!
//! ```text
//! board [1,40,10] ─ conv3x3+relu ×N ─ flatten [C·400] ─ board_fc+relu ─ 128
//! features [70] ─ whiten ─ feat_fc+relu ─ 64
//! concat [board 128 | feat 64] ─ head1+relu ─ head2[0] ─ ×std + mean ─ value
//! ```
//!
//! Weights load from a PyTorch export (`value_net.safetensors` + `config.json`,
//! `schema_version` 1) in PyTorch's own `[cout,cin,3,3]` / `[out,in]` layouts.
//! The forward reproduces the trained model to a small float tolerance — pinned
//! by the committed `conv_rb1` golden vectors (`tests/golden.rs`).

use std::path::Path;

use crate::encode::{BOARD_H, BOARD_LEN, BOARD_W, FEATURE_LEN, Obs};

/// The board plane's flattened length (= [`BOARD_LEN`]), aliased `HW` where the
/// conv/dense math reads more naturally as a count of `H × W` spatial positions.
const HW: usize = BOARD_LEN;
/// Board embedding width (`board_fc` output).
const BOARD_EMB: usize = 128;
/// Feature embedding width (`feat_fc` output).
const FEAT_EMB: usize = 64;
/// Trunk width (`head1` output).
const TRUNK: usize = 128;

/// A 3×3 pad-1 conv, weight pre-reshaped to `[tap=9][cin][cout]` (cout innermost
/// so the accumulate loop vectorises over output channels).
#[derive(Clone)]
struct Conv {
    w: Vec<f32>,
    bias: Vec<f32>,
    cin: usize,
    cout: usize,
}

impl Conv {
    /// `[HW][cin]` channels-last → `[HW][cout]` channels-last, +bias +relu, into
    /// a reused `out` buffer.
    fn forward_into(&self, inp: &[f32], out: &mut [f32]) {
        let (cin, cout) = (self.cin, self.cout);
        for y in 0..BOARD_H {
            for x in 0..BOARD_W {
                let o = (y * BOARD_W + x) * cout;
                let oslice = &mut out[o..o + cout];
                oslice.fill(0.0);
                for ky in 0..3 {
                    let yy = y as isize + ky as isize - 1;
                    if yy < 0 || yy >= BOARD_H as isize {
                        continue;
                    }
                    for kx in 0..3 {
                        let xx = x as isize + kx as isize - 1;
                        if xx < 0 || xx >= BOARD_W as isize {
                            continue;
                        }
                        let ibase = (yy as usize * BOARD_W + xx as usize) * cin;
                        let tbase = (ky * 3 + kx) * cin * cout;
                        for ci in 0..cin {
                            let iv = inp[ibase + ci];
                            let wrow = &self.w[tbase + ci * cout..tbase + ci * cout + cout];
                            for (ov, &wv) in oslice.iter_mut().zip(wrow) {
                                *ov += iv * wv;
                            }
                        }
                    }
                }
                for (co, ov) in oslice.iter_mut().enumerate() {
                    *ov = (*ov + self.bias[co]).max(0.0);
                }
            }
        }
    }
}

/// A dense layer, weight `[n_out][n_in]` row-major (PyTorch `Linear`). The
/// output width is the caller's `out` slice length, so it isn't stored here.
#[derive(Clone)]
struct Dense {
    w: Vec<f32>,
    bias: Vec<f32>,
    n_in: usize,
}

impl Dense {
    /// `y = relu?(Wx + b)` into a reused `out` buffer. The dot product keeps 8
    /// parallel accumulators so LLVM emits wide FMAs (f32 add isn't associative,
    /// so a scalar reduction stays scalar); the lane partial-sums shift float
    /// rounding order slightly, which the golden tolerance absorbs.
    fn forward_into(&self, x: &[f32], out: &mut [f32], relu: bool) {
        debug_assert_eq!(x.len(), self.n_in);
        let n = self.n_in;
        let main = n - n % 8;
        for (o, ov) in out.iter_mut().enumerate() {
            let wrow = &self.w[o * n..o * n + n];
            let mut lanes = [0f32; 8];
            let mut i = 0;
            while i < main {
                for j in 0..8 {
                    lanes[j] += x[i + j] * wrow[i + j];
                }
                i += 8;
            }
            let mut acc = self.bias[o] + lanes.iter().sum::<f32>();
            while i < n {
                acc += x[i] * wrow[i];
                i += 1;
            }
            *ov = if relu { acc.max(0.0) } else { acc };
        }
    }
}

/// Reusable per-thread forward buffers (one per worker → allocation-free hot path).
pub struct Scratch {
    a: Vec<f32>,
    b: Vec<f32>,
    flat: Vec<f32>,
    board_emb: Vec<f32>,
    feats: Vec<f32>,
    feat_emb: Vec<f32>,
    trunk_in: Vec<f32>,
    trunk: Vec<f32>,
}

/// A loading failure with the offending file/tensor named.
#[derive(Debug)]
pub struct LoadError(String);

impl std::fmt::Display for LoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "tetr-valuenet load: {}", self.0)
    }
}

impl std::error::Error for LoadError {}

/// The loaded single-board value net.
#[derive(Clone)]
pub struct ValueNet {
    convs: Vec<Conv>,
    board_fc: Dense,
    feat_fc: Dense,
    head1: Dense,
    head2: Dense,
    feat_mean: Vec<f32>,
    feat_std: Vec<f32>,
    /// De-standardisation for the single value head (`value = raw·std + mean`).
    value_mean: f32,
    value_std: f32,
}

impl ValueNet {
    /// Load a model dir (`value_net.safetensors` + `config.json`, schema 1).
    pub fn load(dir: impl AsRef<Path>) -> Result<Self, LoadError> {
        let dir = dir.as_ref();
        let err = |m: String| LoadError(m);
        let cfg: serde_json::Value = serde_json::from_slice(
            &std::fs::read(dir.join("config.json")).map_err(|e| err(e.to_string()))?,
        )
        .map_err(|e| err(format!("config.json: {e}")))?;

        let arr = |v: &serde_json::Value| -> Vec<f32> {
            v.as_array()
                .map(|a| a.iter().map(|x| x.as_f64().unwrap_or(0.0) as f32).collect())
                .unwrap_or_default()
        };
        // value_mean/std are scalar for a single head (or a 1-element list).
        let scalar = |v: &serde_json::Value| -> Option<f32> {
            match v {
                serde_json::Value::Number(_) => v.as_f64().map(|x| x as f32),
                serde_json::Value::Array(a) => a.first().and_then(|x| x.as_f64()).map(|x| x as f32),
                _ => None,
            }
        };
        let feat_mean = arr(&cfg["feature_mean"]);
        let feat_std = arr(&cfg["feature_std"]);
        let value_mean =
            scalar(&cfg["value_mean"]).ok_or_else(|| err("value_mean missing".into()))?;
        let value_std = scalar(&cfg["value_std"]).ok_or_else(|| err("value_std missing".into()))?;
        let channels: Vec<usize> = cfg["arch"]["conv_channels"]
            .as_array()
            .map(|a| a.iter().map(|v| v.as_u64().unwrap_or(0) as usize).collect())
            .unwrap_or_default();
        // Validate BOTH whitening stats: the forward divides by feat_std[i] for
        // every feature, so a short or zero-containing feature_std would load
        // fine and then panic (out-of-bounds) or produce NaN on the first
        // forward — a per-move crash inside a live game, defeating the
        // registry's "omit the bot if the dir doesn't load" contract. Reject at
        // load instead. (The shipped conv_rb1 config passes this unchanged.)
        if feat_mean.len() != FEATURE_LEN
            || feat_std.len() != FEATURE_LEN
            || feat_std.iter().any(|&s| s.abs() < f32::EPSILON)
            || channels.first() != Some(&1)
            || channels.len() < 2
        {
            return Err(err(format!(
                "config mismatch (feature_mean {}, feature_std {}, channels {channels:?}, \
                 expected {FEATURE_LEN}/{FEATURE_LEN} + nonzero std + [1,..])",
                feat_mean.len(),
                feat_std.len()
            )));
        }

        let buf =
            std::fs::read(dir.join("value_net.safetensors")).map_err(|e| err(e.to_string()))?;
        let st = safetensors::SafeTensors::deserialize(&buf).map_err(|e| err(e.to_string()))?;
        let tensor = |name: &str| -> Result<Vec<f32>, LoadError> {
            let t = st.tensor(name).map_err(|e| err(format!("{name}: {e}")))?;
            Ok(t.data()
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect())
        };

        // Conv stack: PyTorch [cout,cin,3,3] → [tap][cin][cout] (no permutation
        // of values, just a reindex to the channels-last forward layout).
        let mut convs = Vec::with_capacity(channels.len() - 1);
        for (i, pair) in channels.windows(2).enumerate() {
            let (cin, cout) = (pair[0], pair[1]);
            let orig = tensor(&format!("conv{}.weight", i + 1))?;
            let mut w = vec![0f32; 9 * cin * cout];
            for co in 0..cout {
                for ci in 0..cin {
                    for ky in 0..3 {
                        for kx in 0..3 {
                            let src = ((co * cin + ci) * 3 + ky) * 3 + kx;
                            let dst = (ky * 3 + kx) * cin * cout + ci * cout + co;
                            w[dst] = orig[src];
                        }
                    }
                }
            }
            convs.push(Conv {
                w,
                bias: tensor(&format!("conv{}.bias", i + 1))?,
                cin,
                cout,
            });
        }
        let dense = |name: &str, n_in: usize, n_out: usize| -> Result<Dense, LoadError> {
            let w = tensor(&format!("{name}.weight"))?;
            if w.len() != n_in * n_out {
                return Err(err(format!(
                    "{name}.weight: len {} != {n_out}×{n_in}",
                    w.len()
                )));
            }
            Ok(Dense {
                w,
                bias: tensor(&format!("{name}.bias"))?,
                n_in,
            })
        };
        let last_c = *channels.last().expect("checked len >= 2");
        Ok(Self {
            board_fc: dense("board_fc", last_c * HW, BOARD_EMB)?,
            feat_fc: dense("feat_fc", FEATURE_LEN, FEAT_EMB)?,
            head1: dense("head1", BOARD_EMB + FEAT_EMB, TRUNK)?,
            head2: dense("head2", TRUNK, 1)?,
            convs,
            feat_mean,
            feat_std,
            value_mean,
            value_std,
        })
    }

    /// Reusable buffers (one per worker) for [`forward_into`](Self::forward_into).
    pub fn scratch(&self) -> Scratch {
        let max_c = self.convs.iter().map(|c| c.cout).max().unwrap_or(1);
        let last_c = self.convs.last().map(|c| c.cout).unwrap_or(1);
        Scratch {
            a: vec![0.0; HW * max_c],
            b: vec![0.0; HW * max_c],
            flat: vec![0.0; HW * last_c],
            board_emb: vec![0.0; BOARD_EMB],
            feats: vec![0.0; FEATURE_LEN],
            feat_emb: vec![0.0; FEAT_EMB],
            trunk_in: vec![0.0; BOARD_EMB + FEAT_EMB],
            trunk: vec![0.0; TRUNK],
        }
    }

    /// The de-standardised value for one observation, into reused `scratch`.
    pub fn forward_into(&self, obs: &Obs, s: &mut Scratch) -> f32 {
        // Conv stack, channels-last: layer 0 reads the board into `a`, the rest
        // ping-pong a<->b.
        let c0 = self.convs[0].cout;
        self.convs[0].forward_into(&obs.board, &mut s.a[..HW * c0]);
        let mut in_a = true;
        let mut prev = c0;
        for conv in &self.convs[1..] {
            let co = conv.cout;
            if in_a {
                conv.forward_into(&s.a[..HW * prev], &mut s.b[..HW * co]);
            } else {
                conv.forward_into(&s.b[..HW * prev], &mut s.a[..HW * co]);
            }
            in_a = !in_a;
            prev = co;
        }
        // Transpose the final conv output [HW][prev] → [prev][HW] (C-major
        // flatten, matching the PyTorch `flatten(1)` board_fc consumes).
        let cur: &[f32] = if in_a { &s.a } else { &s.b };
        for hw in 0..HW {
            for c in 0..prev {
                s.flat[c * HW + hw] = cur[hw * prev + c];
            }
        }
        self.board_fc
            .forward_into(&s.flat[..prev * HW], &mut s.board_emb, true);
        for i in 0..FEATURE_LEN {
            s.feats[i] = (obs.features[i] - self.feat_mean[i]) / self.feat_std[i];
        }
        self.feat_fc.forward_into(&s.feats, &mut s.feat_emb, true);
        s.trunk_in[..BOARD_EMB].copy_from_slice(&s.board_emb);
        s.trunk_in[BOARD_EMB..].copy_from_slice(&s.feat_emb);
        self.head1.forward_into(&s.trunk_in, &mut s.trunk, true);
        // head2 has a single output, so its forward is inlined as a scalar dot
        // rather than routed through `Dense::forward_into` — it needs no output
        // scratch buffer, and the reduction order here is part of the frozen
        // numerics the golden pins.
        let mut raw = self.head2.bias[0];
        for (xv, &wv) in s.trunk.iter().zip(&self.head2.w[..TRUNK]) {
            raw += xv * wv;
        }
        raw * self.value_std + self.value_mean
    }

    /// The de-standardised value for one observation (allocates a scratch).
    pub fn forward(&self, obs: &Obs) -> f32 {
        self.forward_into(obs, &mut self.scratch())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A short/zero `feature_std` must fail at LOAD (a clean `Err`), not load
    /// OK and then panic on the first forward mid-game — the registry omits a
    /// bot whose dir doesn't load, so a deferred crash would break that.
    #[test]
    fn malformed_feature_std_is_rejected_at_load() {
        let dir = std::env::temp_dir().join(format!("tetr-valuenet-badstd-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        // config.json is read (and the guard fires) before value_net.safetensors,
        // so a config alone reproduces the rejection.
        let cfg = format!(
            r#"{{"arch":{{"conv_channels":[1,16,32,32]}},"value_mean":0.0,"value_std":1.0,
                "feature_mean":{mean},"feature_std":[1.0,2.0]}}"#,
            mean = serde_json::to_string(&vec![0.0f32; FEATURE_LEN]).unwrap()
        );
        std::fs::write(dir.join("config.json"), cfg).unwrap();
        match ValueNet::load(&dir) {
            Err(e) => assert!(e.to_string().contains("config mismatch"), "{e}"),
            Ok(_) => panic!("short feature_std must be rejected at load"),
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
}
