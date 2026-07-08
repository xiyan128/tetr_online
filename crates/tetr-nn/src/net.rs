//! The two-board P+V net: weight loading and the batched forward.
//!
//! Architecture (fixed; `config.json` pins the conv channels):
//!
//! ```text
//! plane [1,40,10] ─ conv3x3+relu ×3 ─ flatten [C·400] ─ board_fc+relu ─ 128
//! (the SAME tower embeds own and opponent planes — siamese, one weight set)
//! features [85] ─ whiten ─ feat_fc+relu ─ 64
//! concat [own 128 | opp 128 | feat 64] ─ head1+relu ─ head2 ─ [5]
//! heads: wdl logits [0..3), policy logit [3], aux tanh [4]
//! ```
//!
//! There is **one** forward and it is batched — a batch of one is the scalar
//! case. Convolution is im2col + one GEMM per layer; dense layers are one GEMM
//! each. On macOS the GEMM is Accelerate (`cblas_sgemm`); elsewhere the
//! `openblas` feature provides it, and without it a plain-loop fallback keeps
//! the crate correct on any target.
//!
//! Weights load from a PyTorch export (`net_v2.safetensors` + `config.json`)
//! in PyTorch's own layouts — conv `[cout, cin, 3, 3]` and dense `[out, in]`
//! flatten row-major into exactly the shapes the GEMMs consume, so loading
//! performs **no permutation** (nothing to get wrong). Parity with the
//! exporting PyTorch model is pinned by the committed golden fixtures
//! (`tests/golden.rs`).

use std::path::Path;

use crate::obs::{BOARD_LEN, BOARD_W, FEATURE_LEN, N_SLOTS, Obs};

/// Spatial size of a plane (`40 × 10`), the per-position row count of the
/// conv GEMMs.
const HW: usize = BOARD_LEN;
/// Board embedding width (`board_fc` output).
const BOARD_EMB: usize = 128;
/// Feature embedding width (`feat_fc` output).
const FEAT_EMB: usize = 64;
/// Trunk width (`head1` output).
const TRUNK: usize = 128;
/// Head outputs: 3 WDL logits, 1 policy logit, 1 aux.
const N_OUT: usize = 5;
/// Whitening floor: a constant training feature has std ≈ 0, and dividing by it
/// yields inf/NaN. Both languages clamp the divisor to this.
const MIN_STD: f32 = 1e-6;

/// A loading failure with the offending file/tensor named.
#[derive(Debug)]
pub struct LoadError(String);

impl std::fmt::Display for LoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "tetr-nn load: {}", self.0)
    }
}

impl std::error::Error for LoadError {}

impl From<String> for LoadError {
    fn from(s: String) -> Self {
        Self(s)
    }
}

/// The leaf contract exported with the model: how a value head becomes a
/// search score (`score = z_scale · z_hat + attack_w · attack`).
#[derive(Clone, Copy, Debug)]
pub struct Contract {
    /// Scale from `z_hat ∈ (−1, 1)` to the integer score domain.
    pub z_scale: f32,
    /// Weight of one attack line in the same domain.
    pub attack_w: f32,
}

/// All head outputs for one leaf.
#[derive(Clone, Copy, Debug)]
pub struct Heads {
    /// Win/draw/loss logits.
    pub wdl: [f32; 3],
    /// Policy logit (softmaxed across a sibling group by the consumer).
    pub policy: f32,
    /// The detached aux head, already `tanh`'d (matching the PyTorch model).
    pub aux: f32,
}

impl Heads {
    /// The deployed value: `p_win − p_loss ∈ (−1, 1)` from the WDL softmax.
    pub fn z_hat(&self) -> f32 {
        let m = self.wdl.iter().cloned().fold(f32::MIN, f32::max);
        let e: [f32; 3] = std::array::from_fn(|i| (self.wdl[i] - m).exp());
        (e[0] - e[2]) / (e[0] + e[1] + e[2])
    }
}

/// A board embedding (the tower + `board_fc` output). Opaque so a serving
/// layer can cache the opponent's per decision and hand it back to
/// [`Net::forward`].
#[derive(Clone, Debug, PartialEq)]
pub struct BoardEmb(Vec<f32>);

/// One conv layer: PyTorch `[cout, cin·9]` weights (row-major flatten of
/// `[cout, cin, 3, 3]`) + bias.
struct Conv {
    w: Vec<f32>,
    bias: Vec<f32>,
    cin: usize,
    cout: usize,
}

/// One dense layer: PyTorch `[out, in]` weights + bias. The output width is a
/// module constant at each use site, so it is not stored here.
struct Dense {
    w: Vec<f32>,
    bias: Vec<f32>,
}

/// Reusable forward buffers — hold one per worker and the forward allocates
/// nothing on the hot path.
#[derive(Default)]
pub struct Scratch {
    /// Ping/pong activations, `[n·HW, C]` rows-of-positions layout.
    act_a: Vec<f32>,
    act_b: Vec<f32>,
    /// im2col gather target, `[n·HW, cin·9]`.
    cols: Vec<f32>,
    /// Per-item channel-major flatten fed to `board_fc`, `[n, C·HW]`.
    flat: Vec<f32>,
    /// Whitened features, `[n, FEATURE_LEN]`.
    feats: Vec<f32>,
    /// Embedding / trunk staging.
    emb: Vec<f32>,
    concat: Vec<f32>,
    trunk: Vec<f32>,
    out: Vec<f32>,
}

/// The loaded net + its exported contract.
pub struct Net {
    convs: Vec<Conv>,
    board_fc: Dense,
    feat_fc: Dense,
    head1: Dense,
    head2: Dense,
    /// The action-indexed policy head (one logit per [`N_SLOTS`] action slot,
    /// forwarded on the PARENT observation). Absent on pre-slot exports.
    slot_head: Option<Dense>,
    feat_mean: Vec<f32>,
    feat_std: Vec<f32>,
    /// The model's frozen leaf contract from `config.json`.
    pub contract: Contract,
}

impl Net {
    /// Load a model directory (`net_v2.safetensors` + `config.json`).
    pub fn load(dir: impl AsRef<Path>) -> Result<Self, LoadError> {
        let dir = dir.as_ref();
        let cfg: serde_json::Value = serde_json::from_slice(
            &std::fs::read(dir.join("config.json"))
                .map_err(|e| format!("{}: {e}", dir.join("config.json").display()))?,
        )
        .map_err(|e| format!("config.json: {e}"))?;
        if cfg["schema_version"].as_u64() != Some(2) {
            return Err(LoadError("config schema_version != 2".into()));
        }
        let stats = |k: &str| -> Result<Vec<f32>, LoadError> {
            let v: Vec<f32> = cfg[k]
                .as_array()
                .map(|a| a.iter().map(|x| x.as_f64().unwrap_or(0.0) as f32).collect())
                .unwrap_or_default();
            if v.len() != FEATURE_LEN {
                return Err(LoadError(format!("{k}: len {} != {FEATURE_LEN}", v.len())));
            }
            Ok(v)
        };
        let channels: Vec<usize> = cfg["arch"]["conv_channels"]
            .as_array()
            .map(|a| a.iter().map(|v| v.as_u64().unwrap_or(0) as usize).collect())
            .unwrap_or_default();
        if channels.len() < 2 || channels[0] != 1 {
            return Err(LoadError(format!("arch.conv_channels {channels:?}")));
        }

        let buf = std::fs::read(dir.join("net_v2.safetensors"))
            .map_err(|e| format!("{}: {e}", dir.join("net_v2.safetensors").display()))?;
        let st = safetensors::SafeTensors::deserialize(&buf).map_err(|e| e.to_string())?;
        let tensor = |name: &str| -> Result<Vec<f32>, LoadError> {
            let t = st.tensor(name).map_err(|e| format!("{name}: {e}"))?;
            Ok(t.data()
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect())
        };
        let dense = |name: &str, n_in: usize, n_out: usize| -> Result<Dense, LoadError> {
            let w = tensor(&format!("{name}.weight"))?;
            if w.len() != n_in * n_out {
                return Err(LoadError(format!(
                    "{name}.weight: len {} != {n_out}×{n_in}",
                    w.len()
                )));
            }
            Ok(Dense {
                w,
                bias: tensor(&format!("{name}.bias"))?,
            })
        };

        let mut convs = Vec::with_capacity(channels.len() - 1);
        for (i, pair) in channels.windows(2).enumerate() {
            let (cin, cout) = (pair[0], pair[1]);
            let w = tensor(&format!("conv{}.weight", i + 1))?;
            if w.len() != cout * cin * 9 {
                return Err(LoadError(format!("conv{}.weight: len {}", i + 1, w.len())));
            }
            convs.push(Conv {
                w,
                bias: tensor(&format!("conv{}.bias", i + 1))?,
                cin,
                cout,
            });
        }
        let last_c = *channels.last().expect("checked non-empty");

        let slot_head = if st.tensor("slot_head.weight").is_ok() {
            Some(dense("slot_head", TRUNK, N_SLOTS)?)
        } else {
            None
        };
        Ok(Self {
            board_fc: dense("board_fc", last_c * HW, BOARD_EMB)?,
            feat_fc: dense("feat_fc", FEATURE_LEN, FEAT_EMB)?,
            head1: dense("head1", 2 * BOARD_EMB + FEAT_EMB, TRUNK)?,
            head2: dense("head2", TRUNK, N_OUT)?,
            slot_head,
            convs,
            feat_mean: stats("feature_mean")?,
            feat_std: stats("feature_std")?,
            contract: Contract {
                z_scale: cfg["contract"]["z_scale"].as_f64().unwrap_or(10_000.0) as f32,
                attack_w: cfg["contract"]["attack_w"].as_f64().unwrap_or(100.0) as f32,
            },
        })
    }

    /// Embed a batch of occupancy planes through the shared tower + `board_fc`.
    /// Used for own planes (per leaf) and the opponent plane (once per decision,
    /// cached by the serving layer).
    pub fn embed_boards(&self, planes: &[&[f32; BOARD_LEN]], s: &mut Scratch) -> Vec<BoardEmb> {
        let n = planes.len();
        if n == 0 {
            return Vec::new();
        }
        // Layout: activations as [n·HW, C] (rows are positions, columns are
        // channels) — im2col gathers straight out of it for every layer,
        // including the first (C = 1).
        s.act_a.clear();
        s.act_a
            .extend(planes.iter().flat_map(|p| p.iter().copied()));

        let mut in_a = true;
        for conv in &self.convs {
            {
                let input = if in_a { &s.act_a } else { &s.act_b };
                im2col(input, n, conv.cin, &mut s.cols);
            }
            let output = if in_a { &mut s.act_b } else { &mut s.act_a };
            dense_batch(
                &s.cols,
                n * HW,
                conv.cin * 9,
                &conv.w,
                &conv.bias,
                conv.cout,
                true,
                output,
            );
            in_a = !in_a;
        }
        let last = if in_a { &s.act_a } else { &s.act_b };
        let c = self.convs.last().expect("at least one conv").cout;

        // Per item, transpose [HW, C] → channel-major [C·HW] for board_fc
        // (PyTorch's `flatten(1)` of [N, C, H, W]).
        s.flat.clear();
        s.flat.resize(n * c * HW, 0.0);
        for b in 0..n {
            for hw in 0..HW {
                for ci in 0..c {
                    s.flat[b * c * HW + ci * HW + hw] = last[(b * HW + hw) * c + ci];
                }
            }
        }
        dense_batch(
            &s.flat,
            n,
            c * HW,
            &self.board_fc.w,
            &self.board_fc.bias,
            BOARD_EMB,
            true,
            &mut s.emb,
        );
        (0..n)
            .map(|b| BoardEmb(s.emb[b * BOARD_EMB..(b + 1) * BOARD_EMB].to_vec()))
            .collect()
    }

    /// The batched leaf forward: each item's own plane and features, under ONE
    /// frozen opponent embedding (broadcast across the batch — the siamese
    /// saving that makes a two-board value cost about one board per leaf).
    pub fn forward(
        &self,
        items: &[(&[f32; BOARD_LEN], &[f32; FEATURE_LEN])],
        opp: &BoardEmb,
        s: &mut Scratch,
    ) -> Vec<Heads> {
        let n = self.trunk_batch(items, opp, s);
        if n == 0 {
            return Vec::new();
        }
        dense_batch(
            &s.trunk,
            n,
            TRUNK,
            &self.head2.w,
            &self.head2.bias,
            N_OUT,
            false,
            &mut s.out,
        );

        (0..n)
            .map(|b| {
                let o = &s.out[b * N_OUT..(b + 1) * N_OUT];
                Heads {
                    wdl: [o[0], o[1], o[2]],
                    policy: o[3],
                    aux: o[4].tanh(),
                }
            })
            .collect()
    }

    /// The action head on PARENT observations: one forward per state, a raw
    /// logit per action slot. Panics if the model was exported without the
    /// slot head (pre-slot models cannot drive a guided search).
    pub fn forward_slots(
        &self,
        items: &[(&[f32; BOARD_LEN], &[f32; FEATURE_LEN])],
        opp: &BoardEmb,
        s: &mut Scratch,
    ) -> Vec<[f32; N_SLOTS]> {
        let head = self
            .slot_head
            .as_ref()
            .expect("model has no slot head (pre-slot export)");
        let n = self.trunk_batch(items, opp, s);
        if n == 0 {
            return Vec::new();
        }
        dense_batch(
            &s.trunk, n, TRUNK, &head.w, &head.bias, N_SLOTS, false, &mut s.out,
        );
        (0..n)
            .map(|b| {
                let mut row = [0.0f32; N_SLOTS];
                row.copy_from_slice(&s.out[b * N_SLOTS..(b + 1) * N_SLOTS]);
                row
            })
            .collect()
    }

    /// Whether this export carries the action head.
    pub fn has_slot_head(&self) -> bool {
        self.slot_head.is_some()
    }

    /// Shared trunk: own tower | opp (broadcast) | whitened feats -> head1,
    /// leaving activations in `s.trunk`. Returns the batch size.
    fn trunk_batch(
        &self,
        items: &[(&[f32; BOARD_LEN], &[f32; FEATURE_LEN])],
        opp: &BoardEmb,
        s: &mut Scratch,
    ) -> usize {
        let n = items.len();
        if n == 0 {
            return 0;
        }
        let planes: Vec<&[f32; BOARD_LEN]> = items.iter().map(|(p, _)| *p).collect();
        let own = self.embed_boards(&planes, s);

        // Whiten features into a [n, FEATURE_LEN] batch.
        s.feats.clear();
        s.feats.reserve(n * FEATURE_LEN);
        for (_, f) in items {
            for i in 0..FEATURE_LEN {
                // Floor the std so a zero-variance (constant) feature can't
                // divide to inf/NaN and silently corrupt the score. The Python
                // forward (model.py) floors identically, so parity holds.
                s.feats
                    .push((f[i] - self.feat_mean[i]) / self.feat_std[i].max(MIN_STD));
            }
        }
        let mut femb = Vec::new();
        dense_batch(
            &s.feats,
            n,
            FEATURE_LEN,
            &self.feat_fc.w,
            &self.feat_fc.bias,
            FEAT_EMB,
            true,
            &mut femb,
        );

        // concat [own | opp (tiled) | feat] → head1 → head2.
        let h_in = 2 * BOARD_EMB + FEAT_EMB;
        s.concat.clear();
        s.concat.reserve(n * h_in);
        for b in 0..n {
            s.concat.extend_from_slice(&own[b].0);
            s.concat.extend_from_slice(&opp.0);
            s.concat
                .extend_from_slice(&femb[b * FEAT_EMB..(b + 1) * FEAT_EMB]);
        }
        dense_batch(
            &s.concat,
            n,
            h_in,
            &self.head1.w,
            &self.head1.bias,
            TRUNK,
            true,
            &mut s.trunk,
        );
        n
    }

    /// Convenience full forward of one observation (embeds the opp plane too).
    /// Golden tests and one-off scoring; serving caches [`embed_boards`] of the
    /// opponent per decision instead.
    ///
    /// [`embed_boards`]: Self::embed_boards
    pub fn forward_obs(&self, obs: &Obs, s: &mut Scratch) -> Heads {
        let opp = self
            .embed_boards(&[&obs.opp_board], s)
            .pop()
            .expect("one plane in, one embedding out");
        self.forward(&[(&obs.own_board, &obs.features)], &opp, s)
            .pop()
            .expect("one item in, one head out")
    }
}

/// im2col for a 3×3, pad-1 conv over `[n·HW, cin]` activations:
/// `cols[(b·HW + y·W + x) · cin·9 + ci·9 + tap]` = the input at the tap's
/// shifted position (zero at borders), with `tap = ky·3 + kx`. This matches
/// the row-major flatten of PyTorch's `[cout, cin, 3, 3]` weights, so the
/// conv is ONE GEMM with untouched weights.
fn im2col(act: &[f32], n: usize, cin: usize, cols: &mut Vec<f32>) {
    let h = HW / BOARD_W;
    let w = BOARD_W;
    cols.clear();
    cols.resize(n * HW * cin * 9, 0.0);
    for b in 0..n {
        for y in 0..h {
            for x in 0..w {
                let row = (b * HW + y * w + x) * cin * 9;
                for (tap, (dy, dx)) in TAPS.iter().enumerate() {
                    let (sy, sx) = (y as isize + dy, x as isize + dx);
                    if sy < 0 || sy >= h as isize || sx < 0 || sx >= w as isize {
                        continue; // zero padding
                    }
                    let src = (b * HW + sy as usize * w + sx as usize) * cin;
                    for ci in 0..cin {
                        cols[row + ci * 9 + tap] = act[src + ci];
                    }
                }
            }
        }
    }
}

/// 3×3 tap offsets in PyTorch kernel order (`ky` major).
const TAPS: [(isize, isize); 9] = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 0),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
];

/// `out[m, n_out] = a[m, k] × w[n_out, k]ᵀ + bias`, optionally ReLU'd — every
/// layer in the net is this one shape. BLAS when available, plain loops
/// otherwise; both paths share the bias/ReLU epilogue.
#[allow(clippy::too_many_arguments)]
fn dense_batch(
    a: &[f32],
    m: usize,
    k: usize,
    w: &[f32],
    bias: &[f32],
    n_out: usize,
    relu: bool,
    out: &mut Vec<f32>,
) {
    debug_assert_eq!(a.len(), m * k);
    debug_assert_eq!(w.len(), n_out * k);
    out.clear();
    out.resize(m * n_out, 0.0);

    #[cfg(any(target_os = "macos", feature = "openblas"))]
    unsafe {
        cblas::sgemm(
            cblas::Layout::RowMajor,
            cblas::Transpose::None,
            cblas::Transpose::Ordinary,
            m as i32,
            n_out as i32,
            k as i32,
            1.0,
            a,
            k as i32,
            w,
            k as i32,
            0.0,
            out,
            n_out as i32,
        );
    }
    #[cfg(not(any(target_os = "macos", feature = "openblas")))]
    for row in 0..m {
        for col in 0..n_out {
            let mut acc = 0.0f32;
            for i in 0..k {
                acc += a[row * k + i] * w[col * k + i];
            }
            out[row * n_out + col] = acc;
        }
    }

    for row in 0..m {
        for col in 0..n_out {
            let v = out[row * n_out + col] + bias[col];
            out[row * n_out + col] = if relu { v.max(0.0) } else { v };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dense_batch_matches_hand_math() {
        // 2×3 input, 2 outputs: y = x·Wᵀ + b.
        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let w = [1.0, 0.0, 0.0, 0.0, 1.0, -1.0]; // rows: [1,0,0], [0,1,-1]
        let bias = [0.5, -0.5];
        let mut out = Vec::new();
        dense_batch(&a, 2, 3, &w, &bias, 2, false, &mut out);
        assert_eq!(out, vec![1.5, -1.5, 4.5, -1.5]);
        dense_batch(&a, 2, 3, &w, &bias, 2, true, &mut out);
        assert_eq!(out, vec![1.5, 0.0, 4.5, 0.0]);
    }

    #[test]
    fn im2col_gathers_the_neighborhood() {
        // One 40×10 plane, cin=1, with a single hot cell; its value must appear
        // at exactly the 9 neighboring positions' mirrored taps.
        let mut plane = vec![0.0f32; HW];
        let (y, x) = (5, 5);
        plane[y * BOARD_W + x] = 1.0;
        let mut cols = Vec::new();
        im2col(&plane, 1, 1, &mut cols);
        let mut hits = Vec::new();
        for pos in 0..HW {
            for tap in 0..9 {
                if cols[pos * 9 + tap] != 0.0 {
                    hits.push((pos / BOARD_W, pos % BOARD_W, tap));
                }
            }
        }
        // 9 positions see the hot cell, each through the tap pointing back at it.
        assert_eq!(hits.len(), 9);
        for (py, px, tap) in hits {
            let (dy, dx) = TAPS[tap];
            assert_eq!(
                (py as isize + dy, px as isize + dx),
                (y as isize, x as isize)
            );
        }
    }

    #[test]
    fn z_hat_is_the_wdl_probability_gap() {
        let h = Heads {
            wdl: [0.0, 0.0, 0.0],
            policy: 0.0,
            aux: 0.0,
        };
        assert!(h.z_hat().abs() < 1e-6);
        let sure_win = Heads {
            wdl: [10.0, 0.0, -10.0],
            policy: 0.0,
            aux: 0.0,
        };
        assert!(sure_win.z_hat() > 0.99);
    }
}
