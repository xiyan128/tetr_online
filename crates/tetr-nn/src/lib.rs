//! `tetr-nn` — a Burn-backed neural board evaluator for the Tetris AI.
//!
//! Replaces the *static board [`Value`]* of `tetr-core`'s `LinearEvaluator` with
//! a small learned value net (an MLP over the 8 Dellacherie / BCTS features),
//! while reusing tetr-core's principled per-move [`Reward`] (line clears / spins
//! / Back-to-Back) via [`tetr_core::ai::eval::compute_reward`]. It implements
//! [`tetr_core::ai::eval::Evaluator`], so it drops into
//! `SearchPolicy::new(planner, Box::new(BurnEvaluator…), …)` with **no** engine,
//! controller, or runner changes.
//!
//! # Backends and the async-WebGPU constraint
//!
//! [`Evaluator::evaluate`] is **synchronous** and is called in a tight per-piece
//! loop (once per reachable placement). On `wasm32` in a browser you cannot
//! synchronously block awaiting a WebGPU buffer read-back on the main thread, so
//! a wgpu-backed `evaluate()` cannot complete there. Therefore:
//!
//! - **in-browser / deterministic** inference uses the [`Cpu`] (`ndarray`)
//!   backend — pure Rust, deterministic (replays + tests stay bit-exact), and a
//!   tiny value net is microseconds per call on CPU;
//! - the optional [`Gpu`] (`wgpu`) backend is for **native + offline** training
//!   and fast batched evaluation, where a blocking read-back off the main thread
//!   is fine.
//!
//! # Weights
//!
//! Weights load from **safetensors** (produced by the `distill` bin — the current
//! asset — or the JAX trainer) with tensor names
//! matching this module (`l1.weight`, `l1.bias`, `l2.weight`, `l2.bias`,
//! `head.weight`, `head.bias`). [`ValueNet::from_safetensors`] parses them with
//! the pure-Rust `safetensors` crate and constructs the [`Linear`] params
//! directly — no `burn-import` round-trip.

use burn::module::{Module, Param};
use burn::nn::{Linear, LinearConfig, Relu};
use burn::tensor::backend::Backend;
use burn::tensor::{ElementConversion, Tensor, TensorData};

use safetensors::SafeTensors;

use core::time::Duration;

use tetr_core::ai::eval::{
    compute_reward, BoardFeatures, EvalContext, Evaluator, Reward, RewardWeights, Value,
};
use tetr_core::ai::{AiController, GreedyPlanner, Policy, SearchBudget, SearchPolicy};
use tetr_core::engine::{BitBoard, Board, LockOutcome, TSpinKind};

/// The CPU backend: deterministic, wasm-safe, used for in-engine inference.
pub type Cpu = burn::backend::NdArray<f32>;

/// The GPU backend (native + offline only). Enabled by the `wgpu` feature.
#[cfg(feature = "wgpu")]
pub type Gpu = burn::backend::Wgpu;

/// Number of network inputs: the Dellacherie-6 + BCTS-2 feature set.
pub const NUM_FEATURES: usize = 8;

/// Per-feature divisors mapping raw integer [`BoardFeatures`] into ~O(1) network
/// inputs. **Must** match the preprocessing used by the trainer (the `distill` bin).
/// Order matches [`features_to_input`].
pub const FEATURE_SCALE: [f32; NUM_FEATURES] = [
    20.0, // landing_height
    4.0,  // eroded_piece_cells
    40.0, // row_transitions
    20.0, // column_transitions
    40.0, // holes
    40.0, // board_wells
    40.0, // hole_depth
    20.0, // rows_with_holes
];

/// Failure to load weights into a [`ValueNet`].
#[derive(Debug)]
pub enum LoadError {
    /// The safetensors blob could not be parsed.
    SafeTensors(safetensors::SafeTensorError),
    /// A required tensor was missing or had an unexpected length.
    Shape(String),
}

impl core::fmt::Display for LoadError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            LoadError::SafeTensors(e) => write!(f, "safetensors parse error: {e}"),
            LoadError::Shape(m) => write!(f, "weight shape error: {m}"),
        }
    }
}

impl std::error::Error for LoadError {}

impl From<safetensors::SafeTensorError> for LoadError {
    fn from(e: safetensors::SafeTensorError) -> Self {
        LoadError::SafeTensors(e)
    }
}

/// The value-net architecture: an MLP `NUM_FEATURES -> hidden -> hidden -> 1`.
#[derive(Module, Debug)]
pub struct ValueNet<B: Backend> {
    l1: Linear<B>,
    l2: Linear<B>,
    head: Linear<B>,
    activation: Relu,
}

/// Hyperparameters for [`ValueNet`]. Kept in sync with the JAX training script so
/// imported weights match the Rust module's layer shapes.
#[derive(Debug, Clone)]
pub struct ValueNetConfig {
    /// Width of the two hidden layers.
    pub hidden: usize,
    /// Number of inputs to the first layer — [`NUM_FEATURES`] (the 8 Dellacherie/BCTS
    /// features of the distilled net).
    pub inputs: usize,
}

impl Default for ValueNetConfig {
    fn default() -> Self {
        Self {
            hidden: 64,
            inputs: NUM_FEATURES,
        }
    }
}

impl ValueNetConfig {
    /// Build a freshly-initialized net on `device` (random weights — overwrite via
    /// [`ValueNet::from_safetensors`]).
    pub fn init<B: Backend>(&self, device: &B::Device) -> ValueNet<B> {
        ValueNet {
            l1: LinearConfig::new(self.inputs, self.hidden).init(device),
            l2: LinearConfig::new(self.hidden, self.hidden).init(device),
            head: LinearConfig::new(self.hidden, 1).init(device),
            activation: Relu::new(),
        }
    }
}

impl<B: Backend> ValueNet<B> {
    /// Forward pass: `[batch, NUM_FEATURES] -> [batch, 1]`.
    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.activation.forward(self.l1.forward(x));
        let x = self.activation.forward(self.l2.forward(x));
        self.head.forward(x)
    }

    /// Build a net from JAX-exported safetensors bytes. Tensor names: `l1.weight`
    /// `[NUM_FEATURES, hidden]`, `l1.bias` `[hidden]`, `l2.weight` `[hidden,
    /// hidden]`, `l2.bias` `[hidden]`, `head.weight` `[hidden, 1]`, `head.bias`
    /// `[1]`. (Burn's `linear` computes `x @ weight`, weight `[d_in, d_out]` —
    /// same layout as a Flax `Dense` kernel, so no transpose is needed.)
    pub fn from_safetensors(
        bytes: &[u8],
        inputs: usize,
        hidden: usize,
        device: &B::Device,
    ) -> Result<Self, LoadError> {
        let st = SafeTensors::deserialize(bytes)?;
        let l1 = linear_from_st::<B>(&st, "l1", inputs, hidden, device)?;
        let l2 = linear_from_st::<B>(&st, "l2", hidden, hidden, device)?;
        let head = linear_from_st::<B>(&st, "head", hidden, 1, device)?;
        Ok(ValueNet {
            l1,
            l2,
            head,
            activation: Relu::new(),
        })
    }

    /// Serialize this net's weights to safetensors bytes under the same tensor
    /// names [`ValueNet::from_safetensors`] reads. Lets the Rust bootstrap
    /// distiller (or any Burn-side training) export a model the in-engine loader
    /// can consume — and is backend-agnostic (call on `.valid()` after training).
    pub fn to_safetensors_bytes(&self) -> Result<Vec<u8>, LoadError> {
        use safetensors::tensor::{Dtype, TensorView};
        use std::collections::HashMap;

        let bias = |b: &Option<burn::module::Param<Tensor<B, 1>>>| {
            b.as_ref().expect("ValueNet linear layers always have a bias").val()
        };

        let mut entries: Vec<(String, Vec<usize>, Vec<u8>)> = Vec::new();
        push_tensor(&mut entries, "l1.weight", self.l1.weight.val())?;
        push_tensor(&mut entries, "l1.bias", bias(&self.l1.bias))?;
        push_tensor(&mut entries, "l2.weight", self.l2.weight.val())?;
        push_tensor(&mut entries, "l2.bias", bias(&self.l2.bias))?;
        push_tensor(&mut entries, "head.weight", self.head.weight.val())?;
        push_tensor(&mut entries, "head.bias", bias(&self.head.bias))?;

        let mut views: HashMap<String, TensorView> = HashMap::new();
        for (name, shape, data) in &entries {
            let view = TensorView::new(Dtype::F32, shape.clone(), data)
                .map_err(LoadError::SafeTensors)?;
            views.insert(name.clone(), view);
        }
        let metadata: Option<HashMap<String, String>> = None;
        safetensors::serialize(views, &metadata).map_err(LoadError::SafeTensors)
    }
}

/// Read a named f32 tensor's raw little-endian data out of a safetensors blob.
fn read_f32(st: &SafeTensors, name: &str) -> Result<Vec<f32>, LoadError> {
    let view = st.tensor(name)?;
    let data = view.data();
    if data.len() % 4 != 0 {
        return Err(LoadError::Shape(format!(
            "{name}: byte length {} is not a multiple of 4",
            data.len()
        )));
    }
    Ok(data
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect())
}

/// Append a module tensor to the safetensors export buffer (name, shape, LE bytes).
fn push_tensor<B: Backend, const D: usize>(
    entries: &mut Vec<(String, Vec<usize>, Vec<u8>)>,
    name: &str,
    t: Tensor<B, D>,
) -> Result<(), LoadError> {
    let dims = t.dims().to_vec();
    let data = t
        .into_data()
        .to_vec::<f32>()
        .map_err(|e| LoadError::Shape(format!("{name}: {e:?}")))?;
    let mut bytes = Vec::with_capacity(data.len() * 4);
    for x in &data {
        bytes.extend_from_slice(&x.to_le_bytes());
    }
    entries.push((name.to_string(), dims, bytes));
    Ok(())
}

/// Construct a [`Linear`] layer (`[d_in, d_out]` weight + `[d_out]` bias) from the
/// `{prefix}.weight` / `{prefix}.bias` tensors in a safetensors blob.
fn linear_from_st<B: Backend>(
    st: &SafeTensors,
    prefix: &str,
    d_in: usize,
    d_out: usize,
    device: &B::Device,
) -> Result<Linear<B>, LoadError> {
    let w = read_f32(st, &format!("{prefix}.weight"))?;
    let b = read_f32(st, &format!("{prefix}.bias"))?;
    if w.len() != d_in * d_out {
        return Err(LoadError::Shape(format!(
            "{prefix}.weight: expected {} elems, got {}",
            d_in * d_out,
            w.len()
        )));
    }
    if b.len() != d_out {
        return Err(LoadError::Shape(format!(
            "{prefix}.bias: expected {d_out} elems, got {}",
            b.len()
        )));
    }
    let weight = Param::from_tensor(Tensor::<B, 2>::from_data(
        TensorData::new(w, [d_in, d_out]),
        device,
    ));
    let bias = Param::from_tensor(Tensor::<B, 1>::from_data(
        TensorData::new(b, [d_out]),
        device,
    ));
    Ok(Linear {
        weight,
        bias: Some(bias),
    })
}

/// Map raw features into the normalized fixed-size network input.
pub fn features_to_input(f: &BoardFeatures) -> [f32; NUM_FEATURES] {
    let raw = [
        f.landing_height as f32,
        f.eroded_piece_cells as f32,
        f.row_transitions as f32,
        f.column_transitions as f32,
        f.holes as f32,
        f.board_wells as f32,
        f.hole_depth as f32,
        f.rows_with_holes as f32,
    ];
    let mut out = [0.0f32; NUM_FEATURES];
    let mut i = 0;
    while i < NUM_FEATURES {
        out[i] = raw[i] / FEATURE_SCALE[i];
        i += 1;
    }
    out
}

/// A neural board evaluator: a learned static board [`Value`] composed with
/// tetr-core's principled per-move [`Reward`].
///
/// Generic over the Burn [`Backend`]; instantiate as `BurnEvaluator<Cpu>` for the
/// in-engine / wasm path, or `BurnEvaluator<Gpu>` for native offline runs.
pub struct BurnEvaluator<B: Backend> {
    model: ValueNet<B>,
    device: B::Device,
    reward_weights: RewardWeights,
    /// Multiplier mapping the net's scalar output onto the integer [`Value`] scale.
    value_scale: f32,
}

impl<B: Backend> BurnEvaluator<B> {
    /// Wrap an already-constructed model.
    pub fn new(model: ValueNet<B>, device: B::Device, reward_weights: RewardWeights) -> Self {
        Self {
            model,
            device,
            reward_weights,
            value_scale: 1.0,
        }
    }

    /// Set the output→[`Value`] multiplier (defaults to `1.0`, i.e. the net
    /// regresses the integer value directly).
    pub fn with_value_scale(mut self, scale: f32) -> Self {
        self.value_scale = scale;
        self
    }

    /// Load weights from JAX-exported safetensors bytes (e.g. `include_bytes!`)
    /// into a `config`-shaped net.
    pub fn from_safetensors(
        bytes: &[u8],
        config: &ValueNetConfig,
        device: B::Device,
        reward_weights: RewardWeights,
    ) -> Result<Self, LoadError> {
        let model = ValueNet::<B>::from_safetensors(bytes, config.inputs, config.hidden, &device)?;
        Ok(Self::new(model, device, reward_weights))
    }
}

impl<B: Backend> Evaluator for BurnEvaluator<B> {
    fn evaluate(
        &self,
        lock: &LockOutcome,
        board: &Board,
        t_spin: Option<TSpinKind>,
        ctx: EvalContext,
    ) -> (Value, Reward) {
        // Input = the 8 Dellacherie board features; the reward half uses `ctx`.
        let input_vec = features_to_input(&BoardFeatures::extract(board, lock)).to_vec();
        let input =
            Tensor::<B, 2>::from_data(TensorData::new(input_vec, [1, NUM_FEATURES]), &self.device);
        let out = self.model.forward(input);
        let raw: f32 = out.into_scalar().elem();
        let value = Value((raw * self.value_scale).round() as i32);
        let reward = compute_reward(&self.reward_weights, lock, board, t_spin, ctx);
        (value, reward)
    }

    /// Score a whole search generation in **one** forward pass.
    ///
    /// Stacks the `N` feature rows into a single `[N, NUM_FEATURES]` tensor (so the
    /// 1-row reshape path is never hit for `N != 1`), runs `forward` once, then
    /// re-uses tetr-core's [`compute_reward`] per item for the [`Reward`] half.
    /// Bit-identical to mapping [`evaluate`](Self::evaluate) over the same inputs.
    fn evaluate_batch(
        &self,
        inputs: &[(&LockOutcome, &BitBoard, Option<TSpinKind>, EvalContext)],
    ) -> Vec<(Value, Reward)> {
        let n = inputs.len();
        if n == 0 {
            return Vec::new();
        }
        let mut flat = Vec::with_capacity(n * NUM_FEATURES);
        for (lock, board, _, _) in inputs {
            // NN feature extraction still consumes a dense `Board`; reconstruct per row.
            // This IS the NN evaluator's hot loop (the beam's per-generation scoring);
            // moving extraction onto `columns()` (an `extract_from_cols` seam) is a follow-up
            // that would drop this `to_array2d`.
            let dense = board.to_array2d();
            flat.extend_from_slice(&features_to_input(&BoardFeatures::extract(&dense, lock)));
        }
        let x = Tensor::<B, 2>::from_data(TensorData::new(flat, [n, NUM_FEATURES]), &self.device);
        let out = self.model.forward(x); // [N, 1] — ONE forward
        let raw: Vec<f32> = out.into_data().to_vec().expect("f32 value-net output");
        inputs
            .iter()
            .zip(raw)
            .map(|((lock, board, t, ctx), v)| {
                let value = Value((v * self.value_scale).round() as i32);
                let reward = compute_reward(&self.reward_weights, lock, &board.to_array2d(), *t, *ctx);
                (value, reward)
            })
            .collect()
    }
}

/// Build an [`AiController`] driven by the neural value net loaded from
/// `model_bytes`, honoring the `reaction` delay and `imperfection` handicap.
///
/// This is the drop-in **production** constructor: same behavior contract as
/// `AiController::new`, but the static board [`Value`] comes from the net (CPU
/// backend) instead of the linear evaluator, while the per-move [`Reward`] stays
/// the engine-faithful `reward_weights`. Callers embed the weights with
/// `include_bytes!` and fall back to the linear bot if this returns `Err`.
pub fn nn_ai_controller(
    model_bytes: &[u8],
    reward_weights: RewardWeights,
    reaction: Duration,
    imperfection: f32,
    seed: u64,
) -> Result<AiController, LoadError> {
    let device = Default::default();
    let eval = BurnEvaluator::<Cpu>::from_safetensors(
        model_bytes,
        &ValueNetConfig::default(),
        device,
        reward_weights,
    )?;
    let policy = SearchPolicy::new(
        Box::new(GreedyPlanner::new()),
        Box::new(eval),
        SearchBudget::greedy(),
        imperfection,
        seed,
    );
    Ok(AiController::with_policy(
        Box::new(policy) as Box<dyn Policy>,
        reaction,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use tetr_core::engine::{CellKind, PieceType};

    /// A fresh (random-init) CPU evaluator — enough to pin the batch == scalar
    /// equivalence (the seam the beam relies on); the actual weights are irrelevant
    /// to that property.
    fn fresh_eval() -> BurnEvaluator<Cpu> {
        let device = Default::default();
        let model = ValueNetConfig::default().init::<Cpu>(&device);
        BurnEvaluator::new(model, device, RewardWeights::SURVIVAL)
    }

    #[test]
    fn nn_batch_matches_scalar() {
        // `evaluate_batch` (one [N, NUM_FEATURES] forward) must be bit-identical to
        // mapping the scalar `evaluate` over the same inputs, on the CPU backend.
        let eval = fresh_eval();

        // A no-clear O lock.
        let mut board_a = Board::new(10, 20);
        board_a.set(0, 0, CellKind::Some(PieceType::O));
        let lock_a = LockOutcome {
            cells_locked: vec![(0, 0, CellKind::Some(PieceType::O))],
            cleared_rows: Vec::new(),
            top_y_after_lock: Some(0),
        };

        // A board with some structure (holes / wells) and a no-clear lock.
        let mut board_b = Board::new(10, 20);
        for x in 0..9 {
            board_b.set(x, 0, CellKind::Some(PieceType::I));
            board_b.set(x, 1, CellKind::Some(PieceType::I));
        }
        board_b.set(2, 2, CellKind::Some(PieceType::L)); // overhang -> hole below
        let lock_b = LockOutcome {
            cells_locked: vec![(2, 2, CellKind::Some(PieceType::L))],
            cleared_rows: Vec::new(),
            top_y_after_lock: Some(2),
        };

        // A Tetris on a board with a stray cell (exercises the Reward path too).
        let mut board_c = Board::new(10, 20);
        board_c.set(0, 0, CellKind::Some(PieceType::O));
        let lock_c = LockOutcome {
            cells_locked: vec![(0, 0, CellKind::Some(PieceType::I))],
            cleared_rows: vec![0, 1, 2, 3],
            top_y_after_lock: None,
        };

        let ctx = EvalContext::default();
        let bb_a = BitBoard::from_board(&board_a);
        let bb_b = BitBoard::from_board(&board_b);
        let bb_c = BitBoard::from_board(&board_c);
        let inputs: Vec<(&LockOutcome, &BitBoard, Option<TSpinKind>, EvalContext)> = vec![
            (&lock_a, &bb_a, None, ctx),
            (&lock_b, &bb_b, None, ctx),
            (&lock_c, &bb_c, None, ctx),
        ];

        // Batched (bitboard) must equal scalar `evaluate` on the equivalent dense boards.
        let batched = eval.evaluate_batch(&inputs);
        let scalar = vec![
            eval.evaluate(&lock_a, &board_a, None, ctx),
            eval.evaluate(&lock_b, &board_b, None, ctx),
            eval.evaluate(&lock_c, &board_c, None, ctx),
        ];

        assert_eq!(batched, scalar);
    }

    #[test]
    fn nn_batch_empty_is_empty() {
        let eval = fresh_eval();
        let inputs: Vec<(&LockOutcome, &BitBoard, Option<TSpinKind>, EvalContext)> = Vec::new();
        assert!(eval.evaluate_batch(&inputs).is_empty());
    }
}
