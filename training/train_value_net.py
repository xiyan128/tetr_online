#!/usr/bin/env python3
"""Train the tetr-nn value net (JAX/Flax) and export weights as safetensors.

This is the *model development* half of the tetr-nn loop. It trains the MLP that
`crates/tetr-nn/src/lib.rs::ValueNet` runs at inference time, and exports weights
under the exact tensor names that crate's `from_safetensors` reads:

    l1.weight [8, hidden]   l1.bias [hidden]
    l2.weight [hidden, hidden]   l2.bias [hidden]
    head.weight [hidden, 1]   head.bias [1]

Flax `Dense` stores its kernel as [in, out], the same layout Burn's `linear`
expects ( y = x @ weight ), so no transpose is needed on export.

v1 target = DISTILL the existing hand-tuned linear evaluator (DT-20 board
weights). The net learns to reproduce `value = DT20 . features`, giving an NN
that starts at ~baseline parity for /autoresearch to climb from, and exercising
the full JAX -> safetensors -> Burn pipeline end to end.

Upgrade path: pass `--data board_features.csv` (rows of 8 raw features + a target
column) generated from real gameplay by the Rust dataset dumper, instead of the
synthetic sampler here. Same training code, better signal.

Deps:  pip install jax flax optax numpy safetensors
Run:   python training/train_value_net.py --out crates/tetr-nn/assets/value_net.safetensors
"""

from __future__ import annotations

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn
from safetensors.numpy import save_file

# --- Constants shared with crates/tetr-nn/src/lib.rs (keep in sync!) -----------

# DT-20 board weights, in the feature order of `BoardFeatures` / features_to_input.
DT20 = np.array(
    [
        -2.68,  # landing_height
        1.38,  # eroded_piece_cells
        -2.41,  # row_transitions
        -6.32,  # column_transitions
        2.03,  # holes
        -2.71,  # board_wells
        -0.43,  # hole_depth
        -9.48,  # rows_with_holes
    ],
    dtype=np.float32,
)

# Per-feature divisors — MUST equal tetr_nn::FEATURE_SCALE.
FEATURE_SCALE = np.array([20.0, 4.0, 40.0, 20.0, 40.0, 40.0, 40.0, 20.0], dtype=np.float32)

# Rough upper bounds for synthetic sampling of each raw feature (10x20 board).
FEATURE_MAX = np.array([20.0, 16.0, 80.0, 40.0, 80.0, 80.0, 80.0, 20.0], dtype=np.float32)

NUM_FEATURES = 8
HIDDEN = 64


class ValueNet(nn.Module):
    hidden: int = HIDDEN

    @nn.compact
    def __call__(self, x):
        x = nn.relu(nn.Dense(self.hidden, name="l1")(x))
        x = nn.relu(nn.Dense(self.hidden, name="l2")(x))
        x = nn.Dense(1, name="head")(x)
        return x


def synthetic_dataset(n: int, rng: np.random.Generator):
    """Sample raw features over a plausible domain; target = DT20 . raw."""
    raw = rng.uniform(0.0, FEATURE_MAX, size=(n, NUM_FEATURES)).astype(np.float32)
    target = raw @ DT20  # the linear evaluator's value, [n]
    inputs = raw / FEATURE_SCALE  # what the net actually sees (normalized)
    return inputs.astype(np.float32), target.astype(np.float32)


def load_csv_dataset(path: Path):
    """Rows: f0..f7 (raw features), target. Normalizes inputs by FEATURE_SCALE."""
    arr = np.loadtxt(path, delimiter=",", dtype=np.float32)
    raw, target = arr[:, :NUM_FEATURES], arr[:, NUM_FEATURES]
    return (raw / FEATURE_SCALE).astype(np.float32), target.astype(np.float32)


def train(inputs, target, *, epochs: int, batch: int, lr: float, seed: int):
    model = ValueNet()
    key = jax.random.PRNGKey(seed)
    params = model.init(key, jnp.ones((1, NUM_FEATURES), jnp.float32))["params"]
    opt = optax.adam(lr)
    opt_state = opt.init(params)

    inputs_j, target_j = jnp.asarray(inputs), jnp.asarray(target)[:, None]

    @jax.jit
    def step(params, opt_state, xb, yb):
        def loss_fn(p):
            pred = model.apply({"params": p}, xb)
            return jnp.mean((pred - yb) ** 2)

        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = opt.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    n = inputs_j.shape[0]
    rng = np.random.default_rng(seed)
    for epoch in range(epochs):
        perm = rng.permutation(n)
        last = 0.0
        for i in range(0, n, batch):
            idx = perm[i : i + batch]
            params, opt_state, loss = step(params, opt_state, inputs_j[idx], target_j[idx])
            last = float(loss)
        if epoch % max(1, epochs // 10) == 0 or epoch == epochs - 1:
            print(f"epoch {epoch:4d}  mse {last:12.3f}")
    return model, params


def export_safetensors(params, out: Path):
    p = params  # {"l1": {"kernel","bias"}, "l2": {...}, "head": {...}}
    tensors = {
        "l1.weight": np.asarray(p["l1"]["kernel"], np.float32),
        "l1.bias": np.asarray(p["l1"]["bias"], np.float32),
        "l2.weight": np.asarray(p["l2"]["kernel"], np.float32),
        "l2.bias": np.asarray(p["l2"]["bias"], np.float32),
        "head.weight": np.asarray(p["head"]["kernel"], np.float32),
        "head.bias": np.asarray(p["head"]["bias"], np.float32),
    }
    for name, t in tensors.items():
        print(f"  {name:14s} {tuple(t.shape)}")
    out.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(out))
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("crates/tetr-nn/assets/value_net.safetensors"))
    ap.add_argument(
        "--data", type=Path, default=None, help="CSV of raw features + target (else synthetic)"
    )
    ap.add_argument("--samples", type=int, default=200_000)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.data is not None:
        inputs, target = load_csv_dataset(args.data)
        print(f"loaded {inputs.shape[0]} rows from {args.data}")
    else:
        inputs, target = synthetic_dataset(args.samples, np.random.default_rng(args.seed))
        print(f"synthetic distillation set: {inputs.shape[0]} samples of DT20 . features")

    _, params = train(
        inputs, target, epochs=args.epochs, batch=args.batch, lr=args.lr, seed=args.seed
    )
    export_safetensors(params, args.out)


if __name__ == "__main__":
    main()
