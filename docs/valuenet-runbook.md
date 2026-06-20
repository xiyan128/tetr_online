# Value-net strike — runbook

The operational guide for `valuenet/strike`. Architecture + rationale: `adr-valuenet-strike.md`.

## Pipeline

```
bc-distill ─→ shards (.safetensors) ─→ train.py ─→ value_net.{safetensors,json}
   (Rust)         tetr-valuenet            (PyTorch)          │
                                                              ├─→ golden.py + verify_infer  (Rust == PyTorch)
                                                              └─→ value-gate                (iron gate vs CC2)
```

## Commands

**1. Distill a dataset** (drive the expert under rain, label by its deep-search value):
```
cargo run --release -p tetr-research -- run bc-distill probe-tp128d9 \
    --out-dir valuenet/data/champion-v1 --allow-dirty
# canary: run bc-distill-smoke probe-tp32d6 --out-dir /tmp/x
```

**2. Train** (val R² is the headline — can the net predict the expert's value on held-out games?):
```
cd python && uv run python -m valuenet.train \
    --data-dir ../valuenet/data/champion-v1 --out ../valuenet/models/v1
```

**3. Verify Rust inference == PyTorch** (catches forward-pass layout bugs):
```
cd python && uv run python -m valuenet.golden --model-dir ../valuenet/models/v1 \
    --out ../valuenet/models/v1/golden.safetensors
cargo run --release -p tetr-valuenet --bin verify_infer -- \
    valuenet/models/v1 valuenet/models/v1/golden.safetensors
```

**4. The iron gate (E9)** — does the learned eval beat CC2 at iso-search, before adding depth?
```
cargo run --release -p tetr-research -- run value-gate \
    --net-dir valuenet/models/v1 --allow-dirty
```

## Status / milestones

- **M0 (data)** ✅ — encoder (`encode.rs`, the input schema), `bc-distill` exporter, safetensors shards.
- **M1 (train + native infer + gate)** — pipeline ✅; iron-gate verdict in progress.
- **M2 (fixed-point + wasm)** — `i32` forward pass, bit-identical; re-confirm the gate; wasm size.
- **M3 (autoresearch)** — `/autoresearch` over architecture / targets / death-coverage / depth.

## Open items (for M3 / the final dataset)

- **Death coverage.** The default `bc-distill` (rain period 8) lets a strong teacher survive
  (`death% ≈ 0`), so the net sees few dying boards — the exact failure of the pruned `tetr-nn`
  stack. The final dataset must MIX rain regimes: a heavy-rain pass (period ≈ 2, ~30% death, as
  `value-probe-heavy` showed) over-samples near-death states. *Add a `bc-distill-heavy` variant
  and pool it with the default before the champion run.*
- **Teacher.** v1 uses `tp16d9` (fast d9) to validate the pipeline; the final teacher is the
  champion `tp128d9` (more width = better d9 search = stronger labels).
- **1× search.** `bc-distill` currently searches twice per piece (controller plays + label
  search). Driving directly by the label's placement (`placement_to_inputs`) would halve it.
- **Reward handling.** The net's value is the whole leaf (`Reward=0`). If the gate is close, try
  net-value + CC2-reward (swap only the static Value half) — but that needs a static-value target.
