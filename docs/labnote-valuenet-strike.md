# Lab note — value-net strike (Phase A)

**Date:** 2026-06-20 · **Branch:** `valuenet/strike` · **Status:** Phase A built; learning signal
positive; strength gate blocked on inference cost. Forward plan: `learned-evaluation-roadmap.md`.

The first end-to-end pass at a learned leaf eval, built to be cheap and gated. Goal: distill the
champion's deep-search value into a **raw-input** net, verify it learns, and race it vs CC2 at
iso-search (the iron gate). This note records what was built and what it measured.

## What was built

| Piece | Where | Notes |
|---|---|---|
| Raw-obs encoder | `crates/tetr-valuenet/src/encode.rs` | `SearchState → [24×10 occupancy plane] + [70 raw features]` (active/hold/queue/bag one-hots, combo/b2b, pending garbage). Single source of truth for export *and* inference. 4 unit tests (bit-exact vs an engine snapshot). |
| Dataset format | `…/dataset.rs`, `sample.rs` | safetensors shards (board, features, value, future_attack, died_soon). |
| `bc-distill` exporter | `crates/tetr-research/src/commands/bc_distill.rs` | drives the expert under rain; labels each position with the teacher's own `think_to_completion` value. Adds `BotSpec::planner_parts` + `Runtime.out_dir`. |
| `evaluate_state` seam | `crates/tetr-core/src/ai/eval/mod.rs` | trait method giving the eval the full `SearchState` (the trait otherwise sees only the board); defaults to `evaluate_cols` → byte-identical for handcrafted evals (**all 277 core tests pass**). |
| PyTorch trainer | `python/valuenet/` (uv) | 2.0M-param board CNN; leakage-aware shard split; val-R² headline; best-checkpoint export. |
| Rust `f32` inference | `…/infer.rs` + `bin/verify_infer` | hand-rolled forward pass mirroring `model.py`; `ValueNetEvaluator` (net value = whole leaf, `Reward=0`). |
| `value-gate` iron gate | `crates/tetr-research/src/commands/value_gate.rs` | net-leaf beam vs CC2-leaf beam at iso-search, rain GSPRT. |
| Golden cross-check | `python/valuenet/golden.py` + `verify_infer` | PyTorch↔Rust forward-pass equivalence. |

Commits: `8ee2240` (M0 + trainer) → `db50d2d` (inference + seam + gate) → `ae6d8d6` (docs).

## Results

- **The net learns the teacher's value.** A 2.0M-param CNN reached **val R² ≈ 0.638** on held-out
  *games* (clean shard-level split — no within-game leakage), from raw input alone. Curve: val R²
  rose to ~0.63 by epoch ~14, train R² kept climbing to 0.83 (mild overfit → best-checkpoint export).
  This **refutes the DT-20 failure mode** of the pruned `tetr-nn` stack (a weak teacher capped the old
  net at ~0.2 APP); a strong teacher + raw input genuinely learns.
- **Rust inference == PyTorch.** Max relative error **1.3e-6** over the golden set — the hand-rolled
  forward pass is correct (no layout bug).
- **Inference is ~1000× too slow to race.** The `f32` per-leaf CNN is ~6 ms/eval; `w16d6` scores
  ~1,600 leaves/decision → **~10 s/decision, ~40 min/game**. The `w16d6` iron gate cannot finish a
  game (the GSPRT budget is checked between games, so it can't even bound the first). **The strength
  verdict is therefore not yet measured.**
- **Dataset (`tp16d9-v1`):** 76,800 positions / 256 rain games. Notably `death% 0` at rain period 8 —
  a strong d9 teacher survives, so this dataset has **weak near-death coverage** (the second `tetr-nn`
  failure mode); the final dataset needs heavy rain + DAgger (see roadmap §4/§5).

## Findings

1. **Positive:** raw input + a strong-teacher target is learnable (val R² 0.64) — the prediction-level
   bet has signal. The pipeline (encode → distill → train → verify → integrate) is sound and gated.
2. **Open / blocking:** a per-node dense net is the wrong runtime for a beam. The strength gate needs
   **batched evaluation** (or the NNUE student) before it can run at `w16d6`; a **depth-1** read gives
   the first eval-quality signal cheaply. (Roadmap §5.)
3. **Caveats not yet addressed (deferred to Phase B+):** Phase A's target is the teacher's CC2 *score*
   (uncalibrated, not reward units); coverage is the deterministic champion's narrow distribution
   (needs DAgger + heavy rain); the value is a single-board *proxy*, not win/loss.

## Receipts

- Data: `valuenet/data/tp16d9-v1/` (+ `meta.json`). Model: `valuenet/models/tp16d9-v1/`
  (`value_net.{safetensors,json}`, `best_val_r2 0.638`). Logs: `valuenet/m1-verdict.log`.
- Reproduce: `docs/valuenet-runbook.md`.
