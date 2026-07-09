---
id: T13
title: Forward throughput fix — ANE fusion (datagen) + BNNS/ANE (deployed)
labels: [wayfinder:prototype]
status: open
assignee:
blocked-by: []
---

## Question

T03 measured the tetr-nn BLAS forward at ~6.9k evals/s peak, batch-invariant, ~99% CPU-side im2col/transpose glue (<1% GEMM) — a **datagen prerequisite**, since net self-play at this eval cost starves (the w8d5 result, and the prior campaign's history). Also: memory's "20.5k BLAS" was wrong by 3×, so its "ANE 34k–150k" is suspect too.

Resolve, with measurements:
1. **Where does the 145 µs/eval go?** Confirm im2col materialization + the channel-major transpose dominate (profile). 
2. **Fix path (per check-existing-tools — do NOT hand-roll kernels):** the cheapest correct option that clears the floor — implicit/direct conv, a mature inference lib (candle / tract / ort), or ANE offload. Bench each on *this* net; pick by evals/s per unit effort, keeping the golden parity contract (PyTorch↔Rust to 1e-4).
3. **Re-verify the ANE path** on the round0/conv net: actual batched evals/s on this hardware (the rl-worktree `OnnxEvaluator` is the reference); does it beat a fixed CPU forward, and is its per-call overhead compatible with a sequential MCTS (small batches) vs batched datagen (large batches)?
4. Determinism note: ANE/candle f32 may drop bit-identical replay — confirm the gate stays on Elo/SPRT so this is acceptable (per the design's determinism stance).

**Acceptance test:** a forward path that lets the Gumbel-n64 operator sustain ≥200 games/hr locally (the T03 THROUGHPUT floor). Feeds the datagen architecture (T09) and the design freeze.

## Root cause CONFIRMED (traffic calculation, 2026-07-08)

The im2col memory traffic exactly accounts for the measured wall-clock, so the forward is **memory-bandwidth-bound on manual im2col**, not compute or GEMM:
- conv3 (32→32) at n=1024: im2col writes `1024·400·32·9 = 118M` floats ≈ 472 MB → ~94 ms at ~5 GB/s.
- conv2 (16→32): `1024·400·16·9 = 59M` ≈ 236 MB → ~47 ms.
- Sum ≈ 140 ms vs the **measured 151 ms/batch** (148 µs/eval × 1024). Match confirms im2col is THE cost; the GEMM itself is <1%.

**Consequence:** no dependency-free restructure fixes this (the MACs need GEMM, GEMM needs the gathered columns, and the gather is the bandwidth wall). The fix must be a conv primitive that avoids materialized im2col.

## The fork (both are mature primitives — honors check-existing-tools, no kernel hand-rolling)

- **BNNS (Accelerate's neural subroutines) — ALREADY LINKED**, no new dependency. CPU implicit conv (no materialized cols). Expected moderate win (3–5×?), CPU-only. FFI to the Accelerate framework's `BNNS*` functions (bounded but real).
- **ANE via CoreML (ort + coreml EP) — exists on the rl branch** (`crates/tetr-valuenet/src/onnx.rs` + `server.rs` fusion server). Recorded receipts (credible — specific methodology): **~13k evals/s small-batch (per-search), 34k @ batch 3840, 150k @ 480 fused** on conv_rb1. That is 2× (sequential) to ~20× (fused large-batch) the BLAS 6.9k. The datagen throughput answer is **ANE + cross-game fusion** (many parallel games → one large ANE batch). Cost: port ort/coreml + onnx-export the two-board `tetr-nn` net + wire the fusion server. Native-only (fine — research bot).

**Decision:** for **datagen throughput** (the campaign's binding constraint) the answer is the **ANE fusion path** (port from the rl branch; large-batch is where the 34k+ lives). For the **deployed sequential search**, small-batch ANE (~13k) or BNNS is the question — measure both once wired. Re-verify the ANE receipts fresh on the two-board net as the first step (no committed .onnx exists — re-export needed).

## T13 progress (2026-07-09): ONNX export landed, parity tail unresolved

`python/tetrnn/export_onnx.py` exports `net_leaf.onnx` + `net_slots.onnx` (dynamic batch, opset 17) from any model dir; verified loadable in onnxruntime. Parity vs the torch forward: **median fp16-scale (~1e-4, k/4096 steps from the exporter's graph optimization) but the tail spikes to ~1e-1 on some inputs** — likely a boundary unit flipping under reduced precision, amplified through the trunk. Fine for datagen-grade eval; **investigate before using the graph in gating races.** Next: ort/CoreML EP integration in Rust (the rl-branch OnnxEvaluator is the reference) + the ANE throughput re-measurement on this net.

## Parity tail RESOLVED (2026-07-09): test artifact, graph is faithful

The ~1e-1 "tail" was my test's fault: random values fed into **training-constant features** (std floored at MIN_STD=1e-6) standardize to ~3e6 and blow trunk activations to ~5e5, where fp32 reorder noise is large in ABSOLUTE terms (2e-7 RELATIVE). With in-distribution inputs (z clamped ±3): **relative parity median 2.3e-7, max 3.2e-6**. The export path is fully validated for the ort/CoreML integration.

**Latent landmine noted:** opp features + venue clocks are constant-zero in ALL current training (OppCtx::default) → their whitening std is floored → any future distribution shift in those features (e.g. wiring `set_opponent`) multiplies by ~1e6. When the two-board path activates, re-derive whitening or exclude constant features from standardization.

## ANE/CoreML RE-VERIFIED on the two-board net (2026-07-09) — the throughput future

Measured via onnxruntime-python on the fixed-batch graphs, **while contended by a running round** (lower bounds):

| backend | b=1 | b=34 | b=68 | b=128 | b=480 | b=1024 |
|---|--:|--:|--:|--:|--:|--:|
| our BLAS forward (uncontended ref) | 4.4k | 6.2k | — | 6.7k | 6.7k | 6.9k |
| tract (pure Rust) — REFUTED | 1.8k | 1.4k | 1.3k | 1.2k | 0.7k | 0.8k |
| ort CPU (MLAS) | 3.0k | 6.2k | 9.4k | 9.1k | 6.7k | 5.8k |
| **ort CoreML EP (MLProgram)** | — | — | **30.0k** | — | **116.7k** | — |
| ort CoreML (force CPU+ANE only) | — | — | 13.4k | — | 13.4k | — |

**CoreML MLProgram (default compute units — GPU+ANE dispatch) = 4-17× our forward**, exceeding the rl-branch's recorded receipts, on the real two-board net. Gotchas burned in: the EP needs `ModelFormat: MLProgram` (the default NeuralNetwork format fails init on partitioned opset-18 graphs: "model_path must not be empty"); fixed-batch graphs (the dynamo exporter's dynamic-batch graph also breaks tract's shape inference).

**Remaining build:** the Rust `ort` integration (feature `coreml`, MLProgram flag; rl-branch `onnx.rs` is the reference) as a tetr-nn evaluator backend for datagen + the guided filter. With per-sibling batches (~68) at 30k evals/s, datagen forward cost drops ~4×; a cross-game fusion server at b≈480 unlocks the 116k regime.

## T13 BUILT (2026-07-09): the CoreML backend is in the tree

`tetr-nn` feature `coreml`: `OrtBackend` (bucketed fixed-batch MLProgram sessions {34,68,128,480}, pad-up, chunk-down) + `OrtNetEvaluator` (evaluate_leaves twin of NetEvaluator) + the datagen seam (`--features coreml` + `TETR_ORT=1`). Rust-measured: **33k @ b68 / 78k @ b480 evals/s (contended)** = 5-11× BLAS. Export side: single-file graphs (`external_data=False` — ort rc.12 mangles external-data paths) + fixed buckets.

**Amdahl verdict at the current vehicle:** datagen A/B (same seeds) = 1,345 → 1,511 games/hr (**1.12×**) — the action-indexed head already removed eval from the datagen critical path (top-12 = ~12 evals/node; bookkeeping+encode dominate). The backend's value is the NEXT phase: deep-config races (w68+ leaf batches at the champion-parity budget) and b480 fusion-server datagen. Default build pulls no ort; gate unaffected.

## Perf block (2026-07-09 night): the bookkeeping ladder opens

Duel-path profile (macOS sample): `Net::embed_boards` (im2col glue) tops on-CPU cost, but the Amdahl reality at m12 configs is that eval is amortized — **search bookkeeping is the wall** (matches the historical movegen-44%/memmove-22% profile). Actions:
- Net arms now honor `TETR_ORT` (CoreML leaf eval in duels/gates too) — play-identical on paired seeds; the win arrives at deep configs.
- **Stage-0 deferred lever #3 LANDED**: partial selection in `ranked_frontier` (bit-identical: 280 tests + exact champion-duel replay 6-6-0/[2,10,0]). Est 5-8%; idle measurement pending.
- Queued from the same ledger, in order: #1 pathless BFS for internal plies (8-15%), #2 deferred materialization of speculative continuations (~10%), #4 bitboard movegen (25-30%, rewrite risk — research mature references first).
