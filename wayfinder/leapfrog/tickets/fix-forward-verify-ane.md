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
