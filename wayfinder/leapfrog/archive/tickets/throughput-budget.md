---
id: T03
title: Throughput budget model — the data rate a real campaign needs
labels: [wayfinder:research]
status: closed
assignee: fable-lead
blocked-by: []
---

## Question

The prior round starved at 206/2000 games; that failure mode must be priced before any design freezes. Build a parameterized budget model:

games/round × rounds-to-SOTA × decisions/game × sims/decision × cost/sim ⇒ required evals/s and wall-clock/round — bracketed across candidate search vehicles (policy-only ≈ 1 eval/move; Gumbel-MCTS 16–64 sims; beam w8d5 ≈ 10²–10³ evals/move).

Anchor with measured receipts (rl worktree + fresh micro-benchmarks on master where stale): local BLAS ≈ 20.5k evals/s, hybrid CPU+ANE ≈ 23.7k, ANE batch receipts 34k–150k; net-driven self-play was 15–90 games/hr vs the CC2 ladder's ~20k/hr.

Output: the design floor the datagen architecture must hit (a) on the Mac for a viable local round cadence, (b) at cloud scale — and where the prior starvation sits in the model (which multiplier killed it). This number is an input to both the datagen architecture and the design freeze.

## Resolution

Resolved 2026-07-07 with fresh measurements on master (this Mac, ~8 effective cores, Accelerate). Full log + budget model: [T03 measurements](../assets/T03-throughput-measurements.md).

**Two load-bearing findings (both correct memory's stale numbers):**

1. **The tetr-nn BLAS forward is glue-bound, not compute-bound.** Peaks ~**6.9k evals/s** (memory claimed 20.5k) and barely scales with batch (1→1024 = 1.6×); per-eval cost is flat ~145 µs = ~0.35 GFLOP/s effective vs Accelerate's 100+. ~99% is im2col/transpose scalar glue, <1% GEMM. **Fixing the forward is a datagen prerequisite, not an optimization.** The ANE 34k–150k claim is now also suspect (same source) — re-verify.

2. **The starving multiplier is `evals/move × s/eval` for a net search, amplified by the game-length confound.** Measured self-play: policy/value (1 fwd/move, weak→short games) 3.3–4.7k games/hr; champion (deep CC2 beam, cheap eval, strong→long games) **278 games/hr**; `beam:round0@w8d5` (hundreds of net evals/move, glue-bound) **<100 games/hr** — reproducing the prior campaign's 15–90/hr starvation exactly.

**Design floor:** ≥200 games/hr sustained locally for a strong-ish agent (overnight 2000-game round) before any campaign is authorized — the THROUGHPUT STOP. The recommended Gumbel-n64 operator is **marginal on the current forward** (~150–300 games/hr) and **clears comfortably (~300–600/hr) once the forward is fixed** (implicit conv / mature lib / ANE), scaling linearly on cloud. This throughput-justifies the T02 operator choice and makes "fix the forward + re-verify ANE" a near-term ticket.
