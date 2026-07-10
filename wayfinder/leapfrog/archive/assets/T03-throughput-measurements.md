# T03 — Throughput measurements (raw log)

Hardware: this Mac, 12 logical / 8 performance cores (Apple silicon), Accelerate BLAS.
Code: master HEAD 575be51. Net: `round0` fixture (siamese conv 16/32/32, the STAGE1 arch), a real trained P+V net (start-gates: std_z_hat 0.458, policy top-1 0.639, wdl_auc 0.607).
All self-play numbers are MIRROR duels (arm vs itself) under the sudden-death venue (rain 8, cap 240) via `tetr-research duel`, which parallelizes pairs across cores with rayon.

## A. Forward microbench (`crates/tetr-nn/tests/throughput.rs`, batched `Net::forward`)

| batch | evals/s | µs/batch | µs/eval |
|------:|--------:|---------:|--------:|
| 1     | 4,356   | 229.6    | 229.6   |
| 8     | 5,501   | 1,454    | 181.8   |
| 16    | 5,509   | 2,904    | 181.5   |
| 34    | 6,151   | 5,528    | 162.6   |
| 64    | 6,591   | 9,711    | 151.7   |
| 128   | 6,727   | 19,028   | 148.7   |
| 256   | 6,557   | 39,042   | 152.5   |
| 480   | 6,685   | 71,800   | 149.6   |
| 1024  | 6,909   | 148,220  | 144.7   |
| 2048  | 4,761   | 430,128  | 210.0   |
| 4096  | 3,195   | 1,282,009| 313.0   |

**Finding (load-bearing):** batching 1→1024 is only **1.6×**; per-eval cost is essentially flat (~145–230 µs). At ~7M MACs/eval and 145 µs, effective throughput is ~0.35 GFLOP/s — Accelerate does 100+ GFLOP/s, so the GEMM is <1% of the time. The forward is **~99% CPU-side glue** — the im2col materialization (per eval ≈ HW·Σcin·9 ≈ 400·(1+16+32)·9 ≈ 176k scalar scatter-writes, memory-bound) plus the channel-major transpose (32·400 writes). This **contradicts the memory note of "20.5k evals/s BLAS"** (that figure was the conv_rb1 HandNet on a different path / the ANE, not this forward). Batching does NOT rescue it because the glue is O(n). Past ~1024 it regresses (cache/working-set).

**Implication:** the single biggest cheap throughput lever is fixing the forward (implicit/direct conv or a mature inference lib — candle/tract/ort — instead of hand-rolled im2col; or ANE/GPU offload which does the conv natively). Peak batched ≈ **6.9k evals/s** today, glue-bound.

## B. Self-play games/hour (aggregate, core-saturating where noted)

| config | evals/move | games | wall (s) | games/hr | notes |
|--------|-----------:|------:|---------:|---------:|-------|
| `policy:round0` mirror | ~34 (1 fwd) | 16 (8 pairs) | 17.0 | **3,295** | weak bot, all topout (SHORT games) |
| `value:round0` mirror (d1 argmax) | ~34 (1 fwd) | 12 (6 pairs) | 9.1 | **4,662** | weak bot, all topout (SHORT games) |
| `tp:cc2@w128d9` mirror (champion) | ~972 nodes | 12 (6 pairs) | 155.0 | **278** | STRONG, 10/12 escalation (LONG ~240+ ply games), CC2 eval (cheap) |
| `beam:round0@w8d5` mirror | ~hundreds | 2 (killed) | >390 CPU | **<100** | net beam; 2 games unfinished after 6.5 min CPU; glue-bound forward × hundreds of evals/move |

**The game-length confound (decisive for the budget):** games/hr is a product of *per-move cost* AND *game length*, and game length ∝ bot strength — weak bots top out early (short games, high games/hr but useless data); strong bots survive to sudden-death escalation (~240+ plies, long games). So the honest datagen unit for a **strong** agent is expensive on *both* axes. The champion (278/hr) is the strong-agent anchor with a **cheap** eval; a strong *net* agent pays the glue-bound forward on top of that → the w8d5 starvation, which reproduces the prior campaign's 15–90 games/hr history exactly.

## C. Budget model

```
games/hr_aggregate ≈ (cores · 3600) / (moves/game · evals/move · s/eval  +  moves/game · nodes/move · s/node_bookkeeping)
                                        └─────── EVAL cost ───────┘        └────────── SEARCH cost ──────────┘
```

Anchors (this Mac, ~8 effective cores): s/eval ≈ 145 µs today (glue-bound, batch-invariant); s/node_bookkeeping ≈ 90 µs (champion: 972 nodes/move × 94 µs ≈ 91 ms/move, matches 278 games/hr at ~300 moves/game). moves/game ≈ 240–400 for a strong (escalation-reaching) agent.

Round cadence targets: 2000-game round overnight (~10 h) = **200 games/hr**; a 4-hour round = **500 games/hr**. The prior campaign's 15–90 games/hr ⇒ 22–133 h/round ⇒ starvation (206/2000). **Confirmed: the starving multiplier was `evals/move × s/eval` for a net beam** — hundreds of evals/move × 145 µs glue-bound forward.

Projection for the **recommended Gumbel-n64 operator** (T02): ~64–128 net-evals/move (≈10× fewer than w8d5-beam, comparable node-bookkeeping to a shallow beam). At the current glue-bound 145 µs/eval it is **marginal** (~150–300 games/hr, near the overnight edge); with the forward fixed (implicit conv or ANE at ~10–20 µs/eval, **to be verified**) it clears **~300–600 games/hr** locally — a real overnight-round cadence — and scales linearly on cloud. So the operator choice is throughput-justified, not only signal-justified.

## D. Design floor (the T03 answer)

1. **Fix the glue-bound forward BEFORE any campaign** — it is a datagen *prerequisite*, not an optimization; ~99% of the 145 µs is avoidable im2col/transpose scalar work. Options: implicit/direct conv, a mature inference lib (candle/tract/ort per the check-existing-tools rule), or ANE offload. **Re-verify the ANE 34k–150k evals/s claim** — memory was wrong about the BLAS number (claimed 20.5k; measured 6.9k), so the ANE number is also suspect until re-measured on *this* net.
2. **Keep per-move eval count low** — the Gumbel-n64 operator is chosen partly for this; deep wide beams (w8d5+, w128) are datagen-infeasible for a net leaf on the Mac.
3. **Architect actor/learner for linear scale-out** — parallel games per process (rayon already gives ~8×), containerized worker = the same actor at n=1 locally and n=many on cloud; per-game shard I/O, never per-node (T09).
4. **Local floor to hit before authorizing a campaign:** ≥200 games/hr sustained for a strong-ish agent (the THROUGHPUT STOP in the T02 kill criteria; the prior campaign never checked this and starved).
