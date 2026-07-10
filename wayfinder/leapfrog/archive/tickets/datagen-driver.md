---
id: T14
title: Self-play datagen driver (writes shards + root scores)
labels: [wayfinder:prototype]
status: open
assignee:
blocked-by: [T27, T28]
---

## Question

The campaign's data plant — none exists on master (only the shard format). Build a driver that plays net-guided self-play under the venue and writes decision shards with per-child root scores (the completed-Q source). Reuse: the gate0a `capture_near_death` pattern (engine loop + `versus_step_piece`), the beam v0 operator (top-m policy roots + `root_scores`), `ShardWriter` (`shards.rs`), the seed-region allocator.

Correctness-first (small scale, verify shards round-trip + root scores align with `hold_placements`), then throughput via T13 (ANE fusion) — the ≥200 games/hr floor is the acceptance gate. Actor design must be the same worker at n=1 locally and n=many for cloud (T09). Resumable (durable = shards ∩ sidecar, byte-identical regen — the proven pattern); atomic writes; ShardWriter resume numbering (a known rl-branch bug not to re-inherit).

## Resolution (v0 BUILT + VERIFIED, 2026-07-08)

`crates/tetr-research/src/datagen.rs`. Plays mirror self-play driving the `BeamPlanner` directly (reads `root_scores`), applies the argmax placement via `placement_to_inputs` + a replay controller through the shared `versus_step_piece` (versus rules stay the engine's), captures each decision (served children obs + per-root Q) → `DecisionRecord` → `ShardWriter` with z backfilled at game end.

**Verified end-to-end** (`datagen_writes_shards` test): 4 CC2-beam games → 727 decisions, shards checksum-round-trip, z ∈ {−1,+1} labels correct, games run ~90 plies (placement replay is faithful — no desync). Key subtlety solved: a fresh engine has no active piece until stepped, and maneuver frames carry dt=0, so `advance_to_active` steps idle (dt=1/60) frames to spawn before planning (mirrors the controller's `neutral()`).

**Throughput** (`datagen_throughput` test): CC2 w8d5 full venue = **5,392 games/hr single-thread** (515 dec/s) — driver overhead is negligible. So the round-0 BC corpus (CC2 teacher) generates ~free (×cores ≈ 40k/hr); **net self-play datagen is bottlenecked purely by the forward (T13)**, confirming the plan.

**Same code, two uses:** CC2 eval = round-0 BC corpus; net eval = round-1+ self-play.

**v1 deferred (noted):** ε-sampling (v0 is argmax → played==argmax); resume (durable = shards ∩ sidecar, byte-identical regen); rayon parallelization across games; the `set_opponent` two-board path (v0 is opponent-blind, matching the net arms).

## CLI wired (2026-07-08)

`tetr-research datagen --width W --depth D --games N --seeds BASE --out DIR [--net <model-dir>]` — reproducible instrument. Verified: `datagen --width 8 --depth 5 --games 40 --seeds 100000` → 40 CC2 games in 22s (6494 games/hr single-thread), balanced 20-20 mirror, 10 shards (268 MB — the store-what-you-serve cost; ~13 GB for a 2000-game corpus, compression is a v1 concern). No `--net` = CC2 round-0 BC corpus; `--net dir` = net self-play. Gate-clean (fmt + clippy). T14 v0 COMPLETE.

## Validity reset — 2026-07-09

The v0 throughput result remains useful, but T14 is open: its rows lack schema
v2 provenance and frozen targets, `OppCtx` is neutral, grounded games couple
net seat to opener, both actors remain learnable, resume is not manifest-safe,
and caller-owned seeds overlap other purposes. T27/T28 supply the repair.
