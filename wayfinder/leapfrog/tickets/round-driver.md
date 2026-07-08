---
id: T16
title: Round driver — one resumable command per round
labels: [wayfinder:prototype]
status: open
assignee:
blocked-by: [T14, T15]
---

## Question

Port the `rounds.py` discipline to the frozen design: one resumable command per round (datagen → train → completed-Q transform → start-gates → promotion pair-GSPRT → ledger row), manifest memoization = rerun-is-resume. Honor the ratified BALANCED rigor: freeze the gate + kill criteria only; architecture floats with a one-line amendment log (not a per-deviation contract). Round ABORTS (not slips) on the throughput STOP (<150 games/hr first hour) or start-gate failure. Incumbent = last PROMOTE row. Not-inherit the rl-branch infra bugs (SPRT verdict latch — already fixed in master's `sprt.rs`; seed-region plumbing; non-atomic shard writes).

## Round-1 amendment log (balanced-rigor contract; written before running)

- **A-r1-1:** round-1 datagen = 100% mirror self-play (no champion-pinning yet — needs two-arm datagen; round-2 item). Seed region 2,000,000+ (disjoint: 100k dev / 1M round-0 / 8xxM duels / 900M throughput probes).
- **A-r1-2:** the round-1 incumbent/datagen driver is round0_v3 @ epoch 1 (training interrupted at 2/3 epochs; slot head learning healthily, sCE 21.8→4.24; policy ≈ v2's epoch-1 level). Acceptable because the round retrains from scratch on new data; the incumbent only drives datagen + gates.
- **A-r1-3:** round-1 training mixes round-0 + round-1 corpora via a symlink dir (replay, ~orig design's spirit) and uses the live-logit reanalyze targets.
- **Round-1 promotion read (pre-registered):** `gate --a guided:round1@m12w8d5 --b guided:round0_v3@m12w8d5` (p1=0.55 latched) + G_π preflight for round1. Promote only on gate PASS.
