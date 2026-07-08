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

## Round-1 execution log

- Datagen: serial run measured ~160 games/hr real (the 4-game probe's 1,370/hr was a short-game artifact — the T03 game-length confound again); **built `--workers` parallel fan-out (committed)** → **1,900 games in 22.8 min = 5,000 games/hr (6 workers)**. Round-1 corpus = 17 serial + 66 parallel shards, ~83k decisions (round-1 mirror games are SHORT, ~72 plies vs round-0's ~292 — fast decisive kills under rain).
- **Seat-A skew found**: 1059-841 (~5σ) in fixed-opener mirror games — z-label noise (obs don't encode seat). Fixed for future rounds (opener staggered by game parity, committed); round-1 data kept (noise, not artifact).
- Training: live-logit reanalyze mode on the 203-shard replay mix (round-1 + every-4th round-0 shard), 3 epochs, running. N_eff read on the first (round-0) shard = 6.68, marginally above band — the live form sharpens with logits; watch epoch metrics.
- Next: G_π preflight (`duel guided:round1@m12w8d5 vs policy:round1`) → promotion gate (`gate guided:round1 vs guided:round0_v3`, p1=0.55 latched, fresh seeds 920M+).
