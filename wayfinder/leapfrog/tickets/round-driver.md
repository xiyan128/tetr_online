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
