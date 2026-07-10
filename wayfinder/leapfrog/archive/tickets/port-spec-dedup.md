---
id: T17
title: Port spec-dedup to master (beam-bookkeeping velocity)
labels: [wayfinder:task]
status: closed
assignee:
blocked-by: []
---

## Question

Velocity infra (ratified infra-first). The rl branch's Stage-0 spec-dedup made the beam re-commit+re-evaluate each placement once per bag-legal piece past the visible queue (~75-80% duplicate work); the fix is one commit+eval per placement + per-piece spawn-block-out recheck, verified **bit-identical** (A/B identity + marathon APP reproduced exactly) at **1.32× CPU** on the rl branch. This speeds every champion race, gate, and net-beam datagen game — the yardstick that gates everything (champion races were the session's dominant cost: ~10 min for 24 games serially).

Port the rl-branch commit to master's `beam.rs`. **Acceptance = bit-identical APP golden** (0.8255555629730225 marathon reproduced exactly) at measurably lower CPU. First check the diff size (master's beam.rs may have diverged); if tangled, re-derive the dedup fresh against master with the same golden gate. Low risk, high verification — do it fresh with full attention (it touches core search).

## Resolution — already on master (verified, no port needed)

Verified 2026-07-08: the v2 landing's PR1 (`d151054` "core: leaf batch seam + speculative eval sharing in the beam") IS the spec-dedup re-derivation. `expand_speculative` (beam.rs) commits once per placement (the `Committed` struct: one `apply_placement` + one board-only eval), then fans the ≤7 bag continuations with cheap `base.clone()+deal_speculative` — commit AND eval shared, exactly the rl-branch d3beb4e optimization, landed with the bit-identical APP golden. No work needed; the champion already races at its best. Closed as no-op.
