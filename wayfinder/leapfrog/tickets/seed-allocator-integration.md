---
id: T21
title: Put leapfrog on the seed-domain allocator
labels: [wayfinder:task]
status: open
assignee:
blocked-by: [T19]
---

## Question

Replace caller-owned arithmetic seed bases with the repository's
`seeds::Campaign` discipline throughout datagen, diagnostics, anchors,
promotion gates, solo development/validation, and final confirmation. Allocation
must be by campaign, round/event, and purpose; the round driver may not invent
numeric offsets. Preserve a stable pair/game identity without truncating a
64-bit seed into the current `u32` game id.

Final-confirmation seeds need a recorded one-time claim opt-in and may never be
read by training, tuning, rehearsal, or a second modified candidate.

## Acceptance evidence

- A machine-readable allocation manifest records campaign id/slot, purpose,
  index interval, realized-seed hash, and consumer artifact.
- An overlap verifier proves all configured train, duel, anchor, promotion,
  solo, and confirmation allocations are pairwise disjoint across the supported
  round/event range; it catches the current round-n gate/round-(n+1) duel overlap.
- CLI and Python tests reject raw unregistered bases in validity-bearing runs.
- Pair/game ids round-trip at full width and remain unique across workers and
  datasets.
- Final-region access without a fresh recorded claim token fails loudly and is
  covered by tests.
