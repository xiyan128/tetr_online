---
id: T24
title: Wire opponent and venue context end to end
labels: [wayfinder:task]
status: open
assignee:
blocked-by: [T20, T23]
---

## Question

Make the promised two-board, venue-conditioned system real. At every versus
decision, freeze the live opposing engine snapshot plus rain/cap clock once and
feed that same `OppCtx` to parent policy ranking, every leaf evaluation,
datagen serialization, duel/gate serving, and deployment. Neutral
`OppCtx::default()` is valid only for an explicitly solo context, never as a
silent versus fallback.

Re-derive preprocessing so a feature that was constant in an old corpus cannot
explode to million-scale standardized values when opponent or venue context is
activated.

## Acceptance evidence

- A same-seed, same-ply probe shows harness and datagen serialize byte-identical
  parent/child observations and opponent context for the same decision.
- Versus-path tests fail if neutral opponent context or zeroed venue clocks are
  used; an explicit solo path remains valid.
- A generated fixture has nonzero opponent-board, opponent-pressure, rain, and
  cap variance, with values matching engine snapshots.
- Whitening is fitted over every served row type, records constant-feature
  handling, and bounds all transformed parent/child/versus features.
- Python, Rust BLAS, and accelerated backend goldens agree with non-neutral
  opponent and venue inputs.
