---
id: T10
title: Single-player from the same system
labels: [wayfinder:grilling]
status: open
assignee:
blocked-by: [T04, T08]
---

## Question

The destination demands the best bot in BOTH settings from one learning system. Decide how solo emerges:

- **Multi-task training** (solo marathon/downstack as additional venues with their own returns), **conditioning** (venue features already exist in obs — extend?), or **post-hoc finetune** of the versus net?
- What is the solo objective without hand-tuned reward — survival + real score? Attack? (Solo APP is combo-farmable and gate-only; the E8 confound says attack-only values are suicidal under rain.)
- Does the solo bar (champion 0.8225 APP held-out, downstack battery) need its own search budget rules?

Output: the solo training + gating plan, consistent with the frozen design and the gate battery.

## First measured solo baseline (2026-07-09): the versus vehicle scores ZERO

`solo` subcommand added (marathon-holdout convention: 16 VALIDATION seeds, cap 150). Reads:
- **`guided:round0_v3@m12w8d5`: mean APP 0.000, topped 16/16** — the versus-dominant vehicle (24-8 over its CC2 teacher) stacks itself to death on an empty board with no pressure. Versus training does not transfer to solo AT ALL (the policy learned pressure-conditioned patterns that self-destruct without rain/opponent).
- `value:round0_v3` (d1): APP 0.225, topped 7/16 — the value head alone is less degenerate than the full vehicle.
- Champion reference on the same battery: **0.787** (tp:cc2@w128d9, 0 topouts).

T10 is a genuine second training axis, not a freebie: candidate designs (multi-task solo datagen venue + mixed training; solo z definition — survival/score; venue-conditioned obs) now have a concrete 0.0 baseline to beat.
