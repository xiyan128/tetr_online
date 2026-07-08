---
id: T08
title: Search vehicle + learning-loop design freeze
labels: [wayfinder:grilling]
status: closed
assignee: fable-lead
blocked-by: [T01, T02, T03, T11, T07]
---

## Question

The keystone decision — freeze the leapfrog system design:

- **Search vehicle** for both self-play and the deployed bot: Gumbel-MCTS with chance nodes / policy-guided beam / hybrid — constrained by the purity checklist and priced by the throughput model.
- **Training targets**: visit counts vs completed-Q vs beam root scores (the near-deterministic-root STOP applies); value target = terminal z only, or n-step?
- **Value backup**: pure V(s) + terminal z (eliminating Z_SCALE/W/λ_att composition entirely — attack's instrumental value learned, not composed) vs reward composition during a transition period.
- **Opponent representation**: two-board joint value (the untried axis — the v1-vs-v2 ablation never ran); who wires `set_opponent` per decision in datagen and duels.
- **Improvement loop**: expert iteration cadence, replay buffer policy, warm start from round-0 or tabula-rasa-with-rain.

Output: the frozen system design (the leapfrog's STAGE1-DESIGN equivalent, written at the weight the rigor contract prescribes) — after this, implementation and round-1 tickets get charted.

## Resolution

Ratified + folded into the frozen design 2026-07-08: [T08 design freeze](../assets/T08-design-freeze-PROPOSED.md) (§RATIFIED). User delegated all four flagged decisions; explicit steer = balanced rigor/velocity, invest in research-velocity infra before grinding rounds.
