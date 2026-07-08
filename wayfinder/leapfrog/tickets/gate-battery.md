---
id: T04
title: SOTA gate battery — pre-register the final showdown
labels: [wayfinder:grilling]
status: closed
assignee:
blocked-by: []
---

## Question

Define exactly what "dethrone the champion" means, frozen before any training:

- **Venue(s)**: is sudden-death rain-8/cap-240 THE versus venue for the claim? Obs features 83..85 condition nets on the venue — does that bind or bias the claim? Escalation attenuation (~0.52 skill expression) was measured once on the CC2 ladder — does it need a fresh read?
- **Budget matching**: matched per-move wall-clock at ~100 ms — on what hardware? Is ANE-vs-CPU fair when the champion is CPU-bound beam bookkeeping? (Precedent: the decisive conv_rb1 race accepted ANE.)
- **Champion fairness**: does the rl-branch spec-dedup (1.32× CPU, bit-identical) land on master first so the champion races at its best?
- **Statistics**: pair-GSPRT config (p1, α, max pairs — default gate is p1=0.55/400 pairs; is that sensitive enough for the final claim?), seed-region discipline, end-reason stratification.
- **Single-player battery**: marathon-holdout APP (champion 0.8225), downstack, censored metrics; anti-gaming rules (solo APP is combo-farmable — never a sole verdict metric).

Output: a frozen gate contract (linked asset) that the campaign cannot move after round 1 starts.

## Resolution

Ratified + folded into the frozen design 2026-07-08: [T08 design freeze](../assets/T08-design-freeze-PROPOSED.md) (§RATIFIED). User delegated all four flagged decisions; explicit steer = balanced rigor/velocity, invest in research-velocity infra before grinding rounds.
