---
id: T00
title: Name the destination
labels: [wayfinder:grilling]
status: closed
assignee: fable-charting-session
blocked-by: []
---

## Question

What is the leapfrog strike finding its way to — the concrete SOTA bar, the purity contract, the compute envelope, and the deployment scope?

## Resolution

Grilled 2026-07-07 (four questions, user answered directly):

1. **SOTA bar = dethrone the champion.** The learned system must BEAT `probe-tp128d9` head-to-head in versus under a pre-registered SPRT AND match-or-beat it in single-player, at a practical per-move budget (champion-comparable think time, ~100 ms/move class).
2. **Purity = fully learned deployed bot.** Final bot = learned policy + learned value + a *generic* search guided only by the nets. No CC2 eval terms, no hand-tuned pruning/ordering/composition weights anywhere in the deployed bot. CC2 permitted only as round-0 warm start and as opponent/yardstick.
3. **Compute = scalable by design.** Small experiments run on the Mac; the architecture must scale to cloud without redesign; no cloud provider assumed.
4. **Deployment = research bot.** SOTA proven native (ANE/GPU allowed at inference). Shipping in-game is out of scope for this map.
