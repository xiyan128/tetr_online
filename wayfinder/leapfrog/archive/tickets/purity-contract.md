---
id: T01
title: Purity contract — classify every fixed component
labels: [wayfinder:grilling]
status: closed
assignee: fable-lead
blocked-by: []
---

## Question

"Fully learned deployed bot" needs an auditable checklist. Enumerate every fixed component in the current stack and classify each as: **environment truth** (allowed — attack tables, garbage rules, topout), **budget knob** (allowed — beam width/depth, MCTS simulation count, wall-clock), **training-time machinery** (allowed if principled — π-target maps, ε-schedules, replay mixing), or **hand-tuned eval/search heuristic** (forbidden in the deployed bot).

Known candidates to rule on: CC2 eval terms; `spec_weight`/SPEC_DECAY bag-speculation discount; the TP collapse rule; DEATH_SCORE sentinel; Z_SCALE / W / λ_att value-reward composition (does pure V-backup with terminal z eliminate composition entirely?); venue constants embedded in obs features 83..85; hold-replan behavior; movegen itself.

Output: the purity checklist the final showdown bot must pass, ratified by the user. This checklist is an input to the search-vehicle design freeze.

## Resolution (proposed default, ratifiable)

Classified every fixed component into four categories — environment truth / compute-budget knobs / training-time machinery (all ALLOWED) vs hand-tuned eval-or-search-heuristic (FORBIDDEN in the deployed bot). Full checklist: [T01 purity contract](../assets/T01-purity-contract.md). Most rulings follow directly from the destination answer ("fully learned deployed bot") or a defensible default.

**Headline:** the deployed showdown bot must contain zero numbers from the FORBIDDEN row (no CC2 eval terms, no SPEC_DECAY, no Z_SCALE/W/λ composition — terminal-WDL value eliminates composition by design) — verifiable by inspecting the deployed arm. CC2 may appear only as the opponent in the race harness.

**One flagged judgment call (default = allowed, user may veto):** hand-chosen search BUDGET (beam width / MCTS m,n / depth / wall-clock) is treated as an allowed *compute-budget knob*, not a forbidden fixed component — matching AlphaZero practice and the "practical per-move budget" destination framing. If vetoed (budget must itself be learned), the system is materially harder.

Design freeze (T08) proceeds on this contract unless the flagged call is vetoed.
