---
id: T26
title: Build the pure learned search vehicle
labels: [wayfinder:task]
status: open
assignee:
blocked-by: [T24, T25]
---

## Question

Implement the T25 search as the one explicit vehicle used by data generation,
research instruments, solo evaluation, and the final deployed arm. Learned
policy must allocate/select actions and learned terminal-WDL value must back up
their consequences; fixed search settings may control compute but may not say
which moves or states are good.

Remove the deployed path's attack reward composition and speculative discount
rather than hiding them behind a model export or runtime default.

## Acceptance evidence

- Differential tests match every T25 deterministic/chance oracle, including
  all-dead, tied, bag-boundary, terminal, and budget-truncated cases.
- The canonical pure arm's reachable decision graph contains no
  `Cc2Evaluator`, `attack_w`/`compose`, `SPEC_DECAY`, handcrafted eval term, or
  hidden vehicle chooser; an automated purity test enforces this.
- Missing or mutated policy and value heads change/refuse decisions, proving
  both learned heads are actually consumed.
- Datagen, duel, gate, solo, and deployment construct the same planner/evaluator
  identity and produce identical actions on shared fixtures.
- Rust/Python target extraction and BLAS/accelerated inference parity pass for
  the pure vehicle, and only explicit compute knobs alter its search budget.
