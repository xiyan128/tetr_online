---
id: T25
title: Specify the purity-compliant generic search
labels: [wayfinder:research]
status: open
assignee:
blocked-by: [T18]
---

## Question

Define the exact search semantics that a fully learned final bot and its
datagen will share. Reconcile the repaired completed-Q/Gumbel reference with
the true tetr-core model: decision selection, root allocation, non-root policy,
terminal WDL backup, observable bag/garbage chance treatment, transpositions,
tie-breaking, and training-only versus inference randomness.

The final decision path may contain learned policy/value, environment rules,
and compute-budget knobs only. Any risk transform or pruning rule must be
mathematically specified, purity-justified, and ablated; attack composition,
CC2 evaluation, `SPEC_DECAY`, and hand-scored move ordering are forbidden.

## Acceptance evidence

- A linked research memo cites the reference implementations/papers and gives
  pseudocode plus invariants for every decision and chance backup.
- Tiny deterministic and stochastic game trees have hand-computed oracle Q,
  improved-policy, and chosen-action vectors suitable for differential tests.
- The spec distinguishes environment truth, learned quantities, compute knobs,
  and forbidden heuristics component by component.
- It states one canonical vehicle identity for datagen, duel/gate, solo, and
  deployment, with no content-dependent ranker dispatch.
- A red-team section identifies falsification conditions, numerical edge cases,
  and any remaining implementation decision; none may be deferred to training.
