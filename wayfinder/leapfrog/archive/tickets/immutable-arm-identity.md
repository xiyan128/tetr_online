---
id: T20
title: Make every arm an immutable experiment identity
labels: [wayfinder:task]
status: open
assignee:
blocked-by: [T19]
---

## Question

Eliminate ambiguous arm strings and evaluator substitutions. A receipt must
identify the exact bot that ran: registered bot/spec, search class and budget,
CC2 weight profile, vehicle/ranker, inference backend, model-file hashes, and
purity class. The champion identity must resolve to the registry's
`probe-tp128d9` with `Cc2Weights::attack_tuned()`; default and attack-tuned CC2
must never share an identity.

Datagen, Gate-0, anchors, duels, and the final showdown must all construct bots
through this same canonical identity path.

## Acceptance evidence

- Canonical identity JSON and a stable identity hash round-trip through parsing,
  display, receipts, and resume manifests.
- Tests prove the final champion arm is byte-for-byte/spec-for-spec equal to the
  registered `probe-tp128d9`, including attack-tuned weights and TP w128d9.
- Default CC2, attack-tuned CC2, low-width anchors, and the champion have
  distinct names and hashes; ambiguous `cc2` parsing is rejected.
- Mutating a model, config, weight profile, backend, ranker, or search budget
  changes the identity hash and invalidates resume.
- `cargo test -p tetr-research arm_identity` and the registry identity tests pass.
