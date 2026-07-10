---
id: T28
title: Make manifests authoritative and resume safe
labels: [wayfinder:task]
status: open
assignee:
blocked-by: [T22, T23, T27]
---

## Question

Replace the round driver's existence-based skipping, mutable symlink mixes, and
ad-hoc receipts with content-addressed step manifests. Datagen, corpus assembly,
training, export, diagnostics, anchor, and promotion must each declare complete
inputs, outputs, hashes, validity state, and predecessor manifest. Resume may
reuse a step only when that contract matches exactly.

Validity-bearing work must run a matching clean binary/tree and immutable model
artifacts; `--allow-dirty`, a stale binary, an empty event stream, or a partial
directory can be exploratory but can never promote.

## Acceptance evidence

- Changing any code/build/model/arm/seed/venue/search/training/dataset input
  invalidates the affected step and all descendants instead of silently skipping.
- Corpus mixes are immutable lists of schema-v2 shard hashes; dangling or
  retargeted symlinks cannot change a completed run.
- Atomic manifests distinguish planned/running/complete/failed/exploratory and
  validate every expected artifact, count, checksum, and raw event stream.
- Crash injection after every round stage, followed by resume, yields the same
  accepted inputs/results as an uninterrupted fixture and exactly one ledger row.
- A manifest verifier rejects stale binaries, dirty validity runs, missing raw
  pairs, incomplete datasets, model-hash drift, and invalid evidence ancestry.
