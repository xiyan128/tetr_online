---
id: T29
title: Put the cross-language learning contract in CI
labels: [wayfinder:task]
status: open
assignee:
blocked-by: [T18, T23, T24, T26, T28]
---

## Question

Make the repaired learning/data/serving contract part of the repository's real
quality gate. CI and `scripts/gate` must run the same Python invariant suite and
a tiny Rust datagen -> Python train/export -> Rust load/pure-play round trip,
rather than allowing standalone self-checks or manually run goldens to carry
load-bearing claims.

Keep the fixture small enough for every push while covering completed-Q,
schema-v2 provenance, CRN grouping, preprocessing, all learned heads,
opponent/venue context, pure search, model identity, and accelerated-export
parity where the platform supports it.

## Acceptance evidence

- A locked Python environment runs pytest in both `scripts/gate` and GitHub CI;
  either side fails if the other omits a load-bearing command.
- The tiny end-to-end fixture generates a manifest-valid corpus, trains/exports,
  loads in Rust, and plays legal identical pure-vehicle decisions on shared
  goldens.
- Negative fixtures prove CI catches completed-Q collapse, row/score
  misalignment, pair leakage, forbidden CC2 rows, constant-feature whitening,
  model-hash drift, and missing policy/value/slot outputs.
- Rust/Python/ONNX-or-CoreML goldens include non-neutral opponent/venue inputs
  and pass at documented tolerances.
- A clean local `scripts/gate` receipt and the corresponding CI job are linked
  from the resolution.
