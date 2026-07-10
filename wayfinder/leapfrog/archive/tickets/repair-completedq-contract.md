---
id: T18
title: Repair the completed-Q improvement contract
labels: [wayfinder:task]
status: closed
assignee:
blocked-by: []
---

## Question

Repair the campaign's load-bearing policy-improvement transform before another
candidate can promote:

1. Reconcile `python/tetrnn/targets.py` with the original Gumbel MuZero / mctx
   completed-Q semantics rather than relying on an unverified fixed scale.
2. Demonstrate non-collapse on adversarial score shapes and characterize the
   real round-0 and filtered top-m corpus distributions (entropy, top-1 mass,
   effective support, ties, and dead-root behavior).
3. Define an acceptance envelope that preserves genuine search preference
   without recreating the near-one-hot target failure this design exists to
   avoid.
4. Put every load-bearing invariant in pytest and in repository CI; the
   standalone `python -m tetrnn.targets` checks are not sufficient.
5. Re-run the targeted Python and Rust/Python contract checks and record the
   evidence here.

Round-11 may complete as an exploratory diagnostic, but no round using the
broken transform can authorize promotion. This ticket blocks the clean
campaign restart.

## Resolution

Resolved 2026-07-09 as a **validity repair, not a retroactive rescue of the
legacy campaign**.

### What the sources actually authorize

The original Gumbel MuZero construction completes each unvisited legal action
with an estimate of the generator policy's state value, transforms completed
Q, and forms `softmax(frozen_generator_logits + transformed_Q)`. Mctx's
reference implementation uses the prior-weighted mixed-value estimator,
min-max rescaling with an `1e-8` denominator floor, and default scale
`(50 + max(visits)) * 0.1`.

Primary sources:

- [Gumbel MuZero, equations 8/10/11 and appendices C-D](https://openreview.net/pdf?id=bERaNdoegnO)
- [Mctx completed-Q implementation, pinned commit](https://github.com/google-deepmind/mctx/blob/450fbf7656b88dd1d8ca5b2db3a2f9464cb322f2/mctx/_src/qtransforms.py#L88-L194)
- [Mctx target construction](https://github.com/google-deepmind/mctx/blob/450fbf7656b88dd1d8ca5b2db3a2f9464cb322f2/mctx/_src/policies.py#L211-L231)

The theorem does **not** promise an entropy floor, and it assumes correct Q and
exact completion value. Tetr's `root_best` is a max-backed-up heuristic score,
so legacy use is now named rank distillation rather than policy improvement.

### Defect and repair

- The old `c=12` self-test was mathematically impossible: on its 30-way
  one-best vector, top mass is `0.99982185` and `N_eff=1.00232`.
- `python/tetrnn/targets.py` now contains a direct NumPy port of Mctx completion
  and target semantics, including mixed value, visit scaling, epsilon behavior,
  prior logits, and an invalid-action mask separate from legal terminal loss.
- The legacy all-roots-observed fallback is separately named
  `rank_distillation_target`; frozen generator logits are mandatory. It starts
  at `beta=5`, Mctx's zero-visit default scale, because inventing visit counts
  for beam max-backups would be false precision.
- Equal Q and zero visits preserve the frozen generator prior. A legal terminal
  loss must carry a finite Q on the evaluator's ordinary scale; only explicit
  invalid actions get zero mass. Microscopic score ranges are no longer
  inflated to `[0,1]`.
- Trainee-live logits are removed. They have no finite fixed point when Q is
  nonconstant: the detached target repeatedly multiplies preferred odds by
  `exp(beta * delta_Q)`.
- The legacy `-1e8` immediate-death sentinel is not a finite-scale Q. On a
  partial-terminal group it dominates global min-max and makes every surviving
  move effectively tied. Such groups are now rejected rather than mislabeled.
- The exact fixed-vocabulary Mctx port is retained for parity. The clean Tetr
  wrapper removes explicit invalid padding before transformation and scatters
  zeros afterward, so adding an invalid slot cannot change legal-action odds;
  invalid actions with positive visits are rejected.
- The current top-m shards omit discarded legal roots and do not store frozen
  generator logits, visits, raw root value, finite-scale terminal Q, or
  validity/provenance. The trainer and slot fitter therefore fail closed with
  **no legacy model-producing escape**. The round driver fails before datagen
  under the campaign validity reset. The already-loaded round-11 process does
  not see that guard and may still write a legacy verdict; it is classified
  exploratory here, while T19 owns the mechanical quarantine that prevents
  selection as incumbent/lineage or promotion evidence.

### Corpus evidence

A read-only pass covered **887,344 decisions**: round-0 full roots (579,406),
round-11 CC2 full roots (55,826), round-11 CC2 net/top-12 roots (55,811), and
round-11 mirror/top-12 roots (140,475).

At the old `c=12`, top-12 targets had median `N_eff=2.44`, about **52%** fell
below 2.5, and about **14%** put at least 0.9 on one action. Full-root targets
looked much softer, proving the pooled calibration did not transfer across
fanout. Re-truncating 50,000 full-root decisions to 12 and recomputing min/max
reproduced the collapse; score units were not the cause.

The first all-roots-legal pass then exposed the terminal-sentinel scale defect:
on the 4.78% of round-0 and 4.31% of r11-full partial-terminal decisions, the
median transformed span among surviving moves was below `0.0006`. Those rows
are now a hard rejection. On the remaining rows, the `beta=5` diagnostic is:

| source | retained / total | N_eff p10 / median / p90 | top-1 median / p90 | top-1 >= .9 | N_eff < 2.5 |
|---|---:|---:|---:|---:|---:|
| round-0 full | 551,734 / 579,406 | 17.99 / 31.57 / 46.15 | .131 / .245 | 0% | .0007% |
| r11 CC2 full | 53,419 / 55,826 | 19.19 / 32.93 / 45.97 | .129 / .236 | 0% | 0% |
| r11 CC2 top-12 | 55,811 / 55,811 | 3.37 / 5.04 / 6.79 | .365 / .581 | .512% | 2.23% |
| r11 mirror top-12 | 140,475 / 140,475 | 3.37 / 5.09 / 6.83 | .364 / .585 | .456% | 2.14% |

This telemetry is descriptive only: incomplete top-m shards remain rejected,
and the full-root exclusions cannot be silently dropped during training.
The executable adversarial envelope pins the 30-way isolated-best case to
top mass `0.83654` and `N_eff=2.70684`, plus monotone expected-score improvement
for the exact all-actions-observed tilt. No universal support claim is made.

### Executable evidence

`python/tests/test_targets.py` puts 22 target and fail-closed cases under
pytest: the official Mctx policy fixture, mixed-value formula, zero visits,
equal Q, legal-terminal vs invalid masking, invalid-padding invariance,
partial/all-terminal legacy sentinels, above-epsilon affine invariance,
pairwise odds, randomized expected-score
monotonicity, epsilon behavior, deterministic bytes, malformed inputs,
live-logit rejection, and round-driver pause. The reproducible
`python -m tetrnn.target_audit` command recursively audits parallel-worker
corpora, emits deterministic JSON, and has six more tests covering seat filters,
sentinel rejection, empty selections, and worker traversal. GitHub CI now
installs `uv` from a pinned action, syncs `python/uv.lock`, and runs Ruff,
Pyright, and pytest on every push.

Verification receipts:

```text
uv run --frozen python -m tetrnn.targets
  reference parity and rank-target smoke checks pass
uv run --frozen ruff check . && uv run --frozen ruff format --check .
  green
uv run --frozen pyright
  0 errors, 0 warnings, 0 informations
uv run --frozen pytest
  31 passed
cargo test -p tetr-nn --locked
  16 passed, 2 ignored; Python/Rust goldens pass
cargo test -p tetr-research --locked datagen_writes_shards -- --test-threads=1
  1 passed
```

The four-corpus telemetry table above is reproducible with:

```text
uv run --frozen python -m tetrnn.target_audit \
  r0=<scratch>/round0_full_v2 \
  r11_cc2_net=<scratch>/r11/corpus/cc2:net-seat \
  r11_cc2_teacher=<scratch>/r11/corpus/cc2:teacher-seat \
  r11_mirror=<scratch>/r11/corpus/mirror
```

The next enabling step is the provenance-rich shard-v2 ticket. Until that
closes, there is no honest learned-generator completed-Q training path.
