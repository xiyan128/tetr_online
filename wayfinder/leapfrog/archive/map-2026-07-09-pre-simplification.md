---
title: Leapfrog strike — SOTA bot from a fully learned system
labels: [wayfinder:map]
created: 2026-07-07
updated: 2026-07-09
---

## Destination

A bot whose deployed decisions are entirely learned: learned policy and learned
terminal-WDL value drive one generic search, with no CC2 evaluator, hand-tuned
board/attack term, `SPEC_DECAY`, or hidden vehicle selection. At a matched
champion-comparable per-move wall-clock it must beat the registered
`probe-tp128d9` under a frozen pair-GSPRT and match or beat the same champion on
the frozen single-player battery. The same system must run small experiments on
the Mac and scale provider-neutrally without redesign. CC2 action supervision
is permitted only for round 0; later CC2 use is opponent/yardstick only.

## Current truth — validity reset

- **No accepted incumbent or clean lineage exists.** Round-0 v2-v4 and rounds
  1-11 are forensic artifacts, not promotion evidence. Rounds 1-10 are void;
  round 11 was already training when the reset was declared and may finish as
  an exploratory diagnostic only.
- New `tetrnn.round`, `tetrnn.train`, and slot-fitting invocations fail closed.
  The pause is removed only after their replacement contracts are executable.
- The old registered-arm shorthand resolves the wrong CC2 weight profile;
  Gate-0a mixed generator/scorer identities; Gate-0b did not run the specified
  operator/curve/CI; the final gate was not fully frozen.
- Legacy guided shards omit discarded actions and frozen generator logits,
  reuse an out-of-scale `-1e8` terminal sentinel, and cannot reconstruct
  completed-Q. The old `c=12` target fails its own adversarial test.
- The deployed net path is not pure: hand attack composition and
  `SPEC_DECAY` remain, while production versus/datagen paths keep opponent and
  venue context neutral.
- Gate/round evidence is not reproducible enough for a claim: seed domains
  overlap, parallel gates consume completion order, event streams are empty,
  resume is existence-based, mixes are mutable symlinks, and artifacts live in
  ephemeral `/private/tmp` paths from dirty builds.

## Decisions that remain valid

- [T00 — Destination](tickets/destination.md): the bar is the registered
  in-repo champion in versus plus single-player, under practical compute, from
  one fully learned system.
- [T01 — Purity contract](tickets/purity-contract.md): environment rules and
  compute budgets are allowed; CC2 evaluation, hand reward/value composition,
  and speculative preference heuristics are forbidden in the deployed path.
- [T02 — Algorithm survey](tickets/algorithm-survey.md): Gumbel-style policy
  improvement on the true model remains a research hypothesis, not an
  authorized implementation or evidence of strength.
- [T03 — Throughput model](tickets/throughput-budget.md): local experimentation
  needs a real-vehicle floor near 200 games/hour; throughput must be measured on
  realistic game lengths and the actual pure vehicle.
- [T06 — Rigor/velocity contract](tickets/rigor-contract.md): freeze validity
  and promotion rules, keep one-variable research levers explicit, and record
  negative results without relabeling them as progress.
- [T17 — Spec-dedup](tickets/port-spec-dedup.md): verified already present on
  master; no port was required.
- [T18 — Completed-Q repair](tickets/repair-completedq-contract.md): reference
  Mctx semantics are separated from legacy rank distillation; frozen logits,
  legal/invalid separation, finite-scale terminal Q, pytest/CI, and fail-closed
  legacy training are now the load-bearing contract.

## Superseded or reopened decisions

- [T08 — old design freeze](tickets/design-freeze.md) is historical and
  superseded by T30.
- [T12 — old operator](tickets/gumbel-operator.md) is historical/retracted;
  T25/T26 replace it.
- [T15 — old completed-Q training](tickets/completedq-training.md) is
  historical/invalidated; its “round-1 authorized” conclusion is withdrawn.
- [T04 — final gate](tickets/gate-battery.md), [T07 — Gate-0b](tickets/search-gain-verification.md),
  and [T11 — Gate-0a](tickets/gate0a-survival-recall.md) are reopened.
- T05, T09, T10, T13, T14, and T16 remain open under repaired blockers.

## Execution DAG before another campaign round

Current frontier:

1. Resolve [T19 — evidence quarantine](tickets/evidence-quarantine.md).
2. After T19, run [T20 — immutable arm identity](tickets/immutable-arm-identity.md)
   and [T21 — seed allocator integration](tickets/seed-allocator-integration.md)
   in parallel.
3. T20 + T21 unlock [T22 — deterministic-prefix instruments](tickets/deterministic-prefix-instruments.md)
   and [T23 — provenance shard v2](tickets/provenance-shard-v2.md).
4. T18 unlocks [T25 — pure search specification](tickets/pure-search-spec.md);
   T20 + T23 unlock [T24 — opponent/venue context](tickets/opponent-venue-context.md).
5. T24 + T25 unlock [T26 — pure learned search](tickets/pure-learned-search.md).
6. T20 + T21 + T23 + T24 + T26 unlock
   [T27 — counterbalanced datagen](tickets/counterbalanced-datagen-repair.md).
7. T22 + T23 + T27 unlock [T28 — authoritative manifests/resume](tickets/authoritative-manifests-resume.md).
8. T18 + T23 + T24 + T26 + T28 unlock
   [T29 — cross-language learning CI](tickets/cross-language-learning-ci.md).
9. T18 + T27 + T29 unlock the clean [T05 — round-0 net](tickets/round0-net.md).
10. With the clean stack, rerun T04/T07/T11 and finish T13 on the actual pure
    vehicle. Their evidence unlocks
    [T30 — validity-restored design freeze](tickets/validity-restored-design-freeze.md).
11. Close repaired T14 and certify [T16 — one-command round driver](tickets/round-driver.md)
    twice, including crash/resume. Only then may a clean round-1 ticket exist.

Every ticket closes on evidence, including a valid negative result. No ticket
may infer that a downstream experiment will pass.

## After the clean driver is certified

Chart one task per clean expert-iteration round. A candidate advances only on
the frozen deterministic incumbent gate, registered-champion no-regression,
purity, provenance, start-gate, and solo-smoke requirements. After three valid
rounds, audit compounding before choosing a fixed wall-clock capacity/search
point or scaling out the provider-neutral actor/artifact store.

Only a validation-qualified, statically/runtime-audited pure artifact may be
frozen for untouched confirmation. Final versus and solo confirmation use the
same model/config hash. A failed confirmation remains a valid negative result;
any modified candidate receives a new claim id and fresh confirmation region.

## Not yet specified

- Whether the pure completed-Q/Gumbel operator passes the repaired Gate-0b.
- Which network capacity/search budget is optimal at matched wall-clock.
- How many clean rounds are needed, if compounding exists at all.
- The final solo multitask mixture and venue conditioning, to be resolved in
  T10 only after a valid versus loop exists.
- Cloud provider and scale, intentionally deferred behind the provider-neutral
  worker/artifact contract in T09.

## Out of scope

- Shipping the bot in the browser/game UI; the first destination is a native
  research bot with defensible evidence.
- Treating an external CC2 release as the claim bar; the bar is the immutable
  in-repo `probe-tp128d9` champion.
- Game/UI features or broad TETR.IO fidelity unrelated to the frozen venues.
