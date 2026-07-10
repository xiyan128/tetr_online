---
id: T23
title: Define provenance-rich decision shards v2
labels: [wayfinder:task]
status: open
assignee:
blocked-by: [T20, T21]
---

## Question

Replace filename-derived generator identity with an explicit versioned shard
contract. Every decision must say which CRN pair/game, actor, model, search,
vehicle, venue, score semantics, and build produced it, and whether that row is
eligible to train. The schema must support full-width ids and group train/holdout
splits by CRN pair, not by whichever shard a worker happened to flush.

Legality/terminality must be structural data, separate from Q. Every legal
terminal child stores its terminal WDL Q on the evaluator's declared finite
scale; an invalid action is masked explicitly. The legacy `-1e8` death/invalid
sentinel may not share `child_q` with ordinary finite values, because it destroys
the resolution of min-max/completed-Q transforms.

After round 0, CC2 may be an opponent/yardstick but its decisions and the
round-0 demonstration corpus may not silently become continuing teacher data.

## Acceptance evidence

- Schema-v2 documentation and Rust/Python readers cover pair/game/seat/opener,
  seed allocation, actor kind, learnable flag, canonical arm/model/search/build
  hashes, venue, score-unit/backup semantics, explicit validity mask, terminal
  flag, and finite terminal WDL Q.
- Shards carry payload checksums and belong to an atomic dataset manifest with
  expected/completed games, pair coverage, row counts, and shard hashes.
- Rust/Python golden fixtures round-trip every field, including ids above
  `u32::MAX`, mixed actors, draws, legal terminal roots, invalid roots, and
  all-terminal/all-invalid decisions.
- Split tests prove the two games of a CRN pair and all of one game remain in one
  partition.
- The trainer rejects unknown schemas/identities, mixed absolute score units,
  nonfinite or legacy sentinel Q values, post-round-0 CC2 learnable rows, and
  post-round-0 round-0 replay. A regression test proves adding an invalid root
  cannot change the completed-Q resolution among legal finite roots.
