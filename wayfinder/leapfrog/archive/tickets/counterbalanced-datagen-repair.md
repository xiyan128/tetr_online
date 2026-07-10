---
id: T27
title: Repair datagen with counterbalanced CRN pairs
labels: [wayfinder:task]
status: open
assignee:
blocked-by: [T20, T21, T23, T24, T26]
---

## Question

Rebuild the data plant around the pure vehicle and immutable arm identities.
Grounded games must be CRN pairs with arms swapped and the opener
counterbalanced independently of net seat; the current parity coupling that
makes the net always open must disappear. Mirror data must retain pair/game
grouping and balanced opener coverage.

Round 0 may record CC2 teacher decisions. Every later grounded game may use CC2
only as an opponent: only learned-agent decisions are training-eligible, while
the opponent's actions remain optional nonlearnable forensic data.

## Acceptance evidence

- For every grounded seed, each arm occupies each seat/opener condition exactly
  once; invariant tests fail on either parity coupling or missing swap.
- Outcome receipts stratify by actor, seat, opener, end reason, and game length;
  net strength is computed from actor identity rather than seat parity.
- Schema-v2 rows identify the actor and eligibility, and a post-round-0 corpus
  audit finds zero learnable CC2 decisions and zero round-0 replay rows.
- Seed-matched driver and harness traces agree ply-for-ply on placement,
  observation, attack routing, terminal result, and vehicle identity.
- Interrupted multiworker generation resumes/regenerates to a complete manifest
  without duplicate games, shard-number collisions, or partial-game rows.
