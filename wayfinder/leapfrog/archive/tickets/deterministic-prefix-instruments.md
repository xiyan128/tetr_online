---
id: T22
title: Make duel and gate deterministic-prefix instruments
labels: [wayfinder:task]
status: open
assignee:
blocked-by: [T20, T21]
---

## Question

Repair the parallel pair-GSPRT so decision statistics consume a deterministic
seed prefix rather than completion order. Buffer completed work by seed index,
commit only the next contiguous pair, latch on that ordered stream, and persist
every started pair with whether it was included or excluded. Slow games must
not be informatively censored.

Also define the fixed-anchor comparison as a paired candidate/incumbent race
against the same immutable champion seeds, rather than a noisy absolute win
threshold against an ambiguous CC2 arm.

## Acceptance evidence

- Artificial completion permutations and 1-thread versus maximum-thread runs
  produce the same included seeds, pair counts, LLR trace, and verdict.
- `pairs.jsonl` records seed/index, both swapped games, timings/end reasons, and
  inclusion status; a completed nonempty run cannot have an empty game stream.
- Replaying the raw pair stream reconstructs the reported result exactly and
  detects a missing, duplicate, or reordered included pair.
- The anchor instrument reports a paired no-regression statistic for candidate
  versus incumbent against the registered champion on common seeds.
- A fast synthetic SPRT calibration exercises H0/H1 error and power, and the
  deterministic-prefix/latch tests pass under repeated scheduling stress.
