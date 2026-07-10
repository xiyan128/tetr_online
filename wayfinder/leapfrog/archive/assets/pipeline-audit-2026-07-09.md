# Pipeline silent-defect audit (2026-07-09, multi-agent adversarial)

Independent of and concurrent with the T18/T19 validity reset: a 6-lens
multi-agent audit (alignment, cross-language divergence, distribution
landmines, statistical validity, hidden state, venue semantics) over the
legacy datagen→train→gate pipeline, every finding cross-examined by two
adversarial refuters. 86 agents; the verify phase was truncated by a session
rate limit (64 verifier deaths), so findings split into CONFIRMED (survived
both refuters) and UNVERIFIED (finder claims whose refuters died — treat as
leads, not verdicts). Full journal:
`~/.claude/projects/.../subagents/workflows/wf_dd7b6e18-100/journal.jsonl`.

These corroborate the validity reset from the outside: every confirmed
finding lands inside a T20-T30 ticket's blast radius. Recorded here so the
repaired stack's acceptance tests can consume them as known failure cases.

## CONFIRMED (survived adversarial verification)

1. **[HIGH] round.py mix glob misses the flat 1-worker shard layout**
   (`round.py` mix step; `main.rs` writes `out/wN/` only when workers > 1).
   With `--workers ≤ 3`, a datagen half gets 1 worker → flat shards → the
   `*/w*/shard-*` glob silently excludes that half from the replay mix; the
   round trains on stale base replay with every log line looking normal.
   Round-11 itself was UNAFFECTED (both halves ran 6 workers → wN layout).
   Ticket home: T21/T28 (manifest-validated mixes); the repaired driver must
   assert mix contents against expected shard counts.

2. **[MEDIUM] mix/train resume on bare existence** (`round.py` steps 2-3).
   `mix.mkdir()` precedes the symlink loop → a mid-loop kill leaves a
   partial mix a rerun silently accepts; train resumes on `config.json`
   existence while train.py exports per-epoch → a killed multi-epoch run
   resumes as "finished". Ticket home: T21/T28 (authoritative manifests).

3. **[MEDIUM] `--boot-value` ignores the score-unit identity contract**
   (`train.py`). Applies `tanh(score/10000)` (net units) to every played
   child, but cc2-tagged two-arm shards interleave net-seat and CC2-seat
   rows within one shard; even name-based resolution cannot separate them
   without game_id%2+seat logic. A rebuilt round-3-class trap one flag away.
   Ticket home: T23 (per-row score-unit provenance; trainer rejects mixed
   absolute units).

4. **[MEDIUM] datagen rerun into an existing --out appends duplicate games**
   (`main.rs` worker loop). Seeds replay unconditionally; ShardWriter
   continues numbering; `recorded_game_ids` (built for this) is never called
   by the CLI. Duplicates double-weight games and can straddle the every-10th
   holdout boundary (train/holdout leakage of the same game). Ticket home:
   T21/T23 (dataset manifests with expected/completed games).

## UNVERIFIED LEADS (refuters died on rate limit — re-verify before acting)

- Seed-region collisions across round roles (the reset's round.py rewrite
  already flags the historical formulas as overlapping — convergent).
- Fine-tune holdout contains base-corpus shards the lineage net already
  trained on (v4 pretraining saw the full base corpus; the mix's every-10th
  holdout includes base shards → holdout pCE partially measures memorized
  data). If true, every fine-tune round's holdout metrics were optimistic.
- Gate GSPRT pair/game accounting semantics; anchor-veto string parsing
  fragility when draws are nonzero.
- Hard-cap (TrueCap) games may get decisive z labels rather than draws
  (E8-adjacent; affects value labels on long games).
- "CC2" is not one bot across the pipeline (attack_tuned vs other weight
  sets in different seams).
- `Net::load` silently defaults missing config fields; Python reader never
  verifies FEATURE_LEN against the file.
- fit_slots holdout uses the lexicographic tail (not every-10th) —
  different convention from train.py within one repo.

## Method note

The four confirmed findings are all *hidden-state / resume / provenance*
class — the same class the validity reset attacks with manifests and
fail-closed entry points. None are reachable once T21/T23/T28 land as
specified; they are the concrete regression cases those tickets should pin.
