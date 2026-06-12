# ADR: the data architecture (research now, live play later)

Status: accepted 2026-06-12. Owners: tetr-research (current scope); the
invariants below also bind any future live-multiplayer telemetry.

## The one principle

**Store the minimal cause; derive everything else.** The engine is
deterministic, so a game is a pure function of its causes:

| regime | minimal cause | size |
|---|---|---|
| research (today) | `(commit, bot names, seed)` | ~bytes |
| live multiplayer (future) | `(rules version, seed, per-player input streams)` | ~KB/game |

Everything downstream — trajectories, metrics, training data — is a
*derivation*, materialized on read (duckdb views) or on demand (trace
replays), never stored in the system of record. Deleting every derived
artifact must lose nothing.

## The invariants (these are the architecture)

1. **Determinism is load-bearing.** A game replays bit-exactly from its
   cause. FP stays IEEE across opt levels and targets (golden-gated); this
   is also the lockstep-multiplayer enabler.
2. **Receipts vs facts.** Every run/match writes one receipt (`spec.json`:
   the parameters and provenance — commit, dirty flag, eval, bots, runtime)
   and one append-only fact stream (`games.jsonl`: raw outcomes only).
   They join on the run id, which is the directory name and is stored
   nowhere else.
3. **Facts are normalized.** No field a receipt, a path, or a query already
   determines. The one ordinal `n` survives because row order is semantic
   and JSON line numbers are not portably queryable. Consequence: fact
   files are byte-identical across replays — `diff` is a replay witness
   (smoke-asserted).
4. **Rows are self-describing JSON; fields are add-only.** Never rename or
   repurpose a field — a semantic change gets a new name (the same
   immutability rule as registry names). Identifiers (seeds, ids) travel as
   strings; u64s corrupt through f64-only readers.
5. **Analytics is read-side, outside the platform.** duckdb owns queries
   (`scripts/research.sql`); denormalization may exist only in views. The
   platform never reads its own telemetry back — telemetry observes runs,
   it must never steer them or end them (emission no-ops unsinked and
   swallows write errors).
6. **Single writer per file; coordination-free distribution.** A run owns
   its directory. A fleet of workers therefore shards naturally: each
   writes locally, directories sync to one analytics root (rsync/object
   store), duckdb globs across it. No locks, no services, no migrations.

## Named landing spots (deferred until a need is real)

- **Volume**: JSONL → Parquet via `COPY … TO` compaction; Arrow IPC if
  event volume ever 100×es or live dashboards want zero-copy.
- **Verdict queries**: a `verdicts` duckdb view implementing the GSPRT fold
  with window functions — read-side, zero storage. Until then, conclusions
  are curated into doc-header RUN RECORDs citing run ids.
- **Training data**: a `trace` eval that replays selected games and
  materializes per-ply rows (layer 1: engine-observable; layer 2:
  search-candidate scores via an opt-in `SearchPolicy` hook), Parquet
  output, its own receipt — a regenerable materialization, not a record.
- **Live play**: per-match replay files = `(rules version, seed, input
  streams)`; an explicit engine `RULES_VERSION` constant must land with the
  first netcode (clients cannot be pinned to commits). Human games then
  feed the same trace/training pipeline.
- **The one logging exception**: externally nondeterministic opponents
  (CC2 over TBP) are not replayable from their cause; those runs must
  record trajectories at play time if ever wanted.

## What we deliberately do not build

Transactions, indices, retention policies, schema registries, tracking
services, or any write-time columnar format. At this scale they are
ceremony; the invariants above keep every one of them adoptable later
without migrating what is already on disk.
