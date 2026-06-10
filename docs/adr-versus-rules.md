# ADR: Versus garbage rules live in the engine

Date: 2026-06-10 · Status: accepted (implemented on `feat/versus-core`)

## Context

The decided roadmap (guideline versus, TETR.IO fidelity dropped) runs through
"versus rules into core → win-rate climb + SPRT". Before this ADR every
garbage-receiving mechanic — the pending queue, cancellation, insertion timing,
hole choice — lived in the research harness (`tetr-research`), where the Bevy
game, the embed, and a future netplay surface could not reuse them, and where
the harness's simplification (dump *all* pending garbage immediately after
every placement) quietly diverged from guideline play. The engine itself had
only attack-sending math (`attack_lines`) and a raw insertion primitive
(`insert_garbage`).

## Decision

The **engine** owns the three rules of garbage exchange; drivers only route
attack between engines.

1. **Pending queue.** `Engine::queue_garbage(lines)` queues an opponent's
   attack as pending — visible as `EngineSnapshot::pending_garbage`, not yet on
   the board. Each batch draws **one hole column** from the receiver's own
   seeded stream (`StdRng`, engine seed XOR a salt so it can never align with
   the piece bag): a `(seed, attack sequence)` reproduces a board exactly,
   with no shared match RNG.
2. **Cancellation (offset).** At lock time the engine computes the clear's
   attack from its own award — same action, same B2B flag, same pre-increment
   combo index the research fold pinned (gated bit-for-bit by
   `engine_attack_events_match_the_research_fold`) — cancels pending garbage
   line-for-line oldest-first, and emits `EngineEvent::AttackSent` with the
   **net** remainder only.
3. **Rising.** Pending garbage enters after a lock that cleared **no** lines
   (clearing defers entry — the window cancellation lives in), between lock and
   spawn, capped per lock by `EngineConfig::garbage_cap` (default 8). A batch
   split by the cap keeps its hole column. An overflowing rise is an ordinary
   in-band `BlockOut`.

`AttackSent` fires in single-player too (nothing pending ⇒ net == gross): it is
informational, and gating it on "versus armed" would be hidden statefulness.

## Consequences

- `play_versus` (the SPRT/win-rate instrument) is now a thin router; match
  dynamics changed deliberately (digging while comboing is possible; garbage
  pressure is paced by the cap) — **prior win-rate numbers are superseded**.
- The behavior/marathon APP baselines are *unchanged*: their scenarios never
  queue garbage, and `fold_combo` (whose conventions the engine reproduces)
  remains their accounting.
- The TBP referee (`cc2_baseline`) keeps its external bookkeeping by design:
  it inserts raw, our engine's queue stays empty, and `step_piece` reports
  gross attack exactly as before. Aligning the CC2-side sim with engine timing
  is out of scope for an external-process bridge that already carries
  documented re-sync caveats.

## Deliberately deferred

- **Messiness** (per-line hole-change probability): one hole per batch matches
  the prior harness; a messiness model is a tuning decision for after the
  first win-rate climbs.
- **Garbage `CellKind`**: garbage rows still paint as I-colour. Needed for a
  real versus UI and clean TBP round-trips; bundled with the versus UI work.
- **AI awareness**: `SearchState`/`EvalContext` still ignore
  `pending_garbage`. Exposing it (so search values cancellation and survival)
  is the first step of the win-rate climb sprint, not a rules concern.
- **Entry delay / charge timing knobs** beyond the per-lock cap (TETR.IO-style
  delays were dropped with TETR.IO fidelity).
- **Opponent/targeting concepts** in core (irrelevant below 3+ player modes).
