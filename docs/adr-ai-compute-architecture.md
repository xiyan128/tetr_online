# ADR: The AI compute architecture — sessions, policies, venues

Date: 2026-06-10 · Status: accepted (implemented on `feat/anytime-search`)

## Context

The AI must serve workloads with contradictory needs from one codebase:

| Workload | Needs |
|---|---|
| Headless eval / benchmarks / future SPRT & training | maximum throughput, exact node budgets, seed-reproducible games; no frame exists |
| Desktop game (Bevy, native) | no frame hitches at 60 Hz; bot strength |
| wasm game + embed (browser main thread) | same, with no second thread to lean on |
| Versus (future) | continuous pondering, root advancement on world events (garbage), anytime deadlines |
| MCGS / NN round 2 (future) | a persistent, re-rootable search object; batched/async eval |

Before this ADR, the search ran **blocking and per-decision**: `SyncRunner::submit`
drain-looped the planner to completion inline, so one `FixedUpdate` slice paid the
whole per-piece search (~25 ms native, worse on wasm), and the interactive node
budget was capped by what a single frame tolerates — the strongest shipped bot ran
at a fraction of its benchmarked strength to stay watchable.

## Decision: two currencies, four layers

The enduring object is an **anytime, re-rootable search session**, and the
load-bearing rule is the separation of currencies: **the search measures effort in
nodes and never reads a clock; callers own time and convert it to work.**

```
direct-drive (research / training / SPRT) ────────────┐
                                                      ▼
Controller — clocks: reaction window, staleness signatures, plan rendering
Runner     — VENUE only (where think() runs): Sync (blocking) / Sliced (cooperative)
Policy     — imperfection error model + decision-taking over a session (owns the RNG)
Mind       — the session: reroot / think(quantum) / best / nodes_expanded
```

- **`Mind`** (`ai/search/mod.rs`): `reroot` fingerprints the root (state + depth
  cap) and seeds the ply-1 placements, making `best()` immediately valid; `think`
  spends a caller-supplied node quantum; `best()` is anytime and never worse for
  more thinking. Greedy is the degenerate session (all work at reroot); the beam
  is batch-grain (one generation per think — a generation is one
  `evaluate_batch`, indivisible by design, the seam a neural value net needs);
  best-first is node-grain. **Decisions are invariant under quantum granularity**
  (pinned by tests): slicing chooses suspension points, never answers.
- **`Policy`**: `decide` (blocking) is literally the fused composition of
  `reroot` + `think`-until-`Ready` + `take`, so one-shot and incremental driving
  can never disagree. The node *budget* is metered here — the mind never sees it.
  `take` is anytime and applies the imperfection model exactly once per decision
  (the RNG stream is venue-independent).
- **Runner = venue**: `SyncRunner` (blocking; headless) and `SlicedRunner`
  (cooperative; one quantum per poll; submit is free, all work in polls). Quanta
  are **configured node counts, never measured time** — a sliced game is
  reproducible from `(seed, quantum, poll cadence)`. `take_now` is the trait's
  anytime valve for deadline pressure; it is implemented and tested but unwired —
  the shipped controller waits for the budget contract (see "operating point").
- **Controller**: pumps the runner every poll (the cooperative venue's quantum
  runs *inside* the reaction window, which is what hides the latency), buffers
  the finished decision until the reaction elapses, and owns staleness — including
  the hold exemption: a plan-initiated hold swap is pre-targeted, so the bot's own
  hold never discards its own maneuver (previously it re-paid reaction + a full
  search per held piece).

## The shipped operating point

`ATTACK_NODE_BUDGET = 192` = window capacity: 12 polls (200 ms default reaction at
60 Hz) × 16 nodes/poll (the wasm worst-case quantum). This is the largest
one-budget value that completes inside the reaction window on every platform —
zero pace cost relative to the blocking venue, pinned by
`attack_budget_fits_the_reaction_window` and by the venue-equivalence gate (the
sliced venue reproduces the blocking venue's game byte-identically at this point).
Raising it past window capacity is a *pace* decision (the bot thinks past its
reaction on wasm), not a constant bump — the gate tests fail loudly to force that
conversation.

## Determinism contract

- The session is deterministic in `(state, evaluator, depth cap, total nodes)`.
- The sliced venue preserves that (fixed quanta per poll) — it is the
  *deterministic* interactive venue.
- A future thread/worker venue deliberately trades timing determinism for
  throughput and is therefore **banned from benchmarks**; research and SPRT stay
  on direct-drive (`Policy::decide` / `think_to_completion`) forever.

## Deliberately not built (and why)

- **ThreadRunner (native request/response thread)** — *deferred*. At every
  pace-neutral budget the sliced venue already delivers within the reaction
  window deterministically and costs ≤ ~5 ms/frame; a request/response thread
  buys nothing until budgets exceed the window, and exploiting *that* properly
  requires continuous-ponder controller semantics (act-at-deadline via
  `take_now`, root advancement on world events) — versus-era work. Shipping a
  timing-nondeterministic venue with no customer is worse engineering than
  waiting for one. The protocol it needs is already the trait
  (`submit/poll/take_now/cancel`).
- **Web Worker venue** — *deferred to the NN era*. Even with COOP/COEP
  attainable, wasm-atomics builds of the full Bevy bundle are the fragile path.
  The strategic wasm scale-up is a dedicated worker running a small headless
  module (the embed already proves the engine compiles there), speaking the
  ThreadRunner protocol over `postMessage` with ~100-byte serialized
  observations. It becomes worth building when per-node eval cost jumps ~100×
  (the value net); at guideline budgets the sliced venue is sufficient.
- **Continuous ponder / root advancement** — versus-era. `reroot`'s fingerprint
  no-op is the hook: today it keeps a run alive across polls; subtree reuse
  across *moves* (and garbage invalidation) is the planned extension.
- **Async/batched evaluator for cross-game GPU inference** — training-era.
  `Mind::think` is the only place evals happen, so the seam stays localized.

## Consequences

- Research/benchmark games keep blocking semantics and stay byte-identical
  across the venue split (gated by tests); the hold fix deliberately changed
  bot traces for every surface (a controller bug, fixed once for all venues).
- The interactive bot no longer hitches the main thread anywhere, and its budget
  rose 150 → 192 for free (window capacity).
- MCGS, the value net, versus pondering, and a worker venue all land behind
  existing seams (`Mind`, `Evaluator::evaluate_batch`, `take_now`, the runner
  trait) rather than new ones.
