# ADR: The AI compute architecture (sessions, policies, venues)

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
budget was capped by what a single frame tolerates. The strongest shipped bot ran
at a fraction of its benchmarked strength to stay watchable.

## Decision: two currencies, four layers

The long-lived object is an **anytime, re-rootable search session**, and the
load-bearing rule is the separation of currencies: **the search measures effort in
nodes and never reads a clock; callers own time and convert it to work.**

```
direct-drive (research / training / SPRT) ────────────┐
                                                      ▼
Controller - clocks: reaction window, staleness signatures, plan rendering
Runner     - VENUE only (where think() runs): Sync (blocking) / Sliced (cooperative)
Policy     - imperfection error model + decision-taking over a session (owns the RNG)
Mind       - the session: reroot / think(quantum) / best / nodes_expanded
```

- **`Mind`** (`ai/search/mod.rs`): `reroot` fingerprints the root (state + depth
  cap) and seeds the ply-1 placements, making `best()` immediately valid; `think`
  spends a caller-supplied node quantum; `best()` is anytime and never worse for
  more thinking. Greedy is the degenerate session (all work at reroot); the beam
  is batch-grain (one generation per think; a generation is one whole-layer
  evaluation, indivisible by design, the seam a neural value net needs);
  best-first is node-grain. **Decisions are invariant under quantum granularity**
  (pinned by tests): slicing chooses suspension points, never answers.
- **`Policy`**: `decide` (blocking) is literally the fused composition of
  `reroot` + `think`-until-`Ready` + `take`, so one-shot and incremental driving
  can never disagree. The node *budget* is metered here; the mind never sees it.
  `take` is anytime and applies the imperfection model exactly once per decision
  (the RNG stream is venue-independent).
- **Runner = venue**: `SyncRunner` (blocking; headless) and `SlicedRunner`
  (cooperative; one quantum per poll; submit is free, all work in polls). Quanta
  are **configured node counts, never measured time**: a sliced game is
  reproducible from `(seed, quantum, poll cadence)`. (An anytime `take_now`
  valve for deadline pressure shipped with the seam, sat unwired, and was
  deleted in the 2026-06-10 no-compat sweep; the policy's `take` verb is the
  anytime primitive, and a deadline venue re-adds the runner verb trivially.)
- **Controller**: pumps the runner every poll (the cooperative venue's quantum
  runs *inside* the reaction window, which is what hides the latency), buffers
  the finished decision until the reaction elapses, and owns staleness,
  including the hold exemption: a plan-initiated hold swap is pre-targeted, so
  the bot's own hold never discards its own maneuver (without the exemption,
  every held piece re-pays the reaction delay plus a full search).

## The shipped operating point

`ATTACK_NODE_BUDGET = 192` = window capacity: 12 polls (200 ms default reaction at
60 Hz) × 16 nodes/poll (the wasm worst-case quantum). This is the largest
one-budget value that completes inside the reaction window on every platform:
zero pace cost relative to the blocking venue, pinned by
`attack_budget_fits_the_reaction_window` and by the venue-equivalence gate (the
sliced venue reproduces the blocking venue's game byte-identically at this point).
Raising it past window capacity is a *pace* decision (the bot thinks past its
reaction on wasm), not a constant bump; the gate tests fail loudly to force that
conversation.

## Determinism contract

- The session is deterministic in `(state, evaluator, depth cap, total nodes)`.
- The sliced venue preserves that (fixed quanta per poll): it is the
  *deterministic* interactive venue.
- A future thread/worker venue deliberately trades timing determinism for
  throughput and is therefore **banned from benchmarks**; research and SPRT stay
  on direct-drive (`Policy::decide` / `think_to_completion`) forever.

## Deliberately not built (and why)

- **ThreadRunner (native request/response thread)**: *deferred*. At every
  pace-neutral budget the sliced venue already delivers within the reaction
  window deterministically and costs ≤ ~5 ms/frame; a request/response thread
  buys nothing until budgets exceed the window, and exploiting *that* properly
  requires continuous-ponder controller semantics (act at a deadline through
  the policy's anytime `take`, root advancement on world events), which is
  versus-era work. Shipping a timing-nondeterministic venue with no customer is
  worse engineering than waiting for one. The protocol it needs is the trait
  (`submit/poll/cancel`) plus a deadline verb that is trivial to re-add.
- **Web Worker venue**: *deferred to the NN era*. Even with COOP/COEP
  attainable, wasm-atomics builds of the full Bevy bundle are the fragile path.
  The strategic wasm scale-up is a dedicated worker running a small headless
  module (the embed already proves the engine compiles there), speaking the
  ThreadRunner protocol over `postMessage` with ~100-byte serialized
  observations. It becomes worth building when per-node eval cost jumps ~100×
  (the value net); at guideline budgets the sliced venue is sufficient.
- **Continuous ponder / root advancement**: versus-era. `reroot`'s fingerprint
  no-op is the hook: today it keeps a run alive across polls; subtree reuse
  across *moves* (and garbage invalidation) is the planned extension.
- **Async/batched evaluator for cross-game GPU inference**: training-era.
  `Mind::think` is the only place evals happen, so the seam stays localized.

## Consequences

- Research/benchmark games keep blocking semantics and stay byte-identical
  across the venue split (gated by tests); the hold fix deliberately changed
  bot traces for every surface (a controller bug, fixed once for all venues).
- The interactive bot no longer hitches the main thread anywhere, and its budget
  rose from 150 to 192 for free (window capacity).
- MCGS, the value net, versus pondering, and a worker venue all land behind
  existing seams rather than new ones: `Mind`, the runner trait, and the
  `Evaluator` trait, where a batched backend re-adds a batch verb.

## Update 2026-06-19: the time-budgeted venue (`BudgetedRunner`, built)

The deadline verb anticipated above ("trivial to re-add") shipped, for a reason the
original sizing missed. The sliced venue spends **one node quantum per frame**, sized
so the node-bounded best-first attack bot (192 nodes ≈ 12 quanta) finishes inside its
12-frame reaction window. But the catalog's *open-ended beams* need far more — the APP
champion is ~30 quanta — so at one quantum per frame the controller waits ~30 frames
(~0.5 s/piece on native) while each poll leaves most of the frame idle. The bot was
throttled by the frame loop, not the CPU.

[`BudgetedRunner`](../crates/tetr-core/src/ai/runner/budgeted.rs) spends quanta until a
per-frame **wall-clock** budget (8 ms native / 4 ms wasm) instead of exactly one,
landing the champion in ~12 frames (~200 ms, inside its reaction window) at full
strength — measured 2 → 5 pieces/s native, worst poll ~10 ms.

This is the venue the "two currencies" rule forbids the *search* from being, made legal
at the *venue* layer: the runner reads a clock, so it is **timing-nondeterministic by
design** (a poll's quantum count depends on machine speed) and is therefore the game's
venue only — benchmarks, research, and the venue-equivalence gate stay on the
deterministic `SyncRunner`/`SlicedRunner`. The clock is an injected `MonotonicClock`
trait, so the core stays clock-free (`std::time::Instant` panics on wasm; the host
supplies Bevy's web-time-backed `Instant`) **and** the budgeting is deterministically
testable with a fake clock. The *decision* is invariant under the budget — the budget
moves only which poll it lands on, pinned by tests. Only `AiController::interactive`
(the catalog beams) switched; `attack`/embed/PC/benches are untouched.

### Parked alternative: an off-thread native venue (`ThreadRunner`)

An off-thread venue was prototyped alongside the budgeted one and parked in its
favour. `ThreadRunner` runs the policy on a dedicated worker thread (`submit` /
non-blocking `poll` / `cancel`, epoch-tagged stale-discard, `Drop`-joins the worker),
so the search costs the render thread **nothing** — the worker computes the full
decision in parallel and the controller acts at its reaction deadline. Like the
budgeted venue it is decision-identical to blocking (the worker drives to `Ready`),
timing-nondeterministic, and benchmark-banned.

We shipped the budgeted venue instead because it fixes **both native and wasm in one
venue** (the curated wasm build has no worker thread) and is simpler / lower-risk; the
thread venue's zero-main-thread-cost edge is native-only and a speculative win while
the renderer is cheap. **Revive it if main-thread contention ever bites** — a much
heavier renderer, many bot seats, or a native-only high-performance build. It drops in
behind the same `DecisionRunner` seam: branch `feat/native-thread-runner` (the
`ThreadRunner` commit `b8801b7`, off `perf/ai-search-perf`), gate-green, including the
per-platform venue selection and the `attack_policy` split / embed cooperative-venue
pin it needs.
