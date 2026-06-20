# Performance Architecture Review

Date: 2026-06-19

Scope: AI decision work, core engine/search hot paths, and Bevy rendering paths that
can affect frame time. This review follows the measurement-first rule from the Rust
performance workflow: identify the hot path, change one hypothesis at a time, and
keep the result only when tests and timing support it.

## Executive Summary

The UI lag with expensive bots was primarily an AI scheduling problem, not a Bevy
rendering problem. The `APP Champion` bot used a wide/deep transposing beam
(`width=128`, `depth=9`) whose old `BeamPlanner::think()` ignored the sliced runner's
node quantum and expanded a whole generation per poll. That made one UI-frame poll
do 40-50 ms of native CPU work before wasm slowdown.

The current worktree fixes the architectural issue: beam search now stages a
generation across calls and expands up to the caller's parent-node quantum. Partial
generation scores are not published until the generation completes, so final decisions
stay deterministic and quantum-independent.

Measured native release result with the wasm-sized 16-node quantum:

| Scenario | Before champion worst poll | After champion worst poll | Node behavior |
|---|---:|---:|---|
| Empty board | 46.39 ms | 5.21 ms | `think(1)=1`, `think(16)=16` |
| Holey 6-row stack | 52.43 ms | 6.62 ms | `think(1)=1`, `think(16)=16` |

Total champion decision time is still around 93-100 ms native because the same amount
of search is spread over more polls. That is the right trade for UI smoothness: the
decision takes more frames to complete, but no single frame is blocked by a generation
burst.

## Evidence

Benchmark harness:

```text
cargo run --release --example profile_beam
```

The harness drives planners like the in-game `SlicedRunner`: each poll re-roots, then
spends a 16-node quantum. It records every poll and reports the worst poll, which is
the frame-blocking burst users perceive as lag.

The figures below are representative single-run captures on one machine (the wasm q16
operating point; the "before" block from `4b4e9c0~1`, the "after" from the beam-staging
commit). Exact milliseconds drift run to run — read the shape (worst-poll collapse,
quantum now honored), not the digits.

Fresh baseline before the beam staging change:

```text
beam w128/d9 (champion), empty:
  polls=8, total=166.11ms, worst poll=46.39ms, nodes per worst poll=128

beam w128/d9 (champion), holey6:
  polls=8, total=198.59ms, worst poll=52.43ms, nodes per worst poll=128

quantum proof:
  beam w128 think(1)=43, think(16)=43 -> ignored quantum
```

Current worktree after the beam staging change:

```text
beam w128/d9 (champion), empty:
  polls=59, total=95.36ms, worst poll=5.21ms, nodes per worst poll=16

beam w128/d9 (champion), holey6:
  polls=59, total=104.15ms, worst poll=6.62ms, nodes per worst poll=16

quantum proof:
  beam w128 think(1)=1, think(16)=16 -> honors quantum
```

Correctness gates run:

```text
cargo test -p tetr-core
cargo test
```

Both passed in the current worktree.

## Implemented Changes

### AI: Beam Search Now Honors the Runner Quantum

Changed file: `crates/tetr-core/src/ai/search/beam.rs`

Old behavior:

- `BeamPlanner::think(_quantum)` ignored `quantum`.
- Each call expanded exactly one whole generation.
- Wide beams expanded up to the beam width in a single UI poll.
- The sliced runner could not bound frame cost for beam bots.

Current behavior:

- `BeamRun` carries an optional `GenerationWork`.
- `think(quantum)` processes up to `quantum` parent frontier nodes.
- Children are scored and staged in canonical order.
- The next frontier and staged `root_best` are published only when the generation is
  complete.
- `best()` remains generation-grain and deterministic.

Why this is the right P0 fix:

- It addresses the scheduling shape that caused visible lag.
- It preserves the existing `Mind`/`Policy`/`SlicedRunner` architecture.
- It avoids changing bot strength or search semantics to mask latency.
- It keeps a future worker-thread venue optional rather than mandatory.

Risk to watch:

- `APP Champion` now needs about 59 polls at quantum 16. If the reaction window is
  short, it may act on a less-refined anytime result. That is preferable to blocking
  the UI, but it should be observed in play. The proper tuning knob is quantum or bot
  operating point, not returning to whole-generation bursts.

### AI/Core: Beam Frontier Ranking Avoids Moving Large Nodes

Changed file: `crates/tetr-core/src/ai/search/beam.rs`

The frontier ranking path sorts compact `(score, index)` pairs instead of sorting
whole `BeamNode` values. Survivors are then moved out once. This preserves canonical
tie ordering with score descending and enumeration index ascending.

Why it matters:

- Wide beam generations create many large `BeamNode` values.
- Sorting whole nodes performs unnecessary memory traffic.
- Compact ranking reduced the earlier champion worst poll from roughly 42-47 ms to
  roughly 21-26 ms before the deeper scheduling fix.

Priority status:

- Keep it. It is a measured P1 hot-path win.
- It is not sufficient alone; the P0 fix was still quantum-honoring staging.

### Core Engine/Movegen: Piece Cell Rotation Is Precomputed

Changed files:

- `crates/tetr-core/src/engine/constants.rs`
- `crates/tetr-core/src/engine/pieces.rs`

`Piece::cells()` now reads a compile-time `(piece, rotation)` table instead of
recomputing rotation on each call. This matters because collision checks and movegen
BFS call through `cells()` repeatedly.

Why it matters:

- `movegen` is called for every searched parent.
- Collision probes are inner-loop work.
- The change removes repeated small computations and keeps geometry deterministic.

Correctness evidence:

- Full `cargo test` passed.
- Piece tests still cover render index mapping, all rotations having four distinct
  cells, spawn coordinates, O rotation behavior, and wall kicks.

### AI/Core: Movegen BFS Scratch Reuse

Changed file: `crates/tetr-core/src/ai/movegen.rs`

The current worktree reuses BFS scratch through a thread-local `BfsScratch`:

- `visited`
- `emitted`
- `frontier`

Why it matters:

- Beam search calls movegen once per parent node.
- Reallocating hash sets and a frontier deque for every parent adds allocator and
  cache pressure.
- Thread-local reuse keeps capacity across calls without cross-thread contention.

Risk to watch:

- The `RefCell` would panic on recursive `enumerate()` re-entry on the same thread.
  The current movegen/search structure is not recursive, and workspace tests passed.
- If future code calls movegen from inside a movegen callback, this should be revisited.

### Bevy Rendering: Active Piece Rebuild Is Cached

Changed file: `src/session/render.rs`

`reconcile_active_pieces` no longer despawns and respawns the falling piece's four
sprites every `Update` frame. It now caches each seat's active rendered cells and only
rebuilds when the cells change.

Why it matters:

- Bevy sprite rendering itself was not the main lag source, but per-frame ECS command
  churn is avoidable.
- This aligns active-piece rendering with the existing cache pattern for locked boards
  and ghosts.
- It reduces work in frames where the simulation snapshot is unchanged.

Remaining render-side opportunities:

- Ghost outlines still rebuild on ghost-cell changes. That is already cached, but each
  visible ghost can spawn up to 16 edge sprites. If rendering becomes hot after AI is
  fixed, pool/reuse ghost edge entities.
- Locked board rendering rebuilds the full static layer when board cells change. This
  is simple and usually acceptable because locks happen at piece cadence, not every
  frame. If board rebuilds show up in traces, diff by cell or retain a fixed board
  entity grid.
- Hold/preview rebuild on piece-list changes. This is low frequency and lower priority.

## Priority Map

### P0: Keep Beam Quantum-Honoring

Status: implemented.

This is the main fix for expensive-bot UI lag. Do not regress to generation-grain
polls. The invariant to protect is:

```text
BeamPlanner::think(1) expands less work than BeamPlanner::think(16)
```

The current test `beam_honors_quantum_and_preserves_final_decision` pins this shape
and verifies tiny-quantum draining reaches the same final answer as one-shot draining.

### P1: Promote the Perf Harness Into a Maintained Diagnostic

Status: partially implemented as `examples/profile_beam.rs`.

Keep this harness or move it into a named benchmark/diagnostic command. It is more
useful than a generic Criterion microbench for this bug because the user-visible
metric is worst poll, not only total throughput.

Recommended next addition:

- Print first-poll cost separately, because `reroot()` seeding happens in the first
  `SlicedRunner::poll`.
- Print p50/p95/max poll, not only max.
- Add a wasm/browser run path or a small JS harness for the actual canvas target.

### P1: Add Runtime AI Poll Diagnostics

Status: recommended.

Add an opt-in diagnostic counter around `SlicedRunner::poll`:

- current model label
- poll count for current decision
- nodes expanded this poll
- elapsed poll wall time in debug/dev builds
- decision completion latency in frames

This would make future reports actionable: "which bot, which poll, how many nodes,
how long." Keep it disabled or dev-only for release determinism/noise.

### P1: Tune Champion Operating Point After Playtesting

Status: recommended.

After staging, `APP Champion` no longer blocks a frame, but it takes about 59 polls at
quantum 16. If this makes champion actions feel late, tune in this order:

1. Increase wasm/native quantum separately if frame budget allows.
2. Reduce `APP Champion` depth before reducing width if late decisions are frequent.
3. Prefer best-first or PC-coverage operating points for interactive bots if they
   produce better strength per millisecond.
4. Move only the most expensive bots to a worker/thread venue if main-thread quantum
   tuning is not enough.

Do not choose "whole generation per poll" as a strength fix; that reintroduces UI lag.

### P2: Reduce Search Allocation Further

Status: optional.

Candidates:

- Reuse `pending`, `nodes`, and `ranked` buffers across beam generations.
- Reserve staged generation vectors based on previous generation fanout.
- Continue measuring movegen allocation impact after `BfsScratch`.
- Consider smaller key/index types only where sizes are proven hot.

Do this only after poll diagnostics or allocation profiling show search allocation is
still visible. The current UI problem has already moved from frame burst to total
decision latency.

### P2: Render Entity Reuse

Status: optional.

If Bevy systems appear in traces after the AI fix:

- Pool active/ghost sprite entities instead of despawn/spawn.
- Consider a fixed 10x40 board-cell entity grid per seat with visibility/tint updates
  instead of rebuilding board children on every lock.
- Keep post effects opt-in. CRT and bloom are explicit render passes/uniform updates;
  they should stay easy to disable for perf comparisons.
- Ambient wave is already limited to a stepped 10 Hz repaint and is user-toggleable;
  do not prioritize it unless browser profiling shows texture upload cost.

### P3: Build/Profile Configuration

Status: recommended for release process, not this bug.

For native profiling, keep using release builds. For deeper profiling:

- Add line-table debug info for release profiles when sampling.
- Use platform profilers (`Instruments` on macOS, `samply` cross-platform) to confirm
  remaining hotspots after the quantum fix.
- For distribution builds, evaluate LTO/codegen/allocator settings only with
  benchmark comparison. Do not mix them into gameplay changes without measurement.

## What Not To Prioritize Now

- Rewriting Bevy rendering before validating post-fix UI behavior. The measured burst
  was AI-side, and render changes should be trace-driven.
- Hand-optimizing evaluator arithmetic. The current worst-poll reduction came from
  scheduling and memory movement, not from scalar math.
- Moving all AI to worker threads immediately. Thread/worker venues are useful, but
  the main-thread cooperative contract now works and preserves deterministic behavior.
- Reducing bot strength blindly. If a bot feels late, tune quantum/depth/worker venue
  with poll diagnostics.

## Completion Criteria For This Performance Track

The performance track is in a good state when all of these are true:

- Expensive bots no longer produce single-poll bursts above the frame budget in native
  and wasm measurements.
- `BeamPlanner` quantum behavior is tested.
- The maintained harness reports worst poll and total decision latency.
- Runtime diagnostics can identify a lagging bot/poll without adding ad hoc prints.
- Render changes are driven by traces after the AI fix, not speculation.

Current status:

- Native release worst poll is below a 16.67 ms frame in the measured fixtures.
- Full Rust test suite passes.
- Wasm/browser timing still needs direct measurement.
- Runtime diagnostics are still recommended.
