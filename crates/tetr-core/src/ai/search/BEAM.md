# BEAM.md — Deterministic, batch-shaped beam search behind the `Planner` trait

> Status: **design, locked**. This is the contract the Colder Clear strike implements.
> Scope: a CC2-style **deterministic beam** Tier-2 planner, dropped in behind the
> existing [`Planner`] trait, gated on `bench-marathon`. North star: beat Cold Clear
> in this stack. This document is normative — implementers follow it verbatim.

---

## 0. What stays the same (and why)

The whole point of the `(Value, Reward)` / `Planner` / `Evaluator` seam is that a
multi-ply search is a **pure backend swap**. Nothing below the search changes:

| Layer | File | Why unchanged |
|---|---|---|
| Engine `Board` & rules | `crates/tetr-core/src/engine/*` | **ADR-7 boundary.** The search clones a `SearchState` and locks through the engine's own [`lock_and_clear`]; it never re-encodes rules. Touching `Board` is out of bounds. |
| Controller | `crates/tetr-core/src/ai/controller.rs` | Holds a `Box<dyn Policy>`; never names a planner. |
| `SearchPolicy` | `crates/tetr-core/src/ai/policy/search.rs` | `plan_best` (lines 86–95) **already** loops `while NeedMoreBudget`. A beam that yields `NeedMoreBudget` per generation runs here with **zero change**. The imperfection softmax (`score_candidates`) re-scores ply-1 placements with the same evaluator — also unchanged. |
| `plan.rs` (plan→input) | `crates/tetr-core/src/ai/plan.rs` | Consumes a `Placement.path`. The beam returns a **ply-1** `Placement` whose path is exactly what movegen produced (incl. a leading `Move::Hold`). Identical shape to greedy's output. |
| `DecisionRunner` / `SyncRunner` | `crates/tetr-core/src/ai/runner/mod.rs` | `submit`/`poll`. The cooperative time-slicing the beam needs is expressed entirely through `PlannerStep::NeedMoreBudget`, which the policy already drives. No new runner is required for v1. |
| Movegen | `crates/tetr-core/src/ai/movegen.rs` | [`generate_with_hold`] already emits hold-aware placements. The beam only *consumes* it. |

The beam is **purely additive**: one new method on `SearchState`
(`commit_placement`), one new default method on `Evaluator` (`evaluate_batch`), and
one new planner (`BeamPlanner`). Everything else is a re-use.

---

## 1. Determinism rule (non-negotiable)

The [`Planner`] contract (`search/mod.rs` docs) promises a planner is a deterministic
function of `(state, evaluator, budget)`: **no RNG, no clock**. The beam upholds this:

1. **Zero RNG in the planner.** Any randomness the AI wants (imperfection) lives in
   `SearchPolicy`'s seeded `StdRng`, downstream of and oblivious to the beam.
2. **Canonical order is the only tie-breaker.** [`generate_with_hold`] already returns
   placements in a stable canonical order (`sort_placements`: rotation discriminant,
   then origin.x, origin.y, then `used_hold`). When two beam nodes tie on score, the
   one **enumerated first** wins. Concretely: children are pushed in
   (parent-order, movegen-order) and we use a **stable** sort descending by score, so
   ties resolve to the earlier-enumerated node. No transposition table, no node
   hashing, no expectimax sampling in v1.
3. **Back-up is order-stable too.** A root's score is the max leaf score reachable
   from it; on equal leaf scores the first-seen leaf wins (`>` comparison, not `>=`),
   matching greedy's "keep the first maximum" rule (`greedy.rs:101`).

Determinism is verified by a test that plans the same `SearchState` twice and asserts
the identical ply-1 `Placement` (origin, rotation, path) and score (mirrors
`greedy_is_deterministic`).

---

## 2. Node shape

```rust
/// One node in the beam frontier: a forked search state plus the bookkeeping the
/// back-up and the final ply-1 decision need.
struct BeamNode {
    /// The forked, owned search state at this node (board + active + hold + queue +
    /// bag + b2b). Cheap to clone (no engine, no RNG, no timers).
    state: SearchState,
    /// Sum of per-move `Reward`s along the path from the root to this node
    /// (Cold Clear's reward-folds-into-value design). Folded into a leaf's static
    /// `Value` at scoring time.
    acc_reward: Reward,
    /// Index into the ply-1 root frontier this node descends from. The whole
    /// subtree carries the *same* `root_index`, so the best leaf can credit the
    /// ply-1 move that owns it.
    root_index: usize,
    /// This node's score: `(leaf_value + acc_reward).0`, in evaluator units. Cached
    /// so sort/truncate is a field read, not a re-evaluation.
    score: i32,
}
```

Notes:
- `Reward` and `Value` are the existing `eval` newtypes; `Value + Reward -> Value`
  and `Reward + Reward -> Reward` are the existing `Add` impls — this is *exactly*
  the path-accumulation the module docs promised, now exercised for the first time.
- `root_index` is the spine of the back-up. The ply-1 frontier is a `Vec<Placement>`
  (the canonical movegen output); a node's `root_index` indexes that vec. The final
  decision is `roots[best_root_index]`.
- The node owns its `SearchState` (not a borrow) because the parent generation is
  dropped once its children are formed; lifetimes would otherwise pin a generation we
  want to free.

---

## 3. The hold-aware transition: `SearchState::commit_placement`

This is the **prerequisite** the whole strike rests on. Today `commit` (state.rs:185)
and `commit_with_next` (state.rs:199) lock the active piece and pull the next piece,
but **do not model a hold swap** — while [`generate_with_hold`] *does* emit `used_hold`
placements. So a multi-ply search can *enumerate* hold moves but cannot *transition*
through them. `commit_placement` closes that gap.

### Contract

```rust
/// Apply a movegen `Placement` to this state, locking it through the engine and
/// advancing to the next piece — **modelling a hold swap** when the placement used
/// one.
///
/// This is the multi-ply transition. A beam clones the state, calls
/// `commit_placement(&placement)`, and recurses from the mutated clone. It mirrors
/// `plan.rs`'s move interpreter: a `used_hold` placement is the *swapped-in* piece
/// already at its resting pose, so the swap must be reflected in `self.hold` and the
/// piece stream exactly as the engine would.
///
/// Semantics:
/// 1. If `placement.used_hold`:
///    - The swapped-in piece (`placement.piece`) becomes active at its **resting**
///      pose (taken from the placement, *not* re-spawned), and the previously-active
///      piece type moves into `self.hold`.
///    - If `self.hold` was empty, the swapped-in piece came from the queue front, so
///      `queue.pop_front()` is consumed to fund the swap.
///    - The bag is **not** re-dealt for the swapped-in piece: it was already dealt
///      when it first entered the stream (hold is outside bag accounting — see
///      `from_snapshot`'s note, state.rs:155-159).
/// 2. Lock the now-active piece via `lock_and_clear` (engine-faithful clears).
/// 3. Advance to the next piece for the *following* ply: `queue.pop_front()` spawns
///    as the new active (and `bag.deal`s it). When the queue is empty, leave active
///    unchanged — a speculative caller supplies the next piece via
///    `commit_with_next` / the bag-speculation path (§5).
/// 4. Update `self.b2b` from the `LockOutcome` (§6).
/// 5. Return the `LockOutcome` for the reward half.
pub fn commit_placement(&mut self, placement: &Placement) -> LockOutcome;
```

### Why the placement carries the pose (and we don't re-spawn)

A `Placement.piece` is the piece **at its landed pose** — that is the whole output of
movegen's BFS. For a `used_hold` placement, movegen already enumerated the swapped-in
piece from its spawn and found resting poses; `Placement.piece` *is* the swapped-in
piece, landed. So `commit_placement` must lock **`placement.piece`**, not the current
`self.active`. This is the one substantive difference from `commit`, which locks
`self.active` at its current pose.

### Reference implementation (locked)

```rust
pub fn commit_placement(&mut self, placement: &Placement) -> LockOutcome {
    if placement.used_hold {
        // Fund the swap: an empty hold pulls the queue front (engine hold rule).
        // The swapped-in piece is whatever `from_snapshot`/movegen put in hold or at
        // the queue front; we consume that slot symmetrically with the swap.
        let displaced = self.active.piece_type();
        if self.hold.is_none() {
            // The swapped-in piece came from the queue front; consume it.
            self.queue.pop_front();
        }
        self.hold = Some(displaced);
    }

    // Lock the piece the placement actually rests as: the swapped-in piece when
    // used_hold, else the current active. Both are `placement.piece` for a movegen
    // placement of the current state, so we lock that uniformly.
    let outcome = lock_and_clear(&placement.piece, &mut self.board);

    // Update branch-local B2B from this clear (see §6).
    self.update_b2b(&outcome, classify_for(&placement.piece /* pre-lock */));

    // Advance to the next piece for the following ply.
    if let Some(next) = self.queue.pop_front() {
        self.spawn(next);
    }
    outcome
}
```

> **Implementation note (T-spin at commit time).** The B2B update needs the T-spin
> classification of *this* placement, which the engine computes **against the
> pre-lock board** ([`classify_t_spin`], greedy.rs:111). `commit_placement` therefore
> classifies `placement.piece` against `self.board` **before** `lock_and_clear`
> mutates it, exactly as `score_placement` does. The pseudocode's `classify_for` is
> that pre-lock call; STEP 0 inlines it (`let t = classify_t_spin(&placement.piece, &self.board);`)
> immediately before the lock. See §6 for the exact b2b rule.

### Invariant: bag is dealt exactly once per piece

The confirmed-fix corpus repeatedly flags an over-deal risk. The rule, locked:

- A piece is `bag.deal`'d **exactly when it first becomes active via `spawn`**.
- `commit_placement` calls `spawn` only for the *next* queued piece (step 3) — the
  normal path, identical to `commit`.
- The swapped-in piece in step 1 is **already dealt** (it was either the held piece,
  which was dealt when it first entered play, or the queue front, which was dealt when
  the queue was built in `from_snapshot`). So step 1 **must not** `spawn` or `deal` it —
  it only relocates it into `active` by locking `placement.piece` directly.

This keeps `BagState` faithful to the engine across hold transitions.

---

## 4. The per-call generation loop (cooperative time-slicing)

`BeamPlanner::plan` processes **exactly one generation per call**, then yields. The
policy's `plan_best` loop (search.rs:88) re-invokes until `Done`, so on native this
finishes in one `decide()`; on threadless WASM a cooperative runner can call it once
per frame (the seam already exists; v1 needs no new runner).

State carried between calls lives on the planner (`&mut self`):

```rust
pub struct BeamPlanner {
    beam_width: usize,
    /// Whether to speculate past the visible queue over the bag (§5).
    speculate: bool,
    /// In-flight search, `None` between decisions. Reset on a new root state.
    run: Option<BeamRun>,
}

struct BeamRun {
    /// The ply-1 placements, in canonical movegen order. `root_index` indexes this.
    roots: Vec<Placement>,
    /// Best leaf score seen so far per root (back-up target). `i32::MIN` = unseen.
    root_best: Vec<i32>,
    /// The current frontier (already truncated to <= beam_width).
    frontier: Vec<BeamNode>,
    /// Plies expanded so far (root seeding = depth 1).
    depth: u8,
    /// Identity of the state this run was seeded from, to detect a stale run.
    root_fingerprint: RootFingerprint,
}
```

### First call (seed depth 1)

```text
roots := generate_with_hold(state.board, state.active, state.hold,
                            state.queue.front(), spawn_for)   // canonical order
if roots is empty: return Done(None)                          // topped out

frontier := []
for (i, p) in roots.enumerate():
    child := state.clone()
    lock  := child.commit_placement(&p)
    // score the ROOT child: value of resulting board + this move's reward
    node  := BeamNode { state: child, acc_reward: Reward(0), root_index: i, score: 0 }
    push node to a scratch list, remember (lock, &child.board, t_spin(p)) for batch
evaluate the scratch list as ONE batch  -> Vec<(Value, Reward)>
for each (node, (value, reward)) :
    node.acc_reward := reward
    node.score      := (value + reward).0
    root_best[node.root_index] := max(root_best[node.root_index], node.score)
stable-sort frontier descending by score; truncate to beam_width
depth := 1
if depth >= budget.max_depth OR frontier empty:
    return Done(best ply-1 placement by root_best)        // single-ply == greedy
else:
    return NeedMoreBudget
```

### Subsequent calls (expand one generation)

```text
next := []
batch_inputs := []                  // (LockOutcome, Board, Option<TSpinKind>) owners
batch_meta   := []                  // (root_index, acc_reward, child_state)
for parent in frontier:             // already canonical (stable-sorted, ties by enum order)
    children := generate_with_hold(parent.state.board, parent.state.active,
                                   parent.state.hold, parent.state.queue.front(), spawn_for)
    for p in children:              // canonical movegen order
        child := parent.state.clone()
        lock  := child.commit_placement(&p)
        t     := t_spin(p) against parent.state.board (pre-lock)
        batch_inputs.push((lock, child.board.clone-or-borrow, t))
        batch_meta.push((parent.root_index, parent.acc_reward, child))

    // §5: if speculate AND parent.queue is empty, also branch over the bag here
    //     (see "Speculation" — same shape, commit_with_next, pessimism decay).

scores := eval.evaluate_batch(&batch_inputs)        // ONE forward for the whole generation
for ((root_index, parent_acc, child), (value, reward)) in zip(batch_meta, scores):
    acc   := parent_acc + reward                    // Reward + Reward
    score := (value + acc).0                        // Value + Reward
    push BeamNode { state: child, acc_reward: acc, root_index, score } to next
    root_best[root_index] := max(root_best[root_index], score)

stable-sort next descending by score; truncate to beam_width
frontier := next
depth += 1
if depth >= budget.max_depth OR frontier empty:
    let best_i := argmax_stable(root_best)          // first max wins (determinism)
    return Done(Some(PlacementPlan { placement: roots[best_i], score: root_best[best_i] }))
else:
    return NeedMoreBudget
```

### Budget semantics

- `budget.max_depth` caps plies; **depth 1 reproduces greedy exactly** (one generation,
  one batch, pick the best root). The bench must show beam@depth-1 == greedy to prove
  the seam is faithful before depth is raised.
- `budget.nodes` caps states expanded *per call* (the WASM time-slice unit). v1 may
  expand a whole generation per call (a generation is `beam_width × ~34` children,
  bounded); honoring `nodes` as a finer mid-generation yield is a documented v1.1
  refinement, not required for the native bench. If `nodes != 0` and a generation
  would exceed it, the planner may still finish the generation (the contract says
  `nodes` is the *unit* a slice is measured in, not a hard mid-expansion cut).
- A new `SearchBudget::beam(width, depth)` constructor sits beside `greedy()`.

### Why argmax over `root_best`, not "best leaf's root"

Backing up the **max** leaf score per root and then taking the best root is a beam's
standard value back-up: a ply-1 move is worth the best line the beam still holds that
descends from it. Tracking `root_best[i]` incrementally (updated every generation)
means the answer is correct even if a root's *only* surviving descendants were pruned
in a later generation — `root_best` retains the best score that root ever achieved
while it was alive in the beam. (A root whose entire subtree is pruned early keeps its
depth-1 score, never worse than not searching it.)

---

## 5. 7-bag speculation past the visible queue

When a branch consumes the entire revealed Next queue, the next piece is *unknown* but
**constrained**: it is drawn from the current 7-bag remainder, which `SearchState.bag`
([`BagState`]) already reconstructs. The beam speculates over it **deterministically**:

- At a node whose `queue` is empty, instead of one child-per-placement we branch over
  **each `PieceType` still in `state.bag`** (`BagState::contains`), iterated in
  `PieceType::all()` order (canonical, deterministic). For each candidate piece, we
  form the placements via the existing `commit_with_next(piece)` path (state.rs:199),
  which deals the speculative piece and advances the bag.
- Each speculative branch's contributed reward is **decayed by a small pessimism
  factor per speculative ply** (we cannot rely on a piece we have not seen). Locked
  shape: a multiplicative `SPEC_DECAY ∈ (0,1]` (start `0.75`) applied to the
  *branch's* `acc_reward` contribution at each speculative depth, *not* to the static
  `Value` (the resulting board is real regardless of which piece arrives). This biases
  the beam toward lines robust across bag orderings without an expectimax average.
- **No RNG, no expectimax sampling in v1.** We do not weight by probability or sample a
  representative piece; we enumerate all bag-legal pieces and let **beam-width
  truncation** prune the combinatorial fan-out. This keeps the planner a pure
  deterministic function (the determinism rule, §1) and keeps the door open for a
  probability-weighted v2 behind the same node shape.

Speculation is gated by `BeamPlanner.speculate` (default on for depth > visible-queue
length; the bench can toggle it). At depths shallower than the visible queue, every
node still has a non-empty `queue`, so speculation never triggers and the search is
exact.

> Rationale for "enumerate, don't sample": the visible Next queue in this engine is
> long enough that a depth-≤queue beam is fully concrete; speculation only matters at
> the deep tail where truncation already dominates, so the cheap deterministic
> over-branch (≤7 pieces) is both faithful and reproducible.

---

## 6. Branch-local B2B (`SearchState::update_b2b`)

`SearchState.b2b` is carried per node but never updated by `commit`/`commit_with_next`
today. Multi-ply reward correctness needs it, because `compute_reward` (eval/mod.rs:171-189)
pays a B2B bonus on every B2B-*eligible* clear. The B2B-eligibility rule already lives
in `compute_reward`'s match; the beam extracts the **state-transition** half:

```rust
/// Update the branch-local Back-to-Back flag from the clear that just happened.
///
/// B2B *continues* on a B2B-eligible clear (a Tetris, or any line-clearing full
/// T-spin), *breaks* on a non-eligible line clear, and is *preserved* when no line
/// cleared (a placement that clears nothing neither sets nor resets the chain — it
/// is a no-op for B2B, matching guideline behaviour).
fn update_b2b(&mut self, outcome: &LockOutcome, t_spin: Option<TSpinKind>) {
    let lines = outcome.cleared_rows.len();
    if lines == 0 {
        return; // no clear: chain neither continues nor breaks
    }
    let eligible = matches!(
        (t_spin, lines),
        (Some(TSpinKind::Full), _) | (Some(TSpinKind::Mini), _) | (None, 4)
    );
    self.b2b = eligible;
}
```

> This mirrors `compute_reward`'s `b2b_eligible` categories (`eval/mod.rs:171-184`):
> Tetris and any full/mini T-spin **line clear** are eligible; singles/doubles/triples
> without a spin break the chain. Keeping the categories in lockstep is a
> [confirmed medium issue]; the cheap fix is to define them once and reference from
> both. v1 may duplicate the match (it is three arms) with a `// keep in sync with
> compute_reward` comment; consolidating into one shared `is_b2b_eligible(t_spin, lines)`
> helper is the preferred follow-up.

> **Note on the reward double-count.** `compute_reward` *already* adds `b2b_clear` to
> every eligible clear's reward (it does not gate on the prior `b2b` flag — see its
> doc comment, eval/mod.rs:128-134). So the beam's per-move reward is unchanged; the
> `update_b2b` flag exists so a future evaluator that *does* read `state.b2b` (e.g. a
> feature) sees a faithful chain, and so the value net can be fed a correct b2b bit if
> training adds one. v1 does not change the reward math — it only keeps the flag
> honest along a branch.

---

## 7. The batch-evaluation seam: `Evaluator::evaluate_batch`

A multi-ply beam scores a whole generation of children at once. For the linear
evaluator this is a loop; for the NN it must be **one batched forward** over an
`[N, NUM_FEATURES]` tensor, or the per-call tensor allocation (one tiny matmul per
placement) dominates. The seam is a defaulted trait method, so it is backward
compatible and object-safe.

```rust
pub trait Evaluator: Send + Sync {
    fn evaluate(&self, lock: &LockOutcome, board: &Board, t_spin: Option<TSpinKind>)
        -> (Value, Reward);

    /// Score a batch of placement results in one shot. Input order is preserved in
    /// the output (`out[i]` scores `inputs[i]`). The default loops `evaluate`, so
    /// every existing evaluator works unchanged; a batched backend (the NN) overrides
    /// it with a single forward pass.
    fn evaluate_batch(
        &self,
        inputs: &[(&LockOutcome, &Board, Option<TSpinKind>)],
    ) -> Vec<(Value, Reward)> {
        inputs
            .iter()
            .map(|(lock, board, t)| self.evaluate(lock, board, *t))
            .collect()
    }
}
```

- **`LinearEvaluator` keeps the default** — the feature dot-product is already cheap,
  no backend gain from batching, and the loop is identical to today's per-placement path.
- **`BurnEvaluator` overrides** it: stack the `N` `features_to_input` vectors into one
  `[N, NUM_FEATURES]` `TensorData`, `forward` once → `[N, 1]`, read the column to a
  `Vec<f32>`, and zip with `compute_reward(reward_weights, lock, board, t)` computed
  per item (the reward half is integer board math, not a tensor op). This also fixes the
  flagged `[1, NUM_FEATURES]` reshape hazard — the batched path builds the 2-D tensor
  directly at the true batch size.
- **Object-safety:** the input is a slice of borrows `(&LockOutcome, &Board,
  Option<TSpinKind>)` so the trait stays `&dyn Evaluator`-usable (no generics on the
  method). The beam owns the `LockOutcome`s and `Board`s for the generation and passes
  borrows.

> **Determinism of the batched path.** The CPU (`ndarray`) backend is deterministic,
> and `evaluate_batch` must produce **bit-identical** results to `evaluate` called
> per item (a batched matmul of independent rows is the same arithmetic). STEP 1 pins
> this with a test: for a handful of boards, `evaluate_batch(&[...])` equals
> `[evaluate(a), evaluate(b), ...]` exactly. This guards the "score holds the same in
> batch vs scalar" concern.

---

## 8. Scoring, restated against the existing `score_placement`

The beam must score **identically** to `score_placement` (search/mod.rs:107) so
beam@depth-1 == greedy. The equivalence, made explicit:

```text
score_placement(board, p, eval):
    board' := board.clone()
    t      := classify_t_spin(&p.piece, &board')        // pre-lock
    lock   := lock_and_clear(&p.piece, &mut board')
    (v, r) := eval.evaluate(&lock, &board', t)
    return (v + r).0
```

The beam's depth-1 child does the same, but via `commit_placement` (which clones the
state, classifies pre-lock, locks `p.piece`, advances) and then `evaluate_batch`. The
single batched call with one element must equal `evaluate`. **Therefore beam@depth-1
and greedy pick the same placement** — the bench asserts this before depth rises.

For depth > 1, the leaf score is `(leaf_value + Σ path_rewards).0` — the Cold Clear
value-with-folded-reward, summed over the branch and met at the leaf. This is the
design the `eval` module docs (lines 11–16) promised and the reason `Reward: Add` and
`Value: Add<Reward>` exist.

---

## 9. Test matrix (what each step must pin)

- `commit_placement` (STEP 0): a `used_hold=false` placement matches `commit`; a
  `used_hold=true` placement moves the old active into `hold`, makes `placement.piece`
  active-and-locked, and (empty-hold case) consumes the queue front; bag is dealt
  exactly once (assert `BagState` equals a hand-rolled expected after a hold + a normal
  commit); b2b transitions (Tetris sets, single clears, no-clear preserves).
- `evaluate_batch` (STEP 1): default loop equals per-item `evaluate`; `BurnEvaluator`
  batched equals per-item; order preserved; empty input → empty output.
- `BeamPlanner` (STEP 2): determinism (same state twice → same plan); beam@depth-1 ==
  greedy on the Tetris-well fixture and on a random snapshot; `Done(None)` on a topped
  board; `NeedMoreBudget` then `Done` across calls for depth ≥ 2; speculation path
  triggers only with an empty queue and stays deterministic.
- Bench (STEP 2/3): `bench-marathon` prints score/sec for greedy vs beam(linear) vs
  beam(NN); beam@depth-1(linear) must match greedy's score/sec within noise (it is the
  same decisions); deeper beam is the experiment.

---

## 10. Out of scope for v1 (documented, deferred)

- Transposition de-duplication / node hashing (the medium "canonical total order"
  issue) — beam truncation + canonical enumeration already give determinism; a
  transposition table is a *speed/quality* optimization for later.
- Probability-weighted / expectimax bag speculation (v2 behind the same node shape).
- Time-based `SearchBudget` and a dedicated cooperative WASM runner — the
  `NeedMoreBudget` seam suffices for native; the WASM runner is a separate task.
- Search-tree introspection (frontier size, pruned counts) — additive, not required to
  beat the bench.
- Bidirectional soft-drop / movegen ordering refinements — movegen is unchanged.

The line that matters: **STEP 0 unblocks the transition, STEP 1 unblocks the NN batch,
STEP 2 lands a deterministic beam that ties greedy at depth 1 and is gated on the
bench, STEP 3 swaps the NN in.** Everything else is re-use.
