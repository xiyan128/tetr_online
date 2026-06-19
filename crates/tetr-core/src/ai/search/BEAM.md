# BEAM.md — Deterministic beam search behind the `Mind` trait

> Status: **shipped — design record, maintained**. This was the contract the Colder
> Clear strike implemented; `beam.rs` is the implementation and, where the two
> disagree, **the code is the source of truth** (this file records the design
> rationale and the load-bearing invariants, and is updated when they change).
> Scope: a CC2-style **deterministic beam** Tier-2 planner behind the existing
> [`Mind`] trait, gated on `bench-marathon`. Notable post-design evolution folded
> in below: the search board is now the `Copy` `BitBoard` (locking via
> `BitBoard::lock_piece`, the differential-tested mirror of `lock_and_clear`), the
> evaluator seam takes a `ColumnView` + `EvalContext`, the queue is a `SmallVec`,
> and the neural value net this design anticipated was later pruned
> (`docs/value-net-postmortem.md`).
>
> **2026-06-10 addendum — the `Mind` seam.** The `Planner`/`PlannerStep` step-API
> this record was written against became the anytime session trait `Mind`
> (`reroot` / `think(quantum)` / `best` / `nodes_expanded`); see
> `docs/adr-ai-compute-architecture.md` for the layering and rationale. Mapping
> for readers of the sections below: `think(quantum)` now processes up to that
> many parent frontier nodes, reporting `Working`/`Exhausted`; seeding moved into
> `reroot` (which fingerprints the root + depth cap and makes `best()` immediately
> valid); the drain-loop became `Policy::decide` over the verbs; and the
> cooperative venue this design anticipated exists (`SlicedRunner`). Every
> determinism/back-up/speculation invariant recorded
> below is unchanged and still pinned by the same tests.

---

## 0. What stays the same (and why)

The whole point of the `(Value, Reward)` / `Planner` / `Evaluator` seam is that a
multi-ply search is a **pure backend swap**. Nothing below the search changes:

| Layer | File | Why unchanged |
|---|---|---|
| Engine rules | `crates/tetr-core/src/engine/*` | **ADR-7 boundary.** The search clones a `SearchState` and locks through `BitBoard::lock_piece` — the bitboard mirror of [`lock_and_clear`], differential-tested against it — so it never re-encodes rules. |
| Controller | `crates/tetr-core/src/ai/controller.rs` | Holds a `Box<dyn Policy>`; never names a planner. |
| `SearchPolicy` | `crates/tetr-core/src/ai/policy/search.rs` | Drives `Mind::think(quantum)` until `PolicyProgress::Ready`. The imperfection softmax (`score_candidates`) re-scores ply-1 placements with the same evaluator — unchanged by beam internals. |
| `plan.rs` (plan→input) | `crates/tetr-core/src/ai/plan.rs` | Consumes a `Placement.path`. The beam returns a **ply-1** `Placement` whose path is exactly what movegen produced (incl. a leading `Move::Hold`). Identical shape to greedy's output. |
| `DecisionRunner` / `SyncRunner` | `crates/tetr-core/src/ai/runner/mod.rs` | `submit`/`poll`. The cooperative runner supplies the per-poll node quantum; the sync runner drains the same `Mind` to completion. |
| Movegen | `crates/tetr-core/src/ai/movegen.rs` | [`generate_with_hold`] already emits hold-aware placements. The beam only *consumes* it. |

The beam is **purely additive**: one new method on `SearchState`
(`commit_placement`), the evaluator fast paths it consumes, and one new planner
(`BeamPlanner`). Everything else is a re-use.

---

## 1. Determinism rule (non-negotiable)

The [`Mind`] contract (`search/mod.rs` docs) promises a planner is deterministic for
the same `(state, evaluator, budget/work)`: **no RNG, no clock**. The beam upholds
this:

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
    /// The branch's speculative reward discount (§5): `1.0` on concrete branches,
    /// multiplied by `SPEC_DECAY` at each speculative expansion.
    spec_weight: f32,
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

This is the **prerequisite** the whole strike rested on. `commit` and
`commit_with_next` lock the active piece and pull the next piece, but **do not model
a hold swap** — while [`generate_with_hold`] *does* emit `used_hold` placements. So a
multi-ply search could *enumerate* hold moves but not *transition* through them.
`commit_placement` (and its speculative sibling `commit_placement_with_next`) closed
that gap.

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

### Reference implementation (as shipped — see `state.rs` for the real thing)

```rust
pub fn commit_placement(&mut self, placement: &Placement) -> LockOutcome {
    if placement.used_hold {
        // Fund the swap: an empty hold pulls the queue front (engine hold rule).
        // The swapped-in piece is whatever `from_snapshot`/movegen put in hold or at
        // the queue front; we consume that slot symmetrically with the swap.
        let displaced = self.active.piece_type();
        if self.hold.is_none() {
            // The swapped-in piece came from the queue front; consume it.
            // (The queue is a SmallVec: `deal_from_queue` is the empty-safe `remove(0)`.)
            self.deal_from_queue();
        }
        self.hold = Some(displaced);
    }

    // Classify the T-spin against the PRE-lock board (engine order), then lock the
    // piece the placement actually rests as: the swapped-in piece when used_hold,
    // else the current active. Both are `placement.piece` for a movegen placement
    // of the current state, so we lock that uniformly — via `BitBoard::lock_piece`,
    // the bitboard mirror of the engine's `lock_and_clear`.
    let t_spin = classify_t_spin(&placement.piece, &self.board);
    let outcome = self.board.lock_piece(&placement.piece);

    // Update the branch-local B2B / combo chains from this clear (see §6).
    self.update_b2b(&outcome, t_spin);
    self.update_combo(&outcome);

    // Advance to the next piece for the following ply.
    if let Some(next) = self.deal_from_queue() {
        self.spawn(next);
    }
    outcome
}
```

> **Implementation note (T-spin at commit time).** The B2B update needs the T-spin
> classification of *this* placement, which the engine computes **against the
> pre-lock board** ([`classify_t_spin`]). `commit_placement` therefore classifies
> `placement.piece` against `self.board` **before** the lock mutates it, exactly as
> the shared `commit_child` helper does. See §6 for the exact b2b rule.

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

`BeamPlanner::think` processes up to the caller's node quantum of parent frontier
nodes, then yields. A generation may therefore span many calls on the interactive
runner. Children are scored as their parent is processed, but the next frontier and
its backed-up `root_best` are published only after the entire generation has been
consumed. This keeps the beam's generation semantics intact while making the
browser-sized quantum a real frame-budget bound.

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
    /// A partially expanded generation, when `think()` yielded mid-generation.
    generation: Option<GenerationWork>,
    /// Plies expanded so far (root seeding = depth 1).
    depth: u8,
    /// Identity of the state this run was seeded from, to detect a stale run
    /// (the shared `RootKey`, also best-first's transposition-key core).
    root_key: RootKey,
}
```

### First call (seed depth 1)

```text
roots := generate_with_hold(state.board, state.active, state.hold,
                            state.queue.front(), spawn_for)   // canonical order
if roots is empty: return Done(None)                          // topped out

pending := []
for (i, p) in roots.enumerate():
    (child, lock, t) := commit_child(state, &p)   // fork, classify pre-lock, commit
    pending.push(PendingChild { state: child, lock, t_spin: t,
                                ctx: decision point's (combo, b2b),
                                root_index: i, parent_acc: Reward(0), spec_weight: 1.0 })
score `pending` and fold into the frontier                // shared with later
                                                          // generations: back-up
                                                          // root_best, rank desc,
                                                          // truncate to width
depth := 1
if depth >= budget.max_depth OR frontier empty:
    return Exhausted                                      // single-ply == greedy
else:
    return Working
```

### Subsequent calls (expand up to the quantum)

```text
if no generation is in progress:
    parents := take(frontier)       // completed prior generation, canonical order
    staged_root_best := root_best
    staged_nodes := []
    staged_ranked := []

while spent < quantum AND parents remain:
    parent := parents[next_parent]
    children := generate_with_hold(parent.state.board, parent.state.active,
                                   parent.state.hold, parent.state.queue.front(), spawn_for)
    for p in children:              // canonical movegen order
        child := parent.state.clone()
        t     := t_spin(p) against parent.state.board (pre-lock)
        lock  := child.commit_placement(&p)
        pending.push(PendingChild { state: child, lock, t_spin: t,
                                    ctx: parent's pre-placement (combo, b2b),
                                    root_index, parent_acc, spec_weight })

    // §5: if speculate AND parent.queue is empty, also branch over the bag here
    //     (see "Speculation" — same shape, commit_placement_with_next, decay).
    // Dead/terminal parents consume one node of quantum and produce no children.

    scores := evaluate_cols for each pending child
    for (p, (value, reward)) in zip(pending, scores):
        weighted := reward scaled by p.spec_weight       // identity on concrete branches
        acc   := p.parent_acc + weighted                 // Reward + Reward
        score := (value + acc).0                         // Value + Reward
        staged_nodes.push(BeamNode { state, acc_reward: acc, root_index, spec_weight })
        staged_ranked.push((score, staged_nodes index))
        staged_root_best[root_index] := max(staged_root_best[root_index], score)

if parents remain:
    store the generation and return Working

sort staged_ranked by (score desc, enumeration index asc)
gather the first beam_width nodes, optionally skipping duplicate transposition keys
root_best := staged_root_best
frontier := gathered nodes
depth += 1
return Exhausted if depth >= budget.max_depth OR frontier empty, else Working
```

### Budget semantics

- `budget.max_depth` caps plies; **depth 1 reproduces greedy exactly** (one generation,
  one batch, pick the best root). The bench must show beam@depth-1 == greedy to prove
  the seam is faithful before depth is raised.
- `SearchBudget::beam(depth)` leaves `budget.nodes` uncapped: total beam work is
  bounded by `beam_width × depth`, while the runner's per-call `quantum` controls
  how many parent frontier nodes are expanded before yielding.
- `SearchBudget::beam(depth)` sits beside `greedy()` and `best_first(nodes, depth)`;
  the beam *width* is a `BeamPlanner` field, not part of the budget.

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
([`BagState`]) carries — exported by the engine snapshot (the generator's own
remainder; exact, not a reconstruction) and advanced along the branch by each
speculative deal. The beam speculates over it **deterministically**:

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

`SearchState.b2b` is carried per node and transitioned on every commit. Multi-ply
reward correctness needs it, and the chain also feeds the evaluator via
`EvalContext`. The transition delegates to the **engine's own predicates** — the
exact rule `ScoreState::lock_result` applies — so the search mirror can never drift
from real play:

```rust
/// Update the branch-local Back-to-Back flag from the clear that just happened —
/// the engine's exact transition: a qualifying clear (Tetris, full T-spin 1-3,
/// mini T-spin *single*) sets the chain, a plain 1-3 line clear breaks it, and
/// anything else (no clear, or a non-qualifying spin clear such as a mini double)
/// preserves it.
fn update_b2b(&mut self, outcome: &LockOutcome, t_spin: Option<TSpinKind>) {
    let lines = outcome.cleared_rows.len();
    if qualifies_for_back_to_back(t_spin, lines) {
        self.b2b = true;
    } else if breaks_back_to_back(t_spin, lines) {
        self.b2b = false;
    }
}
```

> The original design duplicated `compute_reward`'s `b2b_eligible` match here with a
> "keep in sync" comment; that flagged desync risk was since resolved by routing the
> transition (and `compute_reward`'s attack-continuation test) through the engine's
> `qualifies_for_back_to_back` / `breaks_back_to_back`, the single source of truth.
> `compute_reward`'s *abstract bonus table* (which clears earn the `b2b_clear`
> weight) remains a reward-policy choice local to that function.

> **Note on the abstract B2B bonus.** `compute_reward` adds `b2b_clear` to every
> bonus-eligible clear's reward regardless of the prior chain (see its doc comment) —
> that part is unchanged. The faithful per-branch flag is consumed for real now: the
> search threads it (with combo) into every evaluation as `EvalContext`, where the
> CC2 evaluator's `has_back_to_back` value term and `compute_reward`'s engine-exact
> attack term read it.

---

## 7. The evaluation seam

A multi-ply beam scores each child through the evaluator's bitboard fast path
(`evaluate_cols`) while staging the current generation. The original design also
kept an object-safe `evaluate_batch` seam for a batched backend; the neural value
net that motivated that seam was later pruned, but the determinism rule for any
future batched implementation remains the same. As shipped (`eval/mod.rs` — the
evolved signature, with the bitboard `ColumnView` and the `EvalContext` chain state
this design predated):

```rust
pub trait Evaluator: Send + Sync {
    fn evaluate(&self, lock: &LockOutcome, board: &Board, t_spin: Option<TSpinKind>,
        ctx: EvalContext) -> (Value, Reward);

    /// Bitboard fast path: score a placement whose resulting board is a ColumnView.
    /// Default reconstructs a dense Board and defers to `evaluate`; both shipped
    /// evaluators override it to read the columns directly (bit-identical, pinned
    /// by per-impl differential tests).
    fn evaluate_cols(&self, lock: &LockOutcome, board: ColumnView,
        t_spin: Option<TSpinKind>, ctx: EvalContext) -> (Value, Reward) { ... }

    /// Score a batch in one shot. Input order is preserved in the output
    /// (`out[i]` scores `inputs[i]`). The default loops `evaluate_cols`; a batched
    /// backend overrides it with a single forward pass.
    fn evaluate_batch(
        &self,
        inputs: &[(&LockOutcome, ColumnView, Option<TSpinKind>, EvalContext)],
    ) -> Vec<(Value, Reward)> { ... }
}
```

- **Object-safety:** the input is a slice of borrows/`Copy` views, so the trait stays
  `&dyn Evaluator`-usable (no generics on the method). The beam owns the generation's
  `PendingChild`ren and passes borrows of their locks and board views.
- **History:** the pruned `BurnEvaluator` overrode `evaluate_batch` to stack the
  feature vectors into one `[N, NUM_FEATURES]` tensor and forward once — the seam
  this method exists for. It remains the integration point for any future batched
  backend (see `docs/value-net-postmortem.md`).

> **Determinism of the batched path.** `evaluate_batch` must produce **bit-identical**
> results to mapping `evaluate_cols` over the same inputs, and `evaluate_cols` must be
> bit-identical to `evaluate` on the equivalent dense board. Both halves are pinned by
> tests (`default_batch_matches_scalar`, plus the per-impl `*_matches_evaluate`
> differentials). This guards the "score holds the same in batch vs scalar" concern.

---

## 8. Scoring, restated against the existing `score_placement`

The beam must score **identically** to `score_placement` (search/mod.rs) so
beam@depth-1 == greedy. The equivalence, made explicit (both run through the shared
`commit_child` helper — fork, classify pre-lock, `commit_placement` — so they cannot
diverge structurally):

```text
score_placement(state, p, eval, ctx):        // greedy / imperfection path
    (child, lock, t) := commit_child(state, p)
    (v, r) := eval.evaluate_cols(&lock, child.board.view(), t, ctx)
    return (v + r).0
```

The beam's depth-1 child builds the same `commit_child` output into a `PendingChild`
and scores it through `evaluate_cols`, the same evaluator basis as the scalar path.
**Therefore beam@depth-1 and greedy pick the same placement** — the tests assert
this before depth rises.

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
- `Evaluator` (STEP 1): each impl's `evaluate_cols` equals `evaluate` on the
  equivalent dense board (the per-impl differential); `evaluate_batch`, where used,
  must preserve order and match per-item scoring.
- `BeamPlanner` (STEP 2): determinism (same state twice → same plan); beam@depth-1 ==
  greedy on the Tetris-well fixture and on a random snapshot; `Done(None)` on a topped
  board; quantum-sized calls preserve the same final decision as a one-shot drain;
  speculation path triggers only with an empty queue and stays deterministic.
- Bench (STEP 2/3): `bench-marathon` prints score/sec for greedy vs beam contenders;
  beam@depth-1(linear) must match greedy's score/sec within noise (it is the same
  decisions); deeper beam is the experiment.

---

## 10. Out of scope for v1 (documented, deferred)

- Transposition de-duplication is now available as the opt-in
  `BeamPlanner::transposing` path: per-root future-state keys collapse equal states
  before width truncation. Plain `BeamPlanner::new` remains the historical baseline.
- Probability-weighted / expectimax bag speculation (v2 behind the same node shape).
- Time-based `SearchBudget` remains out of scope; the shipped cooperative runner is
  node-quantum based for deterministic native/wasm behavior.
- Search-tree introspection (frontier size, pruned counts) — additive, not required to
  beat the bench.
- Bidirectional soft-drop / movegen ordering refinements — movegen is unchanged.

The line that mattered: **STEP 0 unblocked the transition, STEP 1 the batch seam,
STEP 2 landed a deterministic beam that ties greedy at depth 1, gated on the bench.**
(STEP 3 — swapping the neural value net in — shipped and was later pruned for quality
reasons; `docs/value-net-postmortem.md` is the record.) Everything else is re-use.
