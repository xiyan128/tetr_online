# Learned Evaluation: a roadmap to a calibrated value function

**Status:** direction-setting. Phase A is built and its learning signal is in (val R² ≈ 0.64); the
strength gate is blocked on inference cost (§5). The rest is the plan. This is where we are steering
the AI. §4 records the design decisions settled in the working sessions; §5 the infrastructure
lessons from the prototype.

---

## 1. The vision

Replace the handcrafted leaf score with a **calibrated value function**

$$V(s) \;\approx\; \mathbb{E}\!\left[\textstyle\sum_t \gamma^t R_t \;\middle|\; s\right]$$

in the engine's **reward units** — the expected discounted future return (attack + survival) from
a position. This is the AlphaZero insight (a learned value behind a search), adapted to *our*
engine: we keep the deterministic beam and the engine's **exact** reward, and learn only the leaf
value.

The point is **not a better board score.** Board scoring is saturated — the handcrafted CC2 eval
is at a local optimum (every single-lever probe tied or lost), and the cheap value-net Phase-0 gate
(E8) leaned STOP on the question "does a learned board *score* beat CC2's?". The point is that a
value in *reward units* is a structurally different object that **CC2 can never be**, and it unlocks
a class of search that is otherwise impossible:

| Capability | Heuristic (CC2) | Calibrated V |
|---|---|---|
| Compose path rewards + leaf (`Σγ^t R_t + γ^d V`) | ✗ incompatible scales | ✓ the Bellman equation |
| Prune dead branches early | ✗ can't tell "dead" from "low score" | ✓ `V_srv < ε` ⇒ prune |
| Alpha-beta-style cutoffs in the beam | ✗ scores not comparable across depths | ✓ calibrated returns compare |
| Allocate compute by decision criticality | ✗ no notion of a "gap" | ✓ gap between `V(best)`, `V(2nd)` |
| Plan multi-piece attack timing | ✗ no future-attack signal | ✓ discounted `V_atk` encodes cadence |

These are the real prize. They are also exactly the **adaptive-search** levers our own scaling work
pointed at (compute is mis-allocated; the game is decided by the rare hard move). A calibrated V is
what makes them *principled* instead of heuristic.

### Why this survives the E8 "STOP"

E8 asked whether a learned *static board score* beats CC2's, and leaned STOP. That does **not** kill
this vision, because:
1. The vision's value is **composability + adaptive search**, not board scoring — a dimension E8
   never tested. A value in reward units enables search CC2 structurally cannot do.
2. E8's gate was an **aggregate** regression; the headroom we care about is concentrated in the rare
   **hard positions** that decide games, which the average washes out.

Both are bets, and both are **gated below**. If the gates fire, the honest answer is "push the
current paradigm deeper," and we take it.

---

## 2. Principles (non-negotiable)

- **Keep the engine's exact reward.** Attack is exact, deterministic engine math. Never learn what
  you can compute. We learn the *value*, the reward stays the engine's.
- **Truly raw input.** The net sees the raw board + raw piece/bag/chain/garbage state — not
  handcrafted Dellacherie features (that is a fancy DT-20 and caps the ceiling at what the features
  expose).
- **Every phase has a hard gate and a STOP.** The iron gate — *match CC2 at iso-search before adding
  any depth or unlock* — applies throughout: deep search amplifies a bad eval as readily as a good
  one.
- **Determinism for ship.** Research is `f32`; anything that ships is fixed-point `i32`,
  bit-identical across native/wasm, zero ML deps.
- **Cheap before expensive.** BC-distillation before self-play RL; one afternoon of gating before
  any multi-week front.

What we explicitly **reject** from the maximal blueprint: self-play-RL as the *first* teacher (it is
the last phase, not the first); "no CC2 anywhere" purity (we keep the exact reward); a handcrafted-
feature NNUE masquerading as raw-input; and 6-week timelines for what is a multi-month program.

---

## 3. The roadmap

Each phase is a deliverable + a hard gate. Do not start a phase before the prior gate is green.

### Phase A — Distill the value, and gate it  ·  *(built; learning signal in, strength gate blocked on cost)*

Behavior-clone the **existing champion** (`tp128d9`, ~0.82 APP) into a raw-input net: label every
position with the champion's *own deep-search value*. No RL, no MCTS — the cheapest possible teacher.

- **Built:** the raw-obs encoder (§5), the `bc-distill` exporter, a PyTorch board-CNN trainer, the
  `evaluate_state` seam, the verified `f32` Rust inference, and the `value-gate` iron gate.
- **Results:** a 2M-param CNN predicts the teacher's value at **val R² ≈ 0.64** on held-out games
  (clean shard-level split) — it genuinely learns the teacher's judgment from raw input, refuting
  the DT-20 ceiling. Rust inference matches PyTorch to **1.3e-6** (golden cross-check).
- **GATE A:** `value-gate` — net-leaf beam ≥ CC2-leaf beam at iso-search, rain GSPRT, held-out seeds.
  **Blocked on inference cost** (§5): a per-node `f32` CNN is ~1000× CC2, so the `w16d6` gate can't
  finish a game. Read the first eval-quality signal at **depth 1** (no search expansion); the real
  `w16d6` gate needs batched eval or the NNUE student first.
- *If it loses,* distilling a heuristic *score* isn't enough → Phase B (calibrated returns) before
  concluding STOP. (Note: Phase A's target is the teacher's CC2 *score* — uncalibrated. The
  calibrated V the vision needs is Phase B; A validates the pipeline and shows raw input *can* learn.)

### Phase B — Calibrate to return units  ·  *the crux of the vision*

Re-target from the teacher's heuristic *score* to **realized discounted returns** — two heads, one
trunk, in real reward units (see §4 for the full reward discussion):

- **`V_atk`** = discounted future attack; reward = the canonical **`attack_lines`** (engine
  guideline attack, in lines).
- **`V_srv`** = discounted survival; reward = a **terminal-death** signal (the one reward we must
  *define* — there is no canonical survival reward). `V(s) = V_atk + λ·V_srv` (λ in §4).

**Calibration is Monte-Carlo *policy evaluation*, not self-play.** Play the champion (a fixed strong
policy); the realized `G_t = Σγ^k R_{t+k}` along its trajectories *is* the target → `V^champion ≈ V*`,
no improvement loop. The off-policy gap (`V^champion ≠ V^your-search`) is what self-play (Phase F)
closes; for a leaf eval, `≈V*` on the right distribution is enough.

**Coverage = DAgger, not ad-hoc exploration.** A deterministic strong policy visits a *narrow,
healthy* slice (why our first dataset came back `death% 0`) — the covariate shift BC suffers. DAgger
fixes it the principled way: **roll out the net-bot, label the states *it* visits with the champion**,
aggregate, retrain. Strong returns (target ≈ V*) on student-visited states (coverage), fixed expert
(no self-play, with regret guarantees BC lacks). Layer heavy rain (period ≈ 2, ~30% top-out) for
danger.

- **GATE B:** (1) V is *calibrated* — bucketed `E[V] ≈ E[realized return]`, `V_srv` sharply negative
  on tall/holey boards; and (2) the calibrated-V-leaf bot ≥ CC2 at iso-search. Calibration is the
  prerequisite for every unlock below.

### Phase C — Compose  ·  *the first unlock*

With V in reward units, score a path the *correct* way:
`score = R₁ + γR₂ + … + γ^{d-1}R_d + γ^d · V(leaf)` — the engine's exact path rewards plus the
calibrated leaf. This is the Bellman composition CC2 structurally cannot do, and the single biggest
versus unlock (it can value "spike now, mediocre board" vs "no attack, perfect setup").

- **GATE C:** the composing bot beats the value-only-leaf bot (Phase B) in versus GSPRT.

### Phase D — Adaptive & principled search  ·  *the big payoff*

Layer the calibrated-V unlocks, each behind its own GSPRT gate and STOP:
- **D1 Variable-depth:** prune dead branches (`V_srv < ε`), accept confirmed spikes early.
- **D2 Beam alpha-beta:** calibrated upper-bounds make cross-depth cutoffs sound.
- **D3 Criticality allocation:** spend the per-frame node budget by the `V`-gap between the best
  moves — minimal compute on obvious placements, deep on game-deciding ones. *This is the
  highest-ROI lever our scaling work identified* (uniform compute on non-uniform difficulty), and
  the calibrated V is what makes it principled rather than a heuristic threshold.

### Phase E — Ship: the NNUE student + wasm

Distill the research-grade `f32` CNN teacher into a fast, deterministic **integer NNUE** for the
guideline budget. Use the **dual-head shared-accumulator** design (one incrementally-maintained
accumulator; a cheap linear *fast head* for interior nodes, a small MLP *deep head* for leaves) —
two eval tiers for ~one accumulator's cost. Hand-rolled `i32`, no deps.

- *Caveat to validate, not assume:* the incremental accumulator is only cheap on non-clearing
  placements; clears/T-spins/garbage force a recompute. Measure the real per-node cost before
  committing the design.
- **GATE E:** student matches the teacher within a small value-MSE *and* fits the wasm size +
  per-frame latency budget. (Research strength can ship via the deferred-worker venue if the
  per-node cost is too high for the 60 Hz path.)

### Phase F — Self-play to exceed the teacher  ·  *the AlphaZero loop, LAST*

Only once a calibrated V + composing/adaptive search beats CC2: close the loop. Self-play with the
V-guided search generates data stronger than the champion → retrain → exceed it; gate each
generation head-to-head (≥55%). This is the expensive RL front — deliberately last, and only if
A–D have paid.

---

## 4. Design decisions (settled in the working sessions)

Choices that are easy to get wrong, recorded so they aren't re-litigated.

### Reward — what's canonical, what isn't

- **Canonical (use it):** `attack_lines(action, b2b, combo, perfect_clear)` (`engine/attack.rs`) is
  the exact guideline attack in *lines* — the *same* function the engine sends garbage with.
  `EngineEvent::AttackSent { net }` is its post-cancellation realization. This is `R_atk`.
- **Not canonical (must define):** there is **no** per-step survival or win reward — top-out is
  terminal, `decide_versus` gives a terminal win/loss. `V_srv` needs a *chosen* signal (we pick the
  terminal-death form: 0/step, a large negative at top-out, discounted). And **CC2's `Reward` is a
  heuristic** (re-weighted `attack_lines` in SCALE units) — derived from the canonical reward but not
  it; never train V on it.

### A single-board V is structurally a proxy

Win/loss is a **two-board** quantity (your board *and* the opponent's). Our value sees only *your*
`SearchState`, so a single-board V trained on win/loss predicts `E[win | your board]` with the
opponent's board as a huge unobserved confounder (most of the variance, unlearnable). Consequences:

- Pre-F reward is **necessarily a proxy** — attack + survival, the two channels your board affects
  the outcome through. The split is *not* arbitrary: **E8 showed `V_atk` alone is confounded** (under
  rain, attack tracks downstacking → an attack-only value rewards height → suicidal); **`V_srv` is the
  counterweight.**
- A true win/loss (Elo-aligned) reward waits on **both** self-play *and* a two-board (joint-state)
  value — the two-agent search, not Phase F alone.

### Reward vs objective: freeze the *gate*, not the reward

- **Elo is the objective; win/loss is its frozen, un-gameable, Elo-aligned form** — but it needs a
  versus pool, so it lives in self-play.
- A **frozen *proxy* reward is gameable** (the gameable-APP lesson; E8's downstacking confound is the
  same divergence in miniature). The proxy is the *training signal*, never the acceptance test.
- So: **freeze the gate (Elo / rain-decisive GSPRT); let the reward be the cheapest sufficient
  scaffold per phase,** converging to win/loss as self-play comes online.

### λ — the `V_atk`/`V_srv` mix

`V = V_atk + λ·V_srv` is the offense/defense dial — sensitive for *strength*, but **not a training
commitment**: two separate heads make λ a pure **inference-time knob** (no retraining). So:

- **Normalize the heads** to comparable variance first (lines vs a probability are different units —
  a bare "λ = 5" is meaningless), then **sweep λ against the Elo gate** (tune on TRAIN seeds, gate on
  HOLDOUT).
- **Static λ first.** Dynamic λ-by-match-state is a *handcrafted controller* patching the single-board
  V's blindness to the opponent — reach for it only if static plateaus.
- λ **vanishes** at the endgame: a two-board win-probability value integrates offense/defense per
  state, so there's nothing to mix.

---

## 5. Infrastructure & engineering notes (from building Phase A)

What the prototype taught us — the parts to keep and the traps to avoid.

### The inference-cost wall (the load-bearing one)

A per-node *dense* net is far too slow for the beam: an `f32` board-CNN is ~6 ms/eval, and `w16d6`
scores ~1,600 leaves per decision → **~10 s/decision, ~40 min/game** (≈1000× CC2). The `w16d6` iron
gate literally can't finish a game. Fixes, in order of leverage:

1. **Batched evaluation.** The beam already collects a whole generation's children
   (`score_pending_into`'s `pending: Vec<PendingChild>`); score them in **one matmul** instead of
   per-leaf. Batched + an amortized accumulator is 10–100×. This is the `evaluate_batch` seam the
   postmortem named but never built — build it.
2. **Depth-1 research gate.** For a *first* eval-quality read, race at depth 1 (greedy): no search
   expansion, ~30 evals/decision, feasible. It isolates "is the net's value a better ranker than CC2"
   from search speed.
3. **The NNUE student (Phase E).** Fixed-point `i32` + the incremental dual-head accumulator is the
   ship-speed answer (verify the incremental claim — clears/T-spins/garbage force recompute).
4. **Root-only policy prior.** If per-node value stays too costly, a root-only prior (~1 call/
   decision) is the wasm-affordable alternative.

### The integration seam: `evaluate_state`

The `Evaluator` trait only sees the board (`evaluate_cols`) — not queue/hold/bag/pending. A raw-input
net needs the **full `SearchState`**, available at the leaf (`score_child`). Add
`fn evaluate_state(state, lock, t_spin, ctx)` defaulting to `evaluate_cols` — **byte-identical** for
every handcrafted evaluator (verified: all 277 core tests pass), only the net overrides it. *The*
clean seam; don't widen `EvalContext` with depth/leaf flags.

### The encoder is the single source of truth

One `encode(&SearchState) -> (board plane, feature vector)` shared by **export and inference**, so the
train and serve distributions can't drift. Pin it bit-exact against an engine snapshot.

### Rust ↔ Python: safetensors + a golden cross-check

- **safetensors** is the interchange both languages speak natively (typed, zero-copy) — shards in,
  weights out. No bespoke binary, no JSON-for-tensors.
- **A hand-rolled forward pass WILL have layout bugs** (transposed dims, wrong flatten order). The
  PyTorch↔Rust **golden cross-check** is non-optional: export (input, output) pairs, assert the Rust
  forward matches (we hit 1.3e-6). Random inputs suffice — they exercise the same arithmetic.

### Data-generation traps

- **Write shards incrementally**, with a progress line — the first exporter buffered everything and
  wrote at the *end* (opaque, crash-unsafe; a 45-min run showed nothing until done).
- **Drive the distiller 1×, not 2×** — it searches twice per piece (the controller plays + a label
  search). Drive directly by the label's placement (`placement_to_inputs`) to halve it.
- **Teacher speed:** the champion `tp128d9` is slow under 2× search (~45 min / 256 games). The *depth*
  (d9) is what the value target needs; width is survival-robustness. Iterate the pipeline on a fast d9
  teacher (`tp16d9`), regenerate the final dataset with the champion.
- **Export the best-val checkpoint**, not the overfit final epoch (the CNN overfit: train R² 0.83,
  val peaked ~0.64 at epoch 14).

### Inference engine: candle for velocity, hand-rolled integer as the back-pocket optimization

**Training:** uv + PyTorch (the framework is free — pick the maintainable one).

**Inference — the decision.** The prototype hand-rolled the `f32` forward pass (no deps, but slow and
bug-prone — the golden cross-check exists only to catch the layout bugs hand-rolling invites). If
`f32` + a dep + the determinism trade are acceptable, **prefer `candle`**: it loads our safetensors
directly, has gemm-backed `Conv2d`/`Linear` (no SIMD to write), runs the **same code native and
wasm** (one inference path, no research↔ship divergence), and makes **batched evaluation trivial** (a
`[N,1,24,10]` forward is one call) — which is the structural 10–100× win, not the kernel. candle lets
us delete *both* the hand-rolled forward and the golden check. (Burn stays rejected — heavier dep
tree than candle for the same job, and it was the original prune.)

**The determinism trade (make it an informed call).** candle is `f32`, so:
- *Within a platform* it's deterministic — same-machine byte-identical replay still holds.
- *Cross-platform* (native research ↔ wasm ship, or different CPUs) float reduction order differs →
  leaf values drift ~1e-6 → a rare argmax flip on a near-tie. **Strength impact: negligible** (a few
  moves/game). **Reproducibility impact:** the ledger's "replay is bit-identical across machines"
  property is lost *for net bots*, and a research result becomes "approximately," not bit-exact, what
  ships. Acceptable **iff** we don't anchor reproducibility on the net's exact value and keep gating
  on **Elo** (GSPRT needs reproducibility-enough-to-race, not bit-determinism).

**The hand-rolled fixed-point `i32` NNUE stays in the back pocket** — pulled out only if (a) we need
max wasm throughput (int8 SIMD still beats `f32` candle) or (b) we want bit-exact cross-platform
replay back. It is *not* a prerequisite: candle-first for velocity, hand-roll later if the
perf/determinism bill comes due.

---

## 6. Phase summary

| Phase | Deliverable | Hard gate / STOP |
|---|---|---|
| **A** Distill + gate | champion → raw-input value net; `value-gate` | net ≥ CC2 at iso-search · *built; val R² 0.64; gate blocked on inference cost (§5)* |
| **B** Calibrate | V in return units (atk+srv), death coverage | V calibrated **and** ≥ CC2 at iso-search |
| **C** Compose | Bellman path + leaf | composing > value-only in versus |
| **D** Adaptive search | variable-depth, beam α-β, criticality alloc | each beats its predecessor (GSPRT) |
| **E** Ship | dual-head integer NNUE, wasm | matches teacher + fits budget |
| **F** Self-play | RL loop to exceed the champion | each generation ≥ 55% vs prior |

The discipline that makes this credible is the same one that has run the whole AI program: **predict,
then gate cheaply, and honor the STOP.** Phases A–B answer "is there a calibrated value worth
having?" for one afternoon of racing each. Only if they say yes do we spend on the unlocks (C–D),
the ship student (E), and the self-play loop (F). If they say no, we have lost an afternoon and we
push depth — exactly as designed.
