# Research directions

Principal-investigator roadmap for the tetr-core AI. This is a ranked, actionable
program: what to run, in what order, and what result advances or kills each bet.
It integrates the verified scaling data (`analysis/elo-pareto/elo.csv`), the
operator design, the value-net post-mortem (`docs/value-net-postmortem.md`), and
the compute/determinism constraints (`docs/adr-ai-compute-architecture.md`).

A note on grounding discipline used throughout: every quantitative verdict must
come from pair-GSPRT on death-decisive (versus-under-rain) or downstack seeds,
**never** from solo APP (gameable by combo farming) and **never** from
Bradley-Terry point estimates at the top of the grid (the top configs sit in
overlapping bootstrap CIs in `elo.csv`).

---

## 1. Executive summary

The current paradigm — a handcrafted CC2 evaluator behind a deterministic
node-budgeted beam — is **not yet exhausted on its cheapest axis: depth**. The
verified scaling law (`Elo ≈ 1411 − 1056·ms^−0.28`, R²=0.993; the width/depth
split `73·log₂w + 380·log₂d`, R²=0.948) says a node buys **~5.2× more Elo as
depth than as width**, yet the champion `w128d9` (972 nodes, 99.74 ms, 1128 Elo)
maxes depth at the *grid's* `d=9` wall and then spends every remaining node on
the 5×-weaker width lever. The `d=9` ceiling is a **grid choice, not an engine
limit** (`max_depth` is an unbounded `u8`, `search/mod.rs:71`), and per-ply value
is still **+20…+40 Elo at d7→d9** — so `~1411` is a *lower bound at d≤9*. The
single highest-leverage, zero-code experiment on the platform is therefore to
register a narrow-deep config (`w16d12`/`w16d15`) and race it against the
champion. The honest open risk is that the steep returns live in the **6 concrete
plies** (1 active + 5 preview) and flatten in the `SPEC_DECAY=0.75` speculative
tail past `d6`; that risk is exactly what the depth experiment adjudicates.

The next paradigm is **a learned leaf evaluator, not a learned planner**: this is
Chess-2014 (swap the leaf eval, keep the search), not Go-2016 (MCTS+policy, which
the SOTA plan correctly schedules last). The post-mortem's two blockers are now
removed — a **0.82-APP teacher** (the `tp128d9` champion's own search, replacing
DT-20's ~0.2) and a rain harness that **manufactures near-death coverage** — but
the iron gate stands: a learned eval must **match CC2 at iso-search before any
depth is added**, because deep search amplifies a bad eval as readily as a good
one (`bf+linear 0.05 < greedy+linear 0.20`).

The **policy-improvement-operator unification is a narrative, not a refactor.**
The `Mind` seam already *is* `π' = Improve(π₀, V, budget)`: all three planners
share `commit_child` / `score_child` / `hold_placements` / `best_root_plan` /
`RootKey` / `root_best`, and `SearchPolicy` already holds the
`(Mind, Evaluator, SearchBudget)` triple. Forcing a common trait removes zero
duplication and would have to wrap `PcCoverage`'s *option-coverage* objective
(which belongs as a hierarchical subgoal layer, not a peer). The framing earns
its keep on exactly **one free artifact**: every `Mind` computes `root_best` — a
dense backed-up Value over *every* ply-1 root — and `best()` discards all but the
argmax. Exposing it (`best_distribution()`, one default method) yields an
AlphaZero-style improved-policy target for free, the bridge from "push the beam"
to "close the learning loop."

---

## Results (executed)

Items 2.1–2.3 have been run. Raw output: `analysis/elo-pareto/{e1_race.log,f2_race.log}`;
runners: `crates/tetr-research/examples/{depth_probe,e1_depth_race,f2_champion_depth}.rs`.

**E0 (depth-stabilization probe).** Over 60 mid-game states the ply-1 decision is already
settled by the ~d6 preview horizon for **60–77%** of states, but still flips past d9 for
**10–18%** (p90 stabilization depth d10–11), and the value estimate keeps rising d9→d15. Not a
clean refutation → escalated to E1.

**E1 (pair-GSPRT, rain-decisive, arm-swapped + CRN).** The depth question resolves with nuance:
- **Depth past the d9 cap IS a real lever — but modest and fast-saturating.** w16d12 beats w16d9
  at **58.5%** and w24d12 beats w24d9 at **61.5%** (CIs exclude 50%) — the cap was a *grid*
  choice, and a few plies past it pay. But w16d15 ≈ w16d12 (**50.3%**, dead equal): depth
  saturates by **~d12**. Against the champion, depth ~doubles a narrow config's win-rate
  (w16d9 9.1% → w16d12 18.2%).
- **Depth does NOT substitute for width head-to-head.** Narrow-deep loses decisively to the wide
  champion: w16d12 **18.2%**, w24d12 **21.6%** vs w128d9. Width is a genuine **survival hedge**
  (one over-optimistic deep line that tops out is catastrophic) — the systems lens was right, and
  the earlier "the champion mis-allocates on width" reading was **wrong**. (Calibration: w16d9 vs
  champion = **9.1%**, confirming the test discriminates a ~200-Elo gap.)

**2.2 (interaction + regime refit, on-disk).** The "5.2× depth" headline is an average over a
sloped, *interacting* surface (cross term +38, R² 0.948→0.965; width buys 33 Elo/doubling at d2
but 116 at d9). Split by the preview regime: depth/width is **6.9× in the concrete horizon (d≤6)**
but **1.2× past it (d≥7)** — depth's dominance is a concrete-ply effect, which E0/E1 then confirmed
dynamically.

**2.3 (profile at narrow-deep).** The cost split shifts with shape: clone/memmove is only **9.5%**
at narrow-deep (vs ~40% on the wide champion) because cloning is *width-driven*. Deep-narrow bots
are bound by movegen (collision ~20% + the BFS-scratch TLS ~20%) and eval (~12%), not cloning — so
the clone-deferral lever is champion-specific.

**Net so far.** The current paradigm's cheapest real gain is to **lift the depth cap to ~12** (a
modest, saturating bump). Whether the *champion itself* gains from depth, and whether at the top
the extra compute is better as depth or width, is the F1/F2 follow-up *(running)*; §2.8's
best-first-vs-champion is F3. The headline correction stands: the ~1411 ceiling is **not** "we ran
out of depth" — depth saturates ~3 plies past the cap and cannot buy past width's survival role.

---

## 2. Push the current paradigm (ranked)

Lead with the cheapest, highest-confidence wins. Items 2.1–2.2 are zero-code
measurements that re-price the entire frontier; everything after is gated behind
their results.

### 2.1 Register narrow-deep configs past the `d9` grid wall — *do this first*

- **Mechanism.** Add `BotSpec::tp_beam(16,12)/(16,15)/(24,12).cc2(attack_tuned())`
  rows to the research grid. Zero tetr-core change: `max_depth` is an unbounded
  `u8` (`search/mod.rs:71`), and `tp_beam(256,12)` already builds.
- **Why it follows from the data.** The width/depth split (R²=0.948) explains 95%
  of variance vs 76% for total-nodes-alone, and the per-ply slope is *still
  positive* at d7→d9 (`w16d7=879.6 → w16d9=919.9`; `w128d7=1057.8 → w128d9=1127.8`
  in `elo.csv`). `w16d12 ≈ 176` nodes — less than 1/5 the champion's 972 — so a
  win is also a wasm-budget win.
- **Decisive experiment.** First the *cheap pre-test*: run one instrumented
  `w16d15` decision on mid-game seeds; dump (a) the ply at which `best()`'s argmax
  stops changing and (b) the speculative-vs-concrete leaf fraction. If the argmax
  is fixed by ~d6–d7 and >80% of d12+ leaves are `SPEC_DECAY`-discounted
  remnants, the deeper plies *cannot move the decision* — refuted in minutes.
  Only if the argmax keeps flipping past d7 do you escalate to the overnight
  `w16d12/w16d15/w24d12`-vs-`tp128d9` rain-decisive run, **confirmed by
  pair-GSPRT** (not Bradley-Terry). Pre-register the regime-split null: a
  speculation-subset fit predicts `w16d12 ≈ 914` (*below* `w16d9`), so a
  champion-matching result must clear that bar to count.
- **Expected payoff.** Asymmetric and mostly diagnostic. Either a narrow-deep
  config matches/beats the champion at <1/5 the nodes (the cheapest SOTA gain +
  the strong teacher every flywheel step needs), **or** a clean saturation knee
  that locates the true ceiling at the ~5-piece preview wall and redirects effort
  to the eval/speculation. Both outcomes re-price the frontier.

### 2.2 Re-fit the scaling law with an interaction term + GSPRT-reconfirm the rank

- **Mechanism.** Add an `log₂w · log₂d` interaction term to
  `scaling_analysis.py` (~10-line WLS patch over on-disk data); report the
  *within-depth* width slopes and *within-width* depth slopes, not one averaged
  ratio. Separately re-rank the top-5 configs with `sprt.rs` (pair-GSPRT).
- **Why it follows from the data.** The "5.2×" headline is an *additive* fit on a
  grid where the levers interact: width buys ~16 Elo/doubling at `d2` but ~123 at
  `d9`. The interaction is highly significant (the cross term lifts R² 0.948→0.965)
  and the additive ratio drops to ~4.0× once the degenerate `d2` row is removed.
  The top configs (`w128d9 … w24d9`) have **fully overlapping bootstrap CIs** in
  `elo.csv` — the "champion" rank was never confirmed with the ship-grade
  primitive the platform was built for.
- **Decisive experiment.** Run the refit (<5 min, data on disk). Then GSPRT
  `w128d9` vs `w96d7`/`w64d9` on a disjoint confirmation seed region. Note: the
  candidate's proposed iso-knee triad (`w16d7`/`w8d9`/`w24d4`) is *not* iso-cost
  (4.93–8.56 ms) and separates trivially — skip it; if a knee check is wanted use
  the true iso-node pair `w16d7` vs `w32d4` (both 96 nodes).
- **Expected payoff.** Hardens (not overturns) the depth thesis and replaces a
  point-estimate rank resting on 8-8 near-mirror edges with a real verdict. The
  decision (push depth) survives either way; this protects 2.1's reallocation
  call from chasing a regression artifact.

### 2.3 Profile the per-node self-time split before betting on any rewrite

- **Mechanism.** `samply` over `examples/profile_beam` at `w16d12` (the
  *unmeasured* narrow-deep corner) for the real clone-vs-`evaluate_cols`-vs-movegen
  split, and characterize whether the `94 µs/node` constant (R²=0.98 aggregate)
  drifts in the speculative tail.
- **Why it follows from the data.** The "~15% eval self-time" figure is
  unverified in-repo; the perf campaign's measured number is **~12% eval, ~40%
  memmove** (the dominant term already fixed in `4b4e9c0`). Note the hot
  `SearchState.board` is **already a `Copy` BitBoard** (`[u64;16]`, 128-byte
  memcpy), so a naive copy-on-write board prefix has *no heap to share* — the
  realizable clone savings live in the queue/pending `SmallVec`s and the residual
  `clone-deferral` item the campaign already named.
- **Decisive experiment.** Read the existing receipt first (`git show 4b4e9c0` +
  the beam-perf memory). Only re-profile to answer the *one* open question: does
  the cost model hold at `d12`+ where speculative plies fan out ≤7×? If clone
  still dominates there, the deeper-search lever is the named survivors-only
  clone-deferral, not eval shape.
- **Expected payoff.** Cheap insurance that decides feasibility/sequencing of the
  two expensive next-paradigm bets (deeper search vs learned eval) — mostly
  re-confirming a known split, with the d12 residual being the new bit.

### 2.4 — 2.6 Eval-side levers (lower confidence; gate behind 2.1)

The handcrafted CC2 eval is **near a local optimum for APP** — every single-lever
probe (`mix05`/`well1`/`spin2x`/`combo4`/`pc40`) tied or lost to `attack_tuned`
across three independent confirmations, and `attack_true(λ)`, which put the *true
objective in the search*, lost at every matched config (0.434 vs 0.572 @ d3;
0.618 vs 0.721 @ d6w32) because CC2's shaped tables carry beyond-horizon value an
in-horizon objective cannot see. So the eval surface is *not* where the cheap
wins are. The remaining eval moves, ranked by remaining novelty:

- **2.4 Fidelity restorations (medium).** Two CC2 terms are *omitted*, not
  re-weighted: `softdrop` has a weight slot (`-0.2`) that is never applied, and
  the T-slot cutout is approximated as a single cutout while the champion climbed
  `tslot[2]` up to 4.465. Do the **cutout half only first** (it is a faithful
  restoration via the bag the search already threads); A/B at `d6w32` *and* `d9`
  under rain pair-GSPRT — a *larger* margin at d9 is the depth-amplification
  signature. Skip the softdrop half: this engine's movegen has no faithful
  soft-drop distance to restore. *Expected: most likely a tie; tail upside is a
  small deterministic gain on T-heavy boards.*
- **2.5 Staged-spike readiness term (medium, versus-only).** A board-shaping Value
  term for "loaded but unfired spike" (well-depth ≥4 ∧ `ctx.b2b`). **Caveat
  verified against code:** the candidate's load-bearing gate ("a hold/queue I is
  available") is *not computable at the eval seam* — `EvalContext` carries only
  `{combo, b2b}`; hold/queue live in `SearchState`. Run only the board-only
  version (no trait change) first; if it ties (the `attack_true`/`well1`
  prediction), do not build the `EvalContext` refactor.
- **2.6 Value/Reward calibration (refuted — skip).** The premise is false:
  `cc2.rs score()` builds both halves in the *same* CC2 weight space and applies
  the *same* `SCALE=256` (README: "Both halves use the same factor, so
  Value + Reward still composes"). There is no unit to calibrate; the only
  non-degenerate knob is the board-vs-reward magnitude the hillclimb already
  swept. Listed here only so it is not re-proposed.

### 2.7 Speculation handling — the hidden crux behind 2.1 (medium-high)

With `preview_count=5`, only **6 plies are concrete**; *all* unproven depth
(d7–d15) is `SPEC_DECAY=0.75`-discounted, no-expectimax, truncated bag rollout.
Two contending fixes: **(A) cheapen** the 7-way bag fan-out to a deterministic
canonical-first-K (`with_spec_sample(k=2–3)`), reinvesting saved nodes into
depth; **(B) fix the backup** by folding speculative reward as a bag-uniform mean
(expectimax) instead of optimistic-max. **Do not build either first.** The
prerequisite is 2.1's `w16d12` run: if per-ply Elo keeps paying past d9, stock
speculation is already working and both arms solve a non-problem. Verified
constraint on arm (B): the beam has *no per-parent aggregation seam* (children are
flattened into one global truncation), so true expectimax is a restructure of the
generation-staged invariant, not "~50 lines." A cheaper first cut is a
`SPEC_DECAY` sweep `{0.0, 0.5, 0.75, 0.9, 1.0}` + speculation-off at `d9/d12`
under GSPRT — it localizes a code-only win or refutes the "speculation is the
bottleneck" thesis before anyone touches the backup.

### 2.8 Re-test best-first with a deep cap at versus (medium)

`BestFirstPlanner` is the only adaptive depth allocator (a max-heap under a node
budget with a real per-root transposition DAG) and is the **shipped in-game Mind**
(`controller.rs`), yet it has *never* been measured at versus post-bitboard (only
TRAIN APP). **Verified blocker:** it treats an empty-queue node as a *leaf*
(`best_first.rs:228`), so past the ~6-piece queue its budget goes to *width*, not
depth — `bf2k-d10` is byte-identical to `bf2k-d8` (the cap never binds). So the
zero-code race `best_first(2000,15)` tests *nothing new*; the honest cheap test is
`best_first(972, 9).cc2(attack_tuned)` (node-matched to the champion) vs
`tp128d9` under rain GSPRT — does its per-node TRAIN-APP edge survive on the
non-gameable metric? A loss closes the lingering "best-first is a cheaper
champion" question; a draw/win gates the speculation-port refactor.

---

## 3. The policy-improvement-operator unification

**The operator already exists in the code; name it, do not build a new trait.**

### The abstraction

`π' = Improve(π₀, V, budget)` maps exactly onto the `Mind` seam:

| Operator term | Code |
|---|---|
| `π₀` (prior) | the uniform-over-legal distribution from `hold_placements()` in canonical movegen order |
| `V` | `&dyn Evaluator` (`eval/mod.rs`) |
| `budget` | `SearchBudget { nodes, max_depth }` (`search/mod.rs:64`) |
| `π'(s)` (improved action) | `best()` → the argmax of `root_best` via `best_root_plan` |
| `Improve` instantiated | `SearchPolicy` *already* holds the `(Box<dyn Mind>, Box<dyn Evaluator>, SearchBudget)` triple |

The shared core is already factored in `search/mod.rs`: `commit_child`
(fork→classify→commit), `score_child`/`score_placement`, `hold_placements`,
`best_root_plan`, `RootKey`, `DEATH_SCORE`. The `root_best: Vec<i32>` spine is
carried index-aligned in all three impls.

### What it subsumes (and what it does not)

There are **three** `Mind` impls, not five (the brief overcounts). `TpBeam` is
`BeamPlanner::transposing()` — a `bool` field, not a type. "Greedy" is not a
`Mind` at all: the `GreedyPlanner` type was deleted; it survives only as
`SearchBudget::single_ply()`, beam/best-first at `depth=1` (pinned
byte-identical), or the `score_placement()` helper. So: **2 real operators**
(Beam, BestFirst) + 1 flag (`transpose`) + 1 degenerate budget (greedy).
`PcCoveragePlanner` does **not** fit — it maximizes *option coverage* (how many
7-bag futures stay PC-alive), not leaf Value, and delegates ordinary play to a
beam fallback. It is a **hierarchical subgoal/options policy above a Mind**, not a
peer of `Improve`.

### The flywheel it unlocks

The codebase is already a *partial* policy-improvement operator; what is missing
is a **consumer of the improvement**. Today the beam computes a dense backed-up
Value over every ply-1 root (`root_best`) at depth 9 and throws all but the argmax
away. The one new artifact worth adding:

```
fn best_distribution(&self) -> Option<&[i32]>   // default trait method, returns root_best
```

`softmax(best_distribution()/T)` is exactly AlphaZero's improved-policy target —
**free to extract, deterministic, zero new search work**. It feeds (a) the
learned move-ordering prior (within-paradigm, §4.2) and (b) the value-net teacher
(§4.1), closing the loop the post-mortem says was never closed — now with a
**0.82-APP teacher** instead of DT-20's ~0.2.

### Risks

1. **Over-abstraction** (the codebase's stated allergy). A forced `Improve` trait
   removes *zero* duplication and would degenerate into the `Mind` trait that
   already exists. **Decline the trait; ship `best_distribution()` only.**
2. **Myopia inheritance.** A target distilled from a `d9` search inherits that
   horizon. `root_best` is backed up *from leaves* so it is less myopic than
   `attack_true`, but validation must be held-out death-decisive versus +
   downstack under pair-GSPRT, never solo APP, never Bradley-Terry.
3. **Diffuse-vs-deterministic.** If `root_best` is near-deterministic at the
   champion's `w128`, a policy head buys no ordering signal — a Phase-0 measurement
   (entropy + top-k recall) must fire before any training.

### Migration path (each step a registered immutable name, gate green)

- **Step 0** (cleanup, zero-behavior): fix the dead `GreedyPlanner` reference in
  `score_placement`'s doc; reconcile the `evaluate_batch` story (documented but
  **absent from the live trait** — re-add the default method or correct the docs).
- **Step 1** (free, highest leverage): §2.1 — register `w16d12`/`w16d15`,
  GSPRT-confirm. Settles "push the paradigm" before any ML.
- **Step 2** (instrument, behavior-preserving): add `best_distribution()` as a
  default method (`best()` unchanged → every shipped decision byte-identical); add
  a board-snapshot field to the research record emit (currently `games.jsonl`
  stores **no per-position state**); Phase-0 analysis with two STOP gates (cc2
  Value already explains the variance, *or* `root_best` near-deterministic).
- **Step 3** (gated behind Step 2 firing): re-add `evaluate_batch`; distill the
  champion's targets behind `&dyn Evaluator`. **Hard gate:** match CC2 at
  iso-search on a validation region *before* adding depth.
- **Step 4** (only if Step 3 clears): self-play returns with mandatory near-death
  coverage to exceed the teacher; or ship the wasm-affordable root-only policy
  prior.

---

## 4. The next paradigm — ranked by leverage × feasibility

### 4.1 NNUE-style cheap leaf eval, distilled from the champion (next-paradigm anchor)

- **Why now.** This is Chess-2014. Under a latency wall, a saturating depth law,
  and a handcrafted eval at a local optimum, the winning move is to keep the
  search and replace only the leaf eval. The post-mortem's two blockers are gone:
  a 0.82-APP teacher (`tp128d9`'s search-backed Values) and a rain harness that
  manufactures near-death coverage. The eval seam is the explicit insertion point
  (`beam.rs`: "a batched value-net backend belongs *at* the Evaluator trait").
- **Shippability (determinism + wasm).** Hand-roll the forward pass in
  **fixed-point `i32`** in the `SCALE=256` domain — determinism-proof by
  construction (integer ops bit-identical across opt-z/opt-3/native/wasm), **no
  Burn at inference** (Burn for training only), no ~4k-line dep tree (the *actual*
  prune reason). A per-node value net pays off in-game only at the ~100×-per-node
  point where the deferred worker venue is needed; at guideline budgets prefer a
  **root-only policy prior** (~1 call/decision, ~100× cheaper, wasm-affordable).
  The NNUE incremental-accumulator claim is **unverified** — `commit_child`
  changes one placement but T-spins/clears mutate many columns, so the
  changed-columns accumulator may not be cheap.
- **Decisive gate.** Phase 0: regress realized rain-decisive attack on cc2 Value
  (requires the §3-Step-2 board-snapshot field first — *not* runnable on existing
  `games.jsonl`). If cc2 Value already explains most variance, STOP. Phase 1:
  distill, plug at the same `w16d6` config, **match CC2 at iso-search on holdout
  before adding depth.** Honest odds: ~40% it beats/matches CC2 in native
  research; <20% it ships in-game without the worker venue.

### 4.2 Learned move-ordering prior (the cheapest, safest ML step)

- **Why now.** The width wall is a *truncation* error: a line that scores low
  early but pays off deep is cut before its payoff is visible. A prior folded into
  the **truncation key only** (`ranked_frontier`), never the backed-up Value, lets
  `w8`-`w16` retain the lines `w32`-`w128` would — converting the 5×-weaker width
  lever directly into depth. It **sidesteps the `attack_true` failure mode by
  construction**: a bad prior degrades to current behavior, never to a wrong
  decision, and a ranker cannot be capped by a weak teacher the way the DT-20
  value net was.
- **Shippability.** Handcrafted version is a few bit-ops below eval cost,
  wasm-safe; learned version bakes to a const table (GBDT/2-layer MLP), no Burn
  dep, deterministic pure function of columns+placement.
- **Verified caveat.** This only saves nodes in `best_first` (which pops one node);
  the **beam scores every child before truncating**, so a prior saves the
  champion tp_beam family *zero* nodes unless it lets a *narrower* beam keep the
  right survivors. The decisive test is a **truncation-regret count**: how often a
  `w16` cut drops a child that is an ancestor of the eventual `w128` best-leaf. If
  regret is near zero, the wide beam isn't saving deep winners via width and a
  prior cannot help — refuted for one logging pass. (Do *not* run the candidate's
  stated pilot "order by depth-1 CC2 score + shrink width" — that *is* the
  truncation the beam already performs.)

### 4.3 Expose `root_best` as a ply-1 policy distribution (free bridge)

The §3 artifact, as its own experiment. Phase-0 only, read as a kill-switch:
expose `best_distribution()`, log it from `tp128d9` self-play, measure entropy +
top-k recall. **Verified caveat:** only the ~3–8 roots whose descendants survive
into the `d9` frontier get a genuine beyond-horizon backup; the other 34–68 roots
*freeze* at a shallow value once truncated, so `softmax(root_best)` is **not
beyond-horizon by construction.** Likely outcome at `w128`: near-deterministic →
fires the candidate's own STOP. Cheap, dignified, prevents a multi-week NN revival
from being justified on a free-accessor pretext.

### 4.4 Speculative-leaf-only learned value (sharp minority refinement)

Keep CC2 on the 6 concrete plies (locally optimal); learn `V_spec(board, bag)`
*only* at the empty-queue speculative tail. **Verified problems:** (a) `SPEC_DECAY`
discounts only Reward, never the static board Value — so the survivability signal
is *already* applied undiscounted at speculative leaves; (b) `evaluate_cols`
receives no bag/queue, so `V_spec(board,bag)` is unexpressible without a core
trait change. The genuinely fresh thread it points at — `SPEC_DECAY` is a crude
proxy — is a **constant sweep, not a value net** (see §2.7). Run the sweep; treat
the net as contingent.

### 4.5 Game-theoretic two-agent exchange search (highest ceiling, research-only)

The one axis every ML lens misses: versus is a **two-agent timing game**, but
every `Mind` searches one board vs a fixed pending queue and the driver merely
routes net attack. The garbage exchange is deterministic and fully exported
(`GarbageBatch.hole_col` is on the snapshot; cancel/rise are pure functions), so
*when* you spike relative to the opponent's clear cadence is a real, currently-
invisible lever. **Verified blocker:** the proposed "free log regression" is
**not runnable** — `games.jsonl` stores no per-ply telemetry. The corrected
decisive test is a **board-health-controlled** timing ablation: instrument
per-lock `(attack, was_clear_less, opp_pending)`, re-run rain champion-vs-champion,
and ask whether the winning spike concentrates in the opponent's rising window
*after conditioning on opponent stack height* (so you don't just rediscover
"healthy boards spike well"). Nested k×m search cannot fit the 192-node wasm
budget — this ships research-only as a stronger reference/distillation target, not
a planner. Likely a ~60% stop-sign; high value as one.

### 4.6 Exact opener/PC book as a subgoal layer (niche, demo-grade)

Replace reward-shaped PC hunting (which **cannot find PCs** — `tp256d12 +
perfect_clear=1000` stays ~0.01 PPC) with a precomputed opener book keyed by
`(board-residue, bag-state)`, served before the beam takes over.
**Verified problem:** the exact alternative *already exists* —
`PcCoveragePlanner`'s near-exact scan tops out at **0.0875 PPC** (7/80 on the
opener screen), an order of magnitude below the promised 0.43. Exactness is *not*
the missing ingredient; most random states aren't PC sites and the bot won't
sacrifice survival to chase a marginal one. A book can only convert PCs the scan
*already finds but discards* — pilot by reading the existing `pc-reveal-s28w8`
run's opener-PC rate before encoding anything. Demo-grade at best; does not move
versus survival, which is the real strength metric.

---

## 5. Sequenced experimental program

Ordered to maximize information per unit compute. The panel agreed the **first**
experiment is the narrow-deep depth run.

| # | Experiment | Platform / metric | Advances if… | Kills if… |
|---|---|---|---|---|
| **E0** | *Pre-test:* one instrumented `w16d15` decision — argmax-stabilization ply + speculative-leaf fraction | native single-decision dump | argmax keeps flipping past d7 → run E1 | argmax fixed by d6–d7 ∧ >80% leaves speculative → depth can't move the decision; jump to §2.7 sweep |
| **E1** | Register `w16d12`/`w16d15`/`w24d12` vs `tp128d9` | rain-decisive **pair-GSPRT** (confirmation seeds) | narrow-deep ≥ champion at <1/5 nodes → champion mis-allocated, depth is the cheapest SOTA gain *and* you have the strong teacher | per-ply Elo collapses past d9 → ceiling is the speculation/preview wall → §2.7 / §4.x |
| **E2** | Interaction refit + GSPRT re-rank top-5 | on-disk WLS (free) + `sprt.rs` | hardens depth thesis with honest within-lever slopes | (cannot kill the decision; only sharpens it) |
| **E3** | `SPEC_DECAY` sweep `{0,0.5,0.75,0.9,1.0}` + spec-off at d9/d12 | rain GSPRT | a sweep value beats 0.75 → free code-only win; or localizes the leaf as the bottleneck | flat across the sweep → speculative leaf is *not* the bottleneck; no `V_spec` |
| **E4** | `best_first(972,9)` vs `tp128d9` | rain GSPRT (node-matched) | best-first's per-node edge survives at versus → cheaper champion + speculation-port worth building | ties/loses → beam width is a survival hedge; close the question |
| **E5** | Profile `w16d12` self-time split | `samply` (read `4b4e9c0` first) | clone dominates at d12 → survivors-only clone-deferral is the deeper-search lever | eval dominates → green-light the value net's per-node cost |
| **E6** | `best_distribution()` Phase-0: entropy + top-k recall + truncation-regret | native log analysis | diffuse `root_best` ∧ positive truncation regret → policy prior worth training | near-deterministic ∨ ~zero regret → policy arm dead, ship E1's depth config |
| **E7** | Cutout-fidelity restoration A/B | rain GSPRT @ d6w32 *and* d9 | larger margin at d9 (depth-amplification) ∧ no downstack regression | ties at both depths → eval local optimum holds |
| **E8** | Value-net Phase-0 attack∼Value regression (needs E-step board snapshot) | native regression | large *structured* residual (e.g. near-death class) → proceed to iso-search match gate | cc2 Value explains the variance → no net headroom; go deeper |
| **E9** | Distill champion → fixed-point net; iso-search match gate | validation-region holdout | net matches CC2 at iso-search → add depth; then self-play to exceed | merely ties → not worth the dep weight; answer is "go deeper" |
| **E10** | Board-health-controlled spike-timing ablation | instrumented rain re-run + duckdb | timing concentration survives the height control → two-agent search has headroom | vanishes under control → single-agent eval already captures it |

E0–E2 are hours-to-overnight and zero/near-zero code. **Nothing past E1 is
justified until E1 reports.** E8–E9 (the value net) and E10 (two-agent) are the
multi-week bets, each gated behind a cheap Phase-0 that can stop them.

---

## 6. Open questions + strongest dissent

### Open questions

1. **Does depth keep paying past d9?** Only +20…40 Elo/ply proven at d7→d9;
   d12–d15 are runnable today but unmeasured. The single highest-leverage unknown.
2. **Is the Elo-vs-APP relationship monotone at the top?** The top configs sit in
   overlapping bootstrap CIs; re-confirm with pair-GSPRT, not Bradley-Terry.
3. **Does best-first with a deep cap beat `tp128d9` per node at versus?** Only
   TRAIN-APP numbers exist; never re-tested at versus post-bitboard.
4. **Can a champion-distilled value net beat the handcrafted eval — and only at
   the ~100× per-node cost where the worker venue is needed?** Seam ready,
   training loop not in-repo.
5. **Does `root_best` as a distribution teach anything beyond the eval, or inherit
   `attack_true`'s myopia?** Free to extract; unproven.
6. **Where does eval sit in the self-time budget at d12+?** ~12% at d9; if clones
   dominate at depth, the lever is movegen/clone cost, not eval.
7. **Is the `94 µs/node` constant stable across the grid, or does it drift in the
   speculative tail?** R²=0.98 is aggregate; per-config residuals uncharacterized.

### The strongest dissent (the disagreement worth resolving first)

**Is depth-past-9 a real lever or a measurement artifact?** Most lenses treat the
depth law as actionable and bet narrow-deep beats the champion. The
first-principles position is sharper and, on the data, partly correct: the "5.2×"
ratio is an **additive-regression artifact** — the steep returns live almost
entirely in the **6 concrete plies** (split by regime, the depth coefficient is
~438 on concrete plies and **collapses to ~189** on speculative ones, where it is
only ~1.7× width), the grid *clips* the depth axis at `d9` (inflating the
coefficient), and the confirmed "+20–40 Elo/ply at d7→d9" *is itself the
already-decayed, half-speculative slope* — evidence of saturation, not against it.
The regime-appropriate fit predicts `w16d12 ≈ 914`, **below `w16d9`**. The
corroborating receipt: `bf2k-d10` was byte-identical to `bf2k-d8` (more depth past
the concrete horizon bought nothing). **This is the most decisive scientific
disagreement in the program, and E0→E1 adjudicate it directly** — for the cost of
one instrumented decision and one overnight tournament, with the regime-split
prediction (`w16d12 ≈ 914`) pre-registered as the null to beat. The downstream
diagnosis of the `~1411` ceiling forks on the answer: a **temporal-expressiveness**
ceiling (eval can't price an unfired spike → §2.5/§4.1), a **game-theoretic**
ceiling (single-agent search asks the wrong question → §4.5), or a
**speculation-quality / 5-piece-preview** ceiling (→ §2.7/§4.4) — three different
next paradigms, all downstream of whether the depth slope bends at the preview
boundary.
