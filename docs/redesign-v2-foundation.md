# v2/foundation — redesign map

Fresh implementations off master of the four components the rl/strike review
graded GO ([LANDING.md on the reference branch]). The reference branch is a
*reference*: designs are re-derived, code is not copied, names are free. The
32 review findings are the spec of what must be impossible here, not a patch
list.

## 1. tetr-core: leaf batching + speculative dedup

- `Evaluator` gains two things only:
  - `board_only() -> bool` (default `false`): true iff the score depends
    only on the locked board + chain ctx. Opt-IN to sharing; the safe
    default is the fanned path.
  - `evaluate_leaves(&[Leaf]) -> Vec<(Value, Reward)>` (default: loop
    `evaluate_cols`): `Leaf` borrows the child state + lock + t_spin + ctx,
    so a net backend can batch a sibling group in one forward.
- Beam speculation commits a placement **once** (lock + clears once), fans
  the ≤7 bag continuations via a `deal_speculative` split in `SearchState`,
  and — when `board_only()` — scores once per placement, sharing the score
  across the fan. Enumeration stays piece-major so ranking order (and thus
  tie-breaks) is bit-identical to master.
- One scoring funnel: `PendingChild.score: Option<(Value, Reward)>`. The
  dedup fills it; `score_pending_into` uses it or evaluates. No sibling
  expansion path.
- `Mind::root_scores()` exposes the per-root backup — the expert-iteration
  read interface.
- Tests first: a naive-oracle equality test for the dedup; a queue-sensitive
  fake evaluator proving the fanned path still fans; the full APP golden run
  before the commit is final (bit-identity is claimed, so it is proven).

## 2. tetr-nn: the learned-eval crate (replaces the reference tetr-valuenet)

Reference pain: two nets, two encodes, a Python re-encode mirror, shards
storing a lossy raw form while hashing the lossless encode (the
representability bug), a batched forward nobody called. The redesign removes
the *categories*:

- **One net** (two-board P+V), **one encode**, **one forward** — batched is
  THE forward; a batch of 1 is the scalar case. No v1 anything.
- **Store what you serve**: shards persist the encoded observation bytes the
  net actually consumed (packed planes + f32 features), not raw components
  to re-derive later. Train/serve parity becomes definitional — the trainer
  reads the served bytes — deleting the Python mirror, the obs_hash
  cross-language tripwire, and the truncation bug class in one move. Shards
  regenerate in minutes by design, so losing raw-form re-derivation is YAGNI.
- Modules: `obs.rs` (layout + encode + one FNV impl), `net.rs` (weights +
  batched forward + heads; BLAS on macOS/`openblas` elsewhere, plain loop
  fallback), `serve.rs` (`NetEvaluator`: per-decision frozen opponent,
  cached opp embedding, `evaluate_leaves` batching), `shards.rs` (atomic
  tmp+rename writes, game-aligned, ragged child storage — no 128-slot
  padding), `ane.rs` (`coreml` feature, ort).
- The committed round-0 fixtures (config + weights + golden vectors) are the
  rewrite's proof: `net.rs` keeps the reference weight-layout contract so
  the proven goldens pin the fresh forward.
- Deferred until measured (ponytail): the cross-game fusion server. The
  per-sibling-group batch may be enough off the ANE; build fusion when a
  profile says so.

## 3. Venue: one game loop, sudden death

- `VersusFormat` gains sudden death: rain period halves every 40 post-cap
  plies, hard cap 2×, `EndReason` {Topout, Escalation, TrueCap}; outcomes
  from topout only.
- **One** versus game loop, parameterized by two `Arm`s (below). Every
  instrument and every future datagen drives this loop; the reference
  branch's four hand-rolled copies become impossible, not just deleted.

## 4. Instruments + CLI

- `Arm`: a player in a versus game — `beam:<dir|cc2>@w8d5`, `policy:<dir>`,
  `value:<dir>`, parsed once by one grammar. This is the concept that
  replaces ArmB enums + a repurposed `--infer` flag.
- Typed clap subcommands beside the existing registry runner (which is not
  churned): `duel` (CRN pairs, any two arms), `gate` (latched trinomial
  pair-GSPRT — verdict latches at first boundary crossing; in-flight pairs
  report but never decide), `preflight` (an arm vs its own d1 policy — the
  G_π probe). All take `--seeds <base>+<count>`; no instrument has a
  hardcoded region; receipts record what actually ran.

## Order

core → nn → venue → instruments → docs/gate/APP-golden. Each lands as its
own gate-green commit.

## Deferral notes (recorded at build time)

- **ANE backend (`ane.rs`)**: deferred to the v2 campaign. It requires an
  ONNX export step that lives in the campaign's training pipeline, and
  shipping an untestable feature-gated module would violate this redesign's
  own standards. The proven reference implementation (~100 lines over `ort`)
  ports in an afternoon once an export exists.
- **Cross-game fusion server**: deferred until a profile says the per-
  sibling-group batch is the bottleneck (pre-registered above).

## Build record (2026-07-06)

Everything above landed as 7 gate-green commits on `v2/foundation`; the full
CI-mirror gate is green and the decisive proofs ran:

- **APP golden, bit-identical**: `run marathon attack-tuned-d6` (depth
  crosses the preview, so speculation + the dedup fire throughout) produces
  the identical result JSON on master and this branch — APP
  0.648888885974884 both sides. The dedup's bit-identity is proven at the
  system level, on top of the in-crate naive-fan oracle test.
- **PyTorch golden, 16/16**: the fresh tetr-nn forward matches the round-0
  export to 1e-4 (committed fixtures).
- **End-to-end instrument smoke**: `duel --a greedy --b policy:<fixture>`
  played receipted sudden-death games through the one versus loop.

Sizes: the four components total ~2.6k lines of fresh Rust + tests, against
~8k reviewed lines on the reference branch. Not ported: v1 net paths, the
Python re-encode mirror, three duplicate game loops, fixed-width shard
padding, the ArmB/--infer vocabulary, and every defect in the review's 32
findings whose category the redesign removed.
