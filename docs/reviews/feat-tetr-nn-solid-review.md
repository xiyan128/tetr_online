# SOLID review: `feat/tetr-nn`

Scope: 64 files, +13,270 / −876 versus `master`.

## How this review was done

I read the architectural seams directly: `eval/mod.rs`, `search/mod.rs`, `state.rs`, `registry.rs`, `nn/lib.rs`, `bit_board.rs`, `weights.rs`, `best_first.rs`, `beam.rs`, `lock_clear.rs`, and `api.rs`. Six reviewers covered the rest in parallel. Every Critical and High claim here was checked against the code before being included, not relayed on trust. That check refuted four findings, listed under "Claims I downgraded." This branch had not been reviewed before, so apply the same skepticism to the rest.

## Verdict

The trait architecture is strong. `Occupancy`, `Evaluator`, `Planner`, `Policy`, and `DecisionRunner` are clean, object-safe seams, and the neural net plugs into the same `Evaluator` path as the hand-written bots. That is the difficult part, and it is done well.

The problems cluster in three areas:

1. The `Evaluator` trait's three-method "bit-identical" contract is unenforced, and it cancels the branch's main performance goal for the shipped bot.
2. A positional-flat-vector, magic-constant, stringly-typed pattern runs through the eval, NN, research, and training code, so each extension becomes a multi-file edit.
3. Construction and harness code is duplicated unevenly. The team centralizes the subtle logic (`fold_combo`, `score_placement`, `lock_piece`) but leaves 6x, 4x, and 3x copies elsewhere.

On top of that there is one build-time break and several silent-failure paths that mislabel what the user is actually running.

Severity legend: Critical is a correctness or build break on a real path. High is latent correctness, or a design defect that cancels a stated goal. Medium is an SRP/OCP/DRY/DIP cost, or a missing test on nontrivial logic. Low is naming, docs, or minor duplication.

---

## Part 1: Cross-cutting architecture (read this first)

These span subsystems, so no single reviewer could see them. This is where the leverage is.

### X1. The `Evaluator` trait inverts the cost model: it requires the slow method and makes the fast ones optional overrides that "must be bit-identical," and the shipped evaluator never overrides them

Location: `eval/mod.rs:103-153`. The required method is `evaluate(&Board, ...)` over a dense `Array2D`. `evaluate_cols(&BitBoard)` and `evaluate_batch` are defaulted, and they carry a prose-only "must be bit-identical" obligation. Confirmed consequences:

- `LinearEvaluator` (the shipped greedy default) and `BurnEvaluator` (the NN) implement only `evaluate`. The search calls `evaluate_cols` and `evaluate_batch` exclusively (`search/mod.rs:133`, `best_first.rs:177`, `beam.rs` batch). So every node runs `BitBoard` to `to_array2d()` (heap-allocates a full `Board`), then `evaluate`, then `BoardFeatures::extract`, then `column_bits()`, which rebuilds the bitboard it already had. The whole `bit_board.rs` and `features.rs` rewrite (the documented "minutes vs hours" and "10x latency" win) is undone at the shipped seam. Only `Cc2Evaluator` overrides `evaluate_cols`. This is the most important fix in the branch.
- The "bit-identical" invariant is enforced by one test per impl and never as a per-impl `evaluate` vs `evaluate_cols` differential. That is an LSP hazard on a contract that search correctness depends on.
- The trait is also an ISP smell: three abstraction levels (dense, bitboard, batched) in one interface. Authors want `evaluate`, clients want `evaluate_cols` and `evaluate_batch`, and nobody wants all three. `BitBoard`, an engine type, leaks into the public trait surface.

Fix: make the bitboard kernel the required method, for example `fn score(&self, EvalInput<'_>) -> (Value, Reward)` over `cols: &[u64]` plus a named input struct. Derive the dense path as a free adapter, and default `score_batch` to map `score`. That collapses the dense/bitboard duplication, shrinks the invariant to one path, removes the positional 4-tuple (X2), and structurally gives `LinearEvaluator` and `BurnEvaluator` the fast path. Until then, at minimum override `evaluate_cols` on both and add the differential test.

### X2. Positional flat vectors and "keep in sync" comments are the integration contract, across files and languages

The same hand-indexed pattern recurs. `BoardWeights`, `RewardWeights`, and `Cc2Weights` each hand-list fields in `params()`, `from_params()`, and `dot()` (`weights.rs:126-155, 246-277`, `cc2.rs:92-120`). The NN repeats the feature order and scales in `FEATURE_SCALE` and `features_to_input` (`nn/lib.rs:62-71, 278-295`). The Python trainer re-transcribes them (`train_value_net.py`, DT20 plus `FEATURE_SCALE`). `distill.rs` hand-lists them a fourth time. The canonical field order lives in a fifth place (`features.rs`, `BoardFeatures`).

The round-trip tests catch reorders but not omissions: a field absent from both `params` and `from_params` round-trips fine while being silently untunable. Nothing links the Rust and Python copies except a comment. The struct has 10 board features, but the net silently uses the first 8 by positional convention. This is the central train/inference-skew risk and the dominant OCP cost: adding one feature touches roughly six edit sites across up to five structs and files.

Fix: one source of truth. A `#[derive(Tunable)]` or macro that generates `PARAM_COUNT`, `params`, and `from_params` from the field list (the code's own test comment at `weights.rs:325` asks for this). Index `features_to_input` by explicit per-field accessors so reordering `BoardFeatures` is a compile error. Emit the feature order and scales into the safetensors metadata (`nn/lib.rs:198` is currently `None`) and assert them on load, so a Python-side skew becomes a load-time error instead of silent wrong values.

### X3. Magic constants and tuning data have no home; they are scattered across layers

`ATTACK_BOARD_PARAMS` (11 raw floats, sitting in the UI catalog at `registry.rs:30-42`), `BEAM_WIDTH`/`BEAM_DEPTH`/`ATTACK_BF_BUDGET` (registry), `SPEC_DECAY` (`beam.rs:40`), `EXPAND_CHUNK` (`best_first.rs:43`), `SCALE=256.0` (`cc2.rs:131`), `FEATURE_SCALE` (NN and Python), `MAX_PIECE_FRAMES` (defined twice, `lib.rs:447` and `behavior.rs:26`), and three different XOR seed salts (one duplicated across two files for a comparison that is "fair" only by luck). Research tuning data living in `src/ai/registry.rs` is an SRP inversion: a weights re-tune edits the same file as a menu-label tweak, and the headless bench cannot share the climbed weights.

Fix: move climbed presets into `tetr-core` (`Cc2Weights::ATTACK_CLIMBED`) and reference them from the registry. Name and de-duplicate the seed salts and `MAX_PIECE_FRAMES`. Make quality-affecting hyperparameters (`SPEC_DECAY`, beam width and depth, node budget) into fields with defaulted builders so the autoresearch workflow can sweep them without a recompile.

### X4. Dual-representation drift (Board vs BitBoard) is guarded by tests that prove equivalence to each other, not correctness

The differential tests are the right idea, but they assert `bitboard == engine`, so any shared bug stays invisible:

- ✅ (ENG-2, resolved in b61396d) `cleared_rows` *over-reported* buffer rows in both representations — the textbook case of this risk: a shared visible-bounded clear the differential tests couldn't see, since they only assert `bitboard == engine`. Caught only by the constructed buffer-row tests this section recommends.
- `from_board` silently truncates boards wider than 16 or taller than 64, and the envelope is unenforced anywhere (ENG-4).
- `to_array2d` (used on real fallback paths) has no round-trip test.
- Occupancy predicates exist with three different y-bounds (cc2 `0..40`, features `0..64`, `BitBoard` `>=64`) and opposite out-of-bounds conventions.

Fix: add `from_board` composed with `to_array2d` round-trip property tests. ✅ (done in b61396d) the buffer-row case is now pinned by constructed tests (`clear_lines_clears_a_full_buffer_row`, `clear_full_rows_clears_buffer_rows`) that assert the clear/count, not just `bitboard == engine`. Enforce the 16x64 envelope at `Board` or `EngineConfig` (or `debug_assert` in `from_board`). Centralize occupancy on the existing `Occupancy` trait.

### X5. Identity is stringly-typed or positional throughout, with silent fallthrough

Model registry: a `usize` index plus a `String` label is the identity, and the index meaning shifts with the `nn` feature flag (entries 6-7 exist only with `nn`), so any persisted or passed selection breaks across builds. `select()` silently ignores out-of-range indices. Research: the bot is chosen by `match bot.as_str()` with a `_ => dt20` default, so `BOT=cc2custmo` silently benchmarks the wrong bot. Modes are mutually exclusive but expressed as independent `DOWNSTACK` and `VERSUS` env booleans, so setting both runs whichever is checked first.

Fix: stable typed keys (a `ModelId` enum or `id: &'static str` distinct from the display label, plus `select_by_id`). A `BTreeMap<&str, Factory>` or a `FromStr` that errors on unknown. A single `MODE` enum that errors on unknown. Erroring on unknown is the change that matters.

### X6. Silent failure that mislabels what is actually running

NN load failure degrades to the greedy linear bot with the error dropped (`registry.rs:203,232`, and embed mirrors it). There is no `warn!`, and `selected_label()` still reports the NN, so the HUD lies and "why does the value net play exactly like greedy?" is hard to debug. The research versus metric (RES-1) silently under-counts offense. Silent degradation is the right instinct. Silent and mislabeled is the bug.

Fix: log the error before falling back. Make the fallback observable: amend the label, or have the build return `Result` so the UI cannot display a wrong model name.

### X7. Construction and harness duplication, applied unevenly

The team centralizes the subtle code well: `fold_combo` "killed four copies," `score_placement` is "the one place scoring lives," and `lock_piece` is one source of truth. It then leaves: registry `SearchPolicy` wiring copy-pasted six times (plus a duplicated `include_bytes!`); beam and best-first sharing roughly 250 lines of scaffolding (`best_plan` byte-identical, `placements`, `StateKey`/`RootFingerprint`, child-construction, stale-run detection); CC2 hold/queue bookkeeping three times in `cc2_baseline`; the versus match loop twice (lib vs bin); the per-game event loop four times; and the SplitMix64 PRNG inlined three times while a centralized `cli.rs::SplitMix64` sits dead, with a doc claiming it is the single home.

Fix: the obvious extractions. A `search_controller(planner, eval, budget)` helper. A `search::common` module. A `Cc2Game` mirror struct. A `trait VersusOpponent` plus one match loop. A `step_engine_once` helper. Route all RNG through the one PRNG and delete the dead copy.

### X8. Doc drift, after a commit that claimed to fix doc drift

The tip commit is "fix comment drift from the bitboard refactor," yet: `BEAM.md` (marked "normative, locked, follow verbatim") still shows `lock_and_clear`, `VecDeque::pop_front`, the pre-`EvalContext` `evaluate_batch` signature, omits `spec_weight`, and lists transposition as out of scope while `best_first.rs` ships it. `api.rs:270-274` says "mirroring the spawn-collision path," which is false. `nn/lib.rs:149,328` say "JAX-exported" for an asset the `distill` bin produced. The `EvalContext` docs claim every evaluator "must reduce to chain-agnostic behavior," which `Cc2Evaluator` deliberately violates. The registry inlines competitive benchmark numbers that will go stale. A "normative but wrong" doc is worse than no doc: a future implementer following BEAM.md will reintroduce `lock_and_clear` and drop `EvalContext`.

### X9. Test coverage is bimodal

It is excellent where it exists: `commit_placement`'s five hold-aware tests, the bitboard differential suite, the weights round-trip, and `sandbox.rs`. It is absent on equally important surfaces: the registry and `model_select` (0 tests), the CC2 T-slot detector (0, and it is the most intricate logic in `cc2.rs`), the `evaluate` vs `evaluate_cols` per-impl differential (missing, despite the "bit-identical" contract), the `to_array2d` round-trip, an NN golden input-to-output vector, almost all research-harness logic, and the `DecisionRunner` substitutability test that left with the deleted `native.rs`.

---

## Part 2: Action items by subsystem

### Engine: `bit_board.rs`, `board.rs`, `lock_clear.rs`, `api.rs`, `attack.rs`, `scoring.rs`, `pieces.rs`, `t_spin.rs`

- **[High] ENG-1.** `insert_garbage` does not mirror the spawn-collision path it claims to. `api.rs:275-286`. It latches `BlockOut` but, unlike the spawn path (347-353), emits no `EngineEvent::GameOver` and leaves the buried `active` installed. Event-stream consumers (renderer, replay) miss an attacker-controlled top-out, and a renderer would draw a piece overlapping garbage. Fix: take `&mut Vec<EngineEvent>`, set `active = None`, push `GameOver`, and test it. (A reviewer rated this Critical; I down-rated to High because the blast radius is limited to garbage-top-out event consumers, which may not be wired into shipped gameplay yet. It is still a real defect plus a false doc.)
- **[High] ENG-2.** ✅ **RESOLVED** — chose "clear and count buffer rows" (the guideline whole-matrix rule). `Board::clear_lines` and `BitBoard::clear_full_rows` now scan the full backing height (`backing_rows()` / `total_rows`) in lockstep, so the physical clear removes exactly the rows `cleared_rows` already reported; `cleared_rows.len()` becomes a correct count with no consumer change. Pinned by constructed tests (`clear_lines_clears_a_full_buffer_row`, `clear_full_rows_clears_buffer_rows`) — as the item predicted, the differential tests can't catch it (they only prove engine == bitboard). APP gate verified byte-for-byte unchanged (the bot keeps the stack out of the buffer in survivable play, so the bug never manifested in benchmarked scenarios). *Original finding:* `cleared_rows` over-reported buffer-zone rows while `clear_lines` was visible-bounded, so a full buffer row inflated score/combo/B2B/reward while staying on the board.
- **[High] ENG-3.** ✅ **RESOLVED** (b61396d follow-up) — `clear_full_rows`'s carry-down shift is now `(*col).checked_shr(y + 1).unwrap_or(0) << y`, so the top row at `y = 63` (reachable only at the `total_rows == 64` clamp ceiling) yields a 0 carry instead of a `>> 64` overflow — matching the engine `Board`, which clears the top row fine. Pinned by `clear_full_rows_handles_the_top_row_at_the_64_row_ceiling`. *Original finding:* the shift overflowed (UB/panic) when the loop reached `y = 63`; latent since production backing is 40, but a `pub`-constructor footgun on the hot path that the ENG-2 fix moved from `visible_rows >= 64` to `total_rows == 64`.
- **[High] ENG-4.** `from_board` silently truncates boards wider than 16 or taller than 64. `bit_board.rs:83-89` uses `.take(MAX_WIDTH)` and `width.min(MAX_WIDTH)`. The "mirrors the engine exactly" doc holds only inside an unstated, unenforced 16x64 envelope. Fix: assert the envelope at the `Board` or `EngineConfig` boundary.
- **[Medium] ENG-5.** No `to_array2d` round-trip test, and `insert_garbage_lines` has only example tests. Both are new index arithmetic on real paths. Fix: property tests over the existing `SplitMix64` harness.
- **[Medium] ENG-6.** Three bounds for "on the board": `occupied` uses `y>=64`, `set` uses `total_rows`, `blocked` uses `y<64`, and the bare `64` magic duplicates `u64::BITS`. Fix: a `const COLUMN_BITS`, a private `in_bounds(x,y)` used by all three, routed through `total_rows`.
- **[Medium] ENG-7.** Occupancy is smuggled through colour: `GARBAGE_FILL` and `to_array2d` independently hardcode `PieceType::I`, so a future colour-aware feature or renderer treats garbage as I-pieces. Fix: add `CellKind::Garbage`, or at least share one placeholder const plus a doc note.
- **[Low] ENG-8.** `clear_full_rows` is `pub` and returns a different row set (visible) than `lock` reports (full range), with no cross-warning. Fix: make it private, or document the report-vs-clear distinction on both.
- **[Low] ENG-9.** `Cell.x` is `pub(crate)` but `Cell.y` is private, and `column_bits` packs from stored coords, correct only because `Board::set` is the sole writer. `board.rs:116-127, 427-432`. Fix: pack from the iteration index; unify field visibility.
- **[Low] ENG-10 (mine).** `full_rows` on a width-0 board folds `!0` over an empty slice and reports all 64 rows full (both `lock_clear.rs:78` and `bit_board.rs:161`). Degenerate, but a real latent footgun, and it pairs with EVAL-13. Fix: early-return empty when `cols.is_empty()`.

### Evaluation: `eval/mod.rs`, `cc2.rs`, `features.rs`, `weights.rs`

- **[High] EVAL-1.** `LinearEvaluator` never overrides `evaluate_cols`, so the bitboard refactor is defeated at the shipped seam. See X1 (the primary fix).
- **[High] EVAL-2.** The "bit-identical" contract across three methods is enforced by prose plus one test per impl, with no per-impl `evaluate` vs `evaluate_cols` differential (`Cc2Evaluator` especially, since it reimplements scoring over raw columns). See X1.
- **[High] EVAL-3.** `evaluate_batch`'s element type is a bare 4-tuple `(&LockOutcome, &BitBoard, Option<TSpinKind>, EvalContext)`, re-spelled in five places across crates. It invites a positional swap and leaks `BitBoard`. `eval/mod.rs:144-152`. Fix: a named `EvalInput<'a>` struct (folds into X1).
- **[High] EVAL-4.** The `EvalContext` doc puts a "must reduce to chain-agnostic behavior" contract on the shared trait type that `Cc2Evaluator` deliberately violates, since combo and B2B are core to its scoring. `eval/mod.rs:85-89`. Fix: move that guarantee to `LinearEvaluator` and `RewardWeights::attack`, where it is actually true and tested.
- **[High] EVAL-5.** `SCALE = 256.0` is a magic fixed-point factor with cross-evaluator incommensurability. `LinearEvaluator` emits `Value` and `Reward` in raw weight units (hundreds), while `Cc2Evaluator` emits `f32 * 256`, so any code that compares or mixes them compares incommensurable scales. A NaN weight becomes 0 silently via `as i32`. `cc2.rs:131,304-305`. Fix: document the bound, add `debug_assert!(is_finite())`, and rename to `CC2_FIXED_POINT_SCALE`. (Same scale-commensurability concern as NN-4.)
- **[Medium] EVAL-6.** `Cc2Weights` is a god-struct: 21 fields mixing tunable `f32`, a `u32` clamp, and a topology-switching `bool`, with a hand-maintained 11-of-21 `board_params` projection. `cc2.rs:38-127`. Fix: split `Cc2BoardWeights` and `Cc2RewardWeights`, mirroring `weights.rs`.
- **[Medium] EVAL-7.** `perfect_clear_override: bool` is a boolean trap that gates the entire reward branch via `!perfect_clear || !w.perfect_clear_override`. `cc2.rs:57,253`. Fix: a named local plus a comment, or a `PerfectClearReward` enum.
- **[Medium] EVAL-8.** `well_known_tslot_*` relies on unsigned-wrap edge semantics, and the T-slot detector (the most intricate logic in the file, with a cutout count approximated to 1) has zero tests. `cc2.rs:154-170, 371-412`. Fix: an explicit empty-column guard plus TSD-left, mirror, empty, and N-line-cutout tests.
- **[Medium] EVAL-9.** `near_full_rows` is an O(width * height) double scan with a per-row `.filter().count()`, the one new feature that ignores the file's bit-math discipline. `features.rs:644-656`. Fix: column-parallel bit math.
- **[Medium] EVAL-10.** `placement_reward` recovers piece identity from `lock.cells_locked.first()`, an undocumented `LockOutcome` ordering invariant, and an empty `cells_locked` silently suppresses the wasted-T penalty. `cc2.rs:415-420`. Fix: add `LockOutcome::piece_type`, or assert and test the invariant.
- **[Medium] EVAL-11.** `compute_reward` is a 50-line multi-concern function, and the `(t_spin, lines)` table is duplicated between the inline `match` and `score_action`. `eval/mod.rs:226-289`. Fix: a shared `ClearKind` enum; split classify from attack.
- **[Medium] EVAL-12.** `params`/`from_params` are hand-indexed across three structs, and the omission-class bug is uncaught. See X2.
- **[Low] EVAL-13.** `cc2 board_value` calls `unwrap()` and panics on a width-0 board, while `features.rs` guards it (inconsistent panic-safety). `cc2.rs:196-217`.
- **[Low] EVAL-14.** The combo `floor((combo-1)/2)` formula vs the engine's `COMBO_TABLE` convention is unverified, off-by-one prone, and untested. `cc2.rs:269-271`.
- **[Low] EVAL-15.** `mini_spin_clears[lines.min(2)]` and `normal_clears[lines.min(4)]` silently clamp impossible states. `cc2.rs:264-268`. Fix: `debug_assert!` the bound.
- **[Low] EVAL-16.** `Cc2Weights::DEFAULT` is 21 unverifiable magic numbers ("verbatim from default.json," with no vendored `default.json`). `cc2.rs:62-82`. Fix: vendor it as a fixture and assert, or cite the upstream hash.
- **[Low] EVAL-17 (mine).** The `weights.rs` doc says "9-feature" and "DT-20's 9th feature omitted," but the struct has 10 fields, and `dot()` does `sum.round() as i32` (NaN becomes 0). Minor doc and robustness note.

### Search and state: `beam.rs`, `best_first.rs`, `greedy.rs`, `search/mod.rs`, `state.rs`, `BEAM.md`

- **[High] SR-1.** `BestFirstPlanner` never time-slices at the shipped budget. `best_first.rs:237` loops `while this_call < EXPAND_CHUNK(1024) && expanded < node_budget`, and the production `node_budget=150` (`registry.rs:48`) binds first, so the whole search runs in one `plan` call and `NeedMoreBudget` is never returned. The WASM cooperative-yield contract is violated for the only operating point that ships (WASM in a browser is a primary target via tetr-embed). Fix: make the yield unit independent of the total budget (for example `EXPAND_CHUNK=64`) and test that a depth-2-or-more search at the production budget yields at least once. (Verified. A reviewer said Critical; High here, since it is responsiveness, not a crash.)
- **[High] SR-2.** `SearchBudget.nodes` is honored by no planner: greedy ignores it, beam ignores it, best-first substitutes a constructor field, and `SearchBudget::beam` hardcodes `nodes: 0`. A caller tuning `nodes` to bound latency gets a silent no-op. `mod.rs:43-55`. Fix: drive best-first's total from `budget.nodes` (delete the field), or remove `nodes` from the trait.
- **[High] SR-3.** Roughly 250 lines of strategy scaffolding are duplicated between beam and best-first (`best_plan` byte-identical, `placements`, `StateKey`/`RootFingerprint`, child-construction, stale-run). See X7. The module docs explicitly anticipate a third Tier-2 planner, which would be a third copy.
- **[Medium] SR-4.** `BeamPlanner` is a god-struct (enumerate, fork, classify, batch-marshal, speculate, back-up, time-slice), with a primitive `meta: Vec<(usize, Reward, SearchState, f32)>` tuple. `beam.rs:115-432`. Fix: name the tuple; split expand, score, fold, and truncate.
- **[Medium] SR-5.** `SearchState` exposes all bookkeeping fields `pub` (`board`, `active`, `hold`, `queue`, `bag`, `b2b`, `combo`), so the careful `commit_*` invariants are bypassable by direct writes, and planners hand-assemble identity keys from raw `board.columns()` and rotation discriminants. `state.rs:114-145`. Fix: make `bag`, `b2b`, and `combo` private with a `state.identity_key()` seam.
- **[Medium] SR-6 (mine).** `commit`, `commit_with_next`, and `commit_placement` duplicate the classify, lock, `update_b2b`, `update_combo`, spawn core. `state.rs:198-274`. Fix: extract a private `lock_and_transition`.
- **[Medium] SR-7 (mine).** The b2b-eligibility logic is duplicated and textually divergent between `update_b2b` (`(Mini,_)`) and `compute_reward` (`(Mini, 1|2)`), despite a "kept in sync" comment. Latent (a mini cannot clear 3), but it is exactly the desync the comment fears. `state.rs:296-305` vs `eval/mod.rs:237-250`. Fix: one shared `b2b_eligible(t_spin, lines)` predicate.
- **[Medium] SR-8 (mine).** `commit` with an empty queue silently no-ops the spawn while still locking, leaving `active` pointing at a now-locked piece, an unenforced precondition. `state.rs:198-208`. Fix: return a status, or require `commit_with_next` when the queue may be empty.
- **[Medium] SR-9.** `BEAM.md` has substantial drift, and it is marked normative. See X8.
- **[Medium] SR-10 (was a reviewer's High; refuted as a bug).** The speculative hold-swap is hand-inlined in `expand_speculative` (`beam.rs:337-340`), divergent from the canonical `commit_placement`, and the speculative-hold bag/b2b path is untested. The correctness is fine: `commit_with_next` deals exactly `next_piece`, and the swapped-in piece came from hold (see "Claims I downgraded"). Fix: a `SearchState` speculative-transition method, plus a bag-accounting test.
- **[Low] SR-11.** Beam's reward round-trips through f32 even on the non-speculative `spec_weight == 1.0` path (`(reward.0 as f32 * 1.0).round()`), needlessly lossy in the determinism-critical score path. `beam.rs:282`. Fix: short-circuit when `spec_weight == 1.0`.
- **[Low] SR-12.** `best_plan` is typed `Option<PlacementPlan>` but always returns `Some` and indexes `root_best[0]`, safe only via a non-local "roots never empty" invariant. `beam.rs:371`, `best_first.rs:254`. Fix: a non-optional return plus a `debug_assert!`.
- **[Low] SR-13.** Two beam topped-out tests are vacuous: one asserts a tautology (`plan.is_some() == !placements.is_empty()`), the other wraps its only assert in `if placements.is_empty()`. `beam.rs:525-572`. Fix: construct a guaranteed-empty movegen and assert `Done(None)` unconditionally.
- **[Low] SR-14 (mine).** `BagState::index_of` linear-scans `PieceType::all()` plus `.expect()` on every `bit()` call (the hot bag path). `state.rs:58-67`. Fix: `piece_type as usize`. Also `RootFingerprint` converts the `SmallVec` queue into a `VecDeque` per poll (`beam.rs:91`) needlessly.
- **[Low] SR-15 (mine).** `rebuild_active` passes synthetic kick args, relying on `rotate_to` ignoring all but the target rotation, which couples to an impl detail. `state.rs:376-388`.

### NN and training: `nn/lib.rs`, `nn/Cargo.toml`, `nn/README.md`, `train_value_net.py`, `pyproject.toml`

- **[High] NN-1.** `read_f32` reinterprets raw bytes as little-endian f32 without checking the declared `Dtype`. `nn/lib.rs:204-217`. A future f16/bf16/f64 export (common for NN weights) loads "successfully" as garbage with no error. (I confirmed this independently: the safetensors `view.dtype()` is never read.) Fix: `if view.dtype() != Dtype::F32 { return Err(Shape(...)) }`. Also validate `view.shape()` against `[d_in, d_out]` so a transpose-convention flip is caught.
- **[Medium] NN-2.** `BurnEvaluator` does not override `evaluate_cols`, so it allocates a `Board` per placement on the greedy NN path. Same root as X1 and EVAL-1.
- **[Medium] NN-3.** `Value((raw*scale).round() as i32)` has no NaN/Inf guard, and `Inf` becomes `i32::MAX`, which makes one move look infinitely good. `nn/lib.rs:355,390`. With NN-1, a corrupt blob silently mis-plays instead of failing. Fix: reject or clamp non-finite values.
- **[Medium] NN-4 (mine).** `value_scale` defaults to `1.0` and is never set on the shipped paths (`nn_ai_controller`, registry `from_safetensors`), so the net's raw output composes directly with `compute_reward`'s hundreds-scale `Reward`. Commensurability is assumed, unverified, and untested. `nn/lib.rs:317` plus `registry.rs:216`. (Related to EVAL-5.)
- **[Medium] NN-5 (mine).** `evaluate_batch` calls `board.to_array2d()` twice per item, once for features (`:380`) and again inside `compute_reward` (`:391`), which is pure waste in the batch hot loop (`compute_reward` only needs emptiness, which `BitBoard` answers directly).
- **[Medium] NN-6 (mine).** The "bit-identical" `evaluate_batch == map(evaluate)` guarantee is backend-dependent: true for the ndarray CPU path (tested), but `BurnEvaluator<Gpu>` inherits the same default and the claim is unverified and likely false there (float nondeterminism). It is documented as absolute. Fix: scope the guarantee to the CPU backend in the doc.
- **[Medium] NN-7.** `lib.rs` carries five responsibilities (architecture, encoding, safetensors IO both ways, evaluator, controller factory), with no seam to mock inference. Fix: split `model.rs`, `safetensors_io.rs`, and `evaluator.rs`, plus a small `BoardValue` trait for tests.
- **[Medium] NN-8.** Python `train()` overloads one `seed` for init and shuffle (and the sampler), is a do-everything function, and duplicates the `/= FEATURE_SCALE` normalization across both dataset paths (the Rust contract). Fix: separate the seeds; extract `make_model`, `make_optimizer`, and `train_epoch`; one `normalize()`.
- **[Low] NN-9.** No golden input-to-output test. The tests only assert `batch == scalar` on random weights, blind to encoding and scale drift. Fix: a hand-computed `features_to_input` vector plus a `to_safetensors` then `from_safetensors` then `forward` round-trip.
- **[Low] NN-10.** `expect` on `Linear.bias` in the export path (`:181`, which fires post-training) and on the batch f32 buffer (`:385`). Fix: a `LoadError`, and document the backend assumption.
- **[Low] NN-11.** "JAX-exported" doc on the loader, but the asset is from the `distill` bin. `:149,328`. See X8.
- **[Low] NN-12.** `Cargo.toml` is missing `description`, `license`, and `publish=false` versus sibling crates (the heavy Burn tree makes an accidental publish costly).
- **[Low] NN-13.** `features_to_input` uses a manual `while` loop instead of `std::array::from_fn` (the `distill` bin uses the idiom). `:288-294`.

### Research harness: `lib.rs`, `behavior.rs`, `cc2.rs`, `cli.rs`, `bin/*`

- **[High] RES-1 (was a reviewer's Critical; reframed).** The versus "attack" counts only spillover after self-cancellation (`a_attack += leftover` only when `leftover > 0`, `lib.rs:644-648`), and `decide_versus` breaks ply-cap ties on it. A survival or digging bot that cancels its incoming queue records zero offense. I verified this. Spillover-after-cancel is the standard "garbage sent to opponent," so it is defensible, but it is the sole tiebreaker, it is named "attack," it is undocumented as net vs gross, and it is untested, so it silently decides win-rate by a quantity that may not be the intended fitness. Fix: track gross attack produced separately, document which one decides matches, and test that a bot under pressure that clears lines records non-zero offense.
- **[High] RES-2.** No result persistence or checkpointing on multi-hour runs. `bench_marathon` and `behavior` compute the full `Vec` in memory and `println!` a summary at the end. These jobs have been interrupted twice, so a kill at seed 23/24 loses everything. Fix: stream per-seed JSONL (append and flush), and resume by skipping seeds already present.
- **[High] RES-3.** Metric stdout lines carry no config, seed count, weights, or git SHA, so the "reproduces every number" claim holds only if you remember the exact env. The README "SOTA snapshot" is unprovenanced. Fix: emit a structured config header (including `CARGO_PKG_VERSION` and git SHA) before metrics, and make the stats serializable.
- **[High] RES-4.** Cheese construction reaches into tetr-core's `#[doc(hidden)]` test-only `set_cell` (`lib.rs:345`, `behavior.rs:177`), a deliberately unstable seam, while the public `insert_garbage` sits right there. I verified `set_cell` is `#[doc(hidden)]` test-only at `api.rs:258-268`. Fix: standardize cheese on the public garbage API.
- **[High] RES-5.** CC2 hold/queue bookkeeping is copy-pasted three times in `cc2_baseline.rs` (`run_one`, `run_downstack`, `run_versus`), the most desync-prone part of the bridge. Fix: extract a `Cc2Game` mirror (X7).
- **[High] RES-6.** `run_versus` reimplements `lib::play_versus` (turn order, cancellation, garbage, the magic hole seed) instead of using the `VersusEngine` seam built for it, so the two can silently disagree on garbage rules, defeating the "fair comparison" purpose. Fix: a `trait VersusOpponent` plus one match loop (X7).
- **[High] RES-7.** `lib.rs` at 961 lines is six or more modules (scoring, marathon, downstack, versus, seeds, bots). Fix: split it (the section-divider comments are the admission).
- **[High] RES-8.** OCP: a new metric, bot, or mode means editing core plus each bin's stringly-typed `match`, an unknown `BOT=` silently falls to a default, and `DOWNSTACK`/`VERSUS` are boolean-trap envs. See X5.
- **[High] RES-9.** `cli.rs`'s `SplitMix64`, `next_unit`, and `env_f32` are dead, the PRNG is inlined three times in `lib.rs`, and the module doc claims it is the centralized home. Fix: route through it and delete the copies (X7), or delete the dead code.
- **[Medium] RES-10.** `MAX_PIECE_FRAMES=256` is defined twice (X3). `lib.rs:447`, `behavior.rs:26`.
- **[Medium] RES-11.** The per-game event loop is reimplemented four times, and `play_scenario`'s copy already diverges (it counts pieces differently). Fix: a `step_engine_once` helper.
- **[Medium] RES-12.** CC2's B2B is re-derived by hand (`cc2_baseline.rs:261-265`) while our side uses the engine's authoritative flag. Same `attack_lines` table, different B2B inputs, which is a systematic APP bias that undermines "13.83 vs 14.50." Fix: share one B2B state machine, or assert the hand-rolled one against the engine.
- **[Medium] RES-13.** Over-broad `pub` surface (`ClearInfo`, `cheese_holes`, `versus_hole`, ten bot factories). Fix: `pub(crate)` internals, and a registry for the bots.
- **[Medium] RES-14.** Hard-coded `/tmp/cold-clear-2/...` CC2 binary default (cleared on reboot). `cc2_baseline.rs:473`. Fix: resolve on `PATH`, or require `CC2_BIN`.
- **[Medium] RES-15.** The README sells `cc2-baseline` as a clean "TBP referee" but omits the source's loud "versus is NOT FAIR, infra only" caveat, so a reader could publish a bogus win-rate. Fix: mirror the caveat, or gate the mode.
- **[Medium] RES-16.** `Sim` in `cc2_baseline` is a third hand-rolled board/garbage/clear impl kept in lockstep with `Board` by comment. Fix: drive CC2's mirror through a headless `Engine` where possible, else unit-test `Sim` against `Board`.
- **[Low] RES-17 to RES-22.** Determinism claim untested. Near-zero harness tests (only `decide_versus`). `cli.rs` is misnamed (no CLI, all env). Scattered XOR salts (one duplicated across two files). Mean-folding duplicated five times (behavior omits fields). `gen.next().unwrap()` appears six times.

### App integration: `registry.rs`, `model_select.rs`, `sandbox.rs`, `runner/*`, `movegen.rs`, `plan.rs`, `controller.rs`, `embed`

- **[Critical] APP-1.** `include_bytes!` on a generated asset is a build-time landmine. `registry.rs:191-194,211-214` plus `embed/src/lib.rs:122-125`. The runtime "degrades to linear if the blob fails to parse" guarantee covers only a parse failure of already-embedded bytes. If `value_net.safetensors` is absent, `cargo build --features nn` fails to compile (a fresh clone, or CI that builds before running the distiller, breaks). Fix: commit the asset as a tracked file, or load it at runtime so a missing file degrades gracefully as the doc promises, or use a `build.rs` that emits `compile_error!` with remediation.
- **[High] APP-2.** "Adding a model is one entry" overclaims OCP: every model edits the central `Default` impl. `registry.rs:109-241`. Acceptable as a fixed in-house factory, but then fix the doc, or expose `register(label, build_fn)` and let `#[cfg(feature="nn")]` models self-register beside the NN code. At minimum, extract `fn build_beam_dt20() -> AiController` so `Default` is a flat, individually testable list of `(label, fn)` pairs.
- **[High] APP-3.** Six copies of `SearchPolicy` construction, plus a duplicated `include_bytes!`, plus a redundant `as Box<dyn Policy>` five times. See X7. Fix: a `search_controller(planner, eval, budget)` helper, and one module-level `const MODEL`.
- **[High] APP-4.** Silent NN-to-linear fallback, with the error dropped and the HUD mislabeled. See X6. `registry.rs:203,232`.
- **[Medium] APP-5.** Stringly-typed and positional model identity, a feature-flag-dependent index, and `select()` silently ignoring out-of-range. See X5.
- **[Medium] APP-6.** The registry and `model_select` have zero tests, and identity fused with construction blocks testing the NN entry's registration without the blob. See X9.
- **[Medium] APP-7.** Dead defensive branches (`selected_controller`'s `None` arm, the `"?"` label) mask the structural non-empty invariant and add a seventh `AiController::new` site. `registry.rs:96-106`. Fix: encode the invariant in the type.
- **[Medium] APP-8.** Registry SRP: research tuning data (`ATTACK_BOARD_PARAMS`, beam consts) lives in the UI catalog. See X3.
- **[Medium] APP-9.** `selected_controller()` runs a blocking safetensors parse plus tensor alloc inside an exclusive-world `OnEnter(Playing)` system, a main-thread hitch on session start that scales with model size. `sandbox.rs:118-133`. It is fine now (19 KB), but cache the built controller or load off-thread if models grow.
- **[Medium] APP-10.** A label inconsistency breaks "compare like-for-like": the linear greedy entry is tagged "(greedy)," but the NN greedy entry is not, though both are 1-ply. `registry.rs:114` vs `:190`. Fix: uniform `Planner - Evaluator` labels, ideally from typed kind enums. (The runtime contract is fine: I verified all entries thread `h.imperfection`, `h.reaction`, and `DEFAULT_AI_SEED`, including the value-net beam at `:227`.)
- **[Low] APP-11.** The `native.rs` deletion removed the only `DecisionRunner` contract test (off-thread equals sync, supersede, reclaim), so the seam's substitutability is now asserted only by `SyncRunner`. The deletion is defensible (a dead reference impl), but consider keeping a parametrized `#[cfg(test)]` contract test.
- **[Low] APP-12.** `controller.rs:178` calls `placement_to_inputs(&obs.board.to_array2d(), ...)`, which materializes a dense `Board` per decision, and `plan.rs` was not generalized to `Occupancy` the way `movegen.rs` was. Fix: generalize it, removing one `to_array2d` round-trip.
- **[Low] APP-13.** The "(ported)" registry entries inline competitive numbers that will go stale (X8). The remaining runner-doc "would drop in" phrasings are well handled: I confirmed zero dangling `ThreadedRunner`, `runner::native`, or `web.rs` references.

---

## Part 3: Claims I downgraded or refuted (be skeptical of reviews too)

1. The best-first transposition table is not unsound (a reviewer said High). Because `board` is in the `StateKey`, `leaf_value` is identical for any two paths colliding on a key, so the stored `score` orders the same as `acc_reward`, and a higher `acc_reward` dominates for every shared continuation. Pruning the lower path is correct, and `>=` is consistent with the "first max wins" rule. The bag is not in the key, but best-first drops empty-queue nodes as leaves, so it never matters. The real residue: the soundness rests on a subtle, uncommented, untested invariant (Low; see SR-12 territory). Add a transposition-collision test.
2. The beam speculative re-deal does not corrupt the bag (a reviewer said High). `commit_with_next` deals exactly `next_piece`, and the swapped-in piece came from hold (already dealt). The real residue: the hold-swap is hand-inlined and the path is untested (SR-10, Medium DRY).
3. The research `a_attack` is not a clear correctness bug (a reviewer said Critical). Reframed to RES-1 (High, metric definition): spillover-after-cancel is the standard measure, and the issue is that it is the undocumented, untested sole tiebreaker.
4. The NN README's "two Watch-AI registry models" is accurate (a reviewer flagged it as overclaim). The reviewer grepped `crates/tetr_online/` and missed that the game crate is at the repo root (`src/ai/registry.rs`), which has exactly two `#[cfg(feature="nn")]` entries (`:189-234`). Cut this finding.
5. `insert_garbage` is High, not Critical (ENG-1): a real defect plus a false doc, but limited blast radius.

---

## Part 4: What is good (do not "fix" these)

- The trait seams. `Occupancy`, `Evaluator`, `Planner`, `Policy`, and `DecisionRunner` are clean and object-safe. `Occupancy` deleted duplicated corner-collision code and lets T-spin and movegen run on either representation. The NN integrates through the same path as the hand-crafted bots. `ModelEntry { label, build: Box<dyn Fn> }` is a good factory that could support open registration.
- Single source of truth where it counts. `lock_piece`, `score_placement`, `fold_combo`, and `compute_reward` (as a free function the NN reuses) are the right centralizations. The ask is to apply that instinct to the 6x, 4x, and 3x copies elsewhere.
- Determinism is taken seriously: a `>`-first-max plus a stable sort plus an insertion-order heap tie-break, with a plan-twice test in both planners and a speculation-determinism test.
- The differential and invariant testing is good. The bitboard-vs-engine suite, `commit_placement`'s five hold-aware tests, the weights round-trip, and the depth-1-equals-greedy gate are the right invariants to pin. The gaps are which surfaces got covered (X9).
- Honest negatives. The NN post-mortem README and `run_versus`'s in-source "NOT FAIR" caveat are documentation that saves the next engineer weeks. `sandbox.rs` is a good model: a documented rationale plus a real test suite.

---

## Part 5: Suggested fix order

1. APP-1 (build landmine). Unblocks anyone building `--features nn`.
2. X1 / EVAL-1 / NN-2 (override `evaluate_cols`, invert the trait). Recovers the branch's main performance goal for the shipped and NN bots, and removes the bit-identical-contract LSP surface.
3. SR-1 / SR-2 (best-first time-slicing, and the dead `nodes` budget). Fixes the WASM responsiveness contract on a primary target.
4. NN-1 plus NN-3 / ENG-1 / ENG-2 (dtype guard, finite guard, garbage game-over event, buffer-row count). The correctness and robustness cluster on attacker-controlled and ML-load paths.
5. X6 / APP-4 plus RES-1 / RES-3 (stop silent mislabeled fallback, make the metrics observable and provenanced). So you can trust what you are measuring and watching.
6. X2 (single-source feature and weight vectors, plus safetensors metadata). Removes the largest train/inference-skew and OCP cost before more features land.
7. X7 / X8 / X9 sweeps (de-dup, doc reconciliation, fill the test gaps). Add the missing differential tests (`evaluate` vs `evaluate_cols`, `to_array2d` round-trip, T-slot, golden NN vector, transposition collision) as the regression net for everything above.
