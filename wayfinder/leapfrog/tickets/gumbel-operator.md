---
id: T12
title: Build the minimal Gumbel expectimax search vehicle (probe-grade)
labels: [wayfinder:prototype]
status: open
assignee:
blocked-by: []
---

## Question

Build the smallest Gumbel-expectimax search over tetr-core's true model that is sufficient to (a) measure Gate-0b (low-width G_π for the *actual* operator, not the beam) and (b) seed the eventual deployed vehicle. This is the instrument the whole design freeze rests on — the prior campaign's authorizing number was for a beam, not this operator, so it must exist before the premise can be tested honestly.

Scope (probe-grade, per the T02 memo build-vs-reuse call — native Rust, no per-node FFI):
- Root: Gumbel-top-m sampling (m=8–16) over the net policy logits + Sequential Halving (n=32–128 sims), score `g(a)+logit(a)+σ(Q̂(a))`.
- Non-root: deterministic completed-Q improved-policy selection (NOT PUCT).
- Chance nodes: exact enumeration of the bag remainder (uniform-without-replacement) and salted-hole PRNG; backup = CVaR_α (α≈0.25) with `α→mean` as an ablation flag. Replaces SPEC_DECAY.
- Value: terminal-WDL negamax through decision nodes, CVaR-expectation through chance nodes.
- Reuse: `beam.rs` SearchState fork / `commit_placement` / transposition / afterstate+bag loop; the `Mind` Policy trait; the ANE/BLAS eval seam. Reference mctx `gumbel_muzero_policy` + `qtransforms` and diff the port against it on toy states.
- Emit `π'` (completed-Q policy target) and per-root Q for the G_π/target-extraction probes.

Needs a two-headed net (value + ~34-way policy). The `round0` fixture already has a policy head, so the probe can run on it before the campaign net exists.

Output: a working `Arm` (e.g. `gumbel:<dir>@m8n64`) wired into `duel`/`gate`, validated against mctx on toy positions, with its per-move eval count and wall-clock measured (feeds T03).

## Scope correction (2026-07-08, from code recon — the panel over-specified this)

Investigating the build surfaced that **the panel conflated two separable things**, and a from-scratch MCTS is NOT the near-term need:

1. **The beam already produces correct per-root Q** (`root_scores()` = max-backed-up value per `hold_placements` root; single-agent max-backup is optimal for the G_π-vs-policy setup, which searches only my board — the net is opponent-blind, like the existing `beam:` arm). So a *correct* value-backup search already exists.
2. **The shard format already stores per-child backed-up root score** (`shards.rs:60`, `DecisionRecord::from_served(meta, &[(&Obs, i32)])`). So the completed-Q *source* is already persisted.
3. **The panel's "completed-Q dodges fact (d)" is a TARGET-EXTRACTION fix, not a search-algorithm fix.** π' = softmax(logit + σ(completedQ)) over all roots is a **Python training-side transform over the stored root scores** — it does not require rewriting the search as MCTS. The beam's near-deterministic *argmax* is irrelevant because completedQ reads Q over *all* roots.
4. **Sequential Halving is a THROUGHPUT optimization only** (fewer evals for the same Q estimates) — it does not change whether search-of-this-kind beats the policy or whether the target is sound. Deferrable behind the datagen-throughput work (T03/T13).
5. **There is no self-play datagen driver on master** — only the shard *format*. The real near-term Rust piece is a **datagen driver** (plays net-guided games, writes shards + root scores), which overlaps T09 and is campaign infrastructure that should follow the design freeze.

**Revised plan:** v0 "operator" = the **existing beam restricted to the net's top-m policy roots** (correct, reuses tested code) + a **Python completed-Q → π' transform** over stored root scores. The from-scratch Gumbel-SH MCTS + CVaR chance nodes is a **deferred throughput/quality refinement**, not a pre-freeze blocker. Gate-0b's premise (does search beat the policy) is measured by `duel beam:<M>@w.. vs policy:<M>` **today** (see T07) — no new operator required for the first read.

This ticket is therefore **downgraded**: it is no longer a design-freeze blocker as a from-scratch MCTS. It re-scopes to "the completed-Q target path" and folds the SH-MCTS into post-freeze throughput work. T07 (Gate-0b) unblocks immediately (uses the beam).

## Measurement finding (2026-07-08): the action-indexed policy head is load-bearing for throughput

Built the root-filter seam (`BeamPlanner::with_root_filter`, byte-neutral default — 280 core tests green) + the `guided:<dir>@m<M>w<W>d<D>` arm (policy top-m roots, TP beam, net leaf). Measured: guided m12w8d5 mirror ≈ **30 games/hr** (trainer-contended) — NO throughput win over the plain net beam. Two structural reasons the panel and I both missed:

1. **Root-filtering cannot cheapen a beam**: interior generations still fan every child (~68/node), and width-truncation already bounds interior work regardless of root count. Root restriction only trims generation-1 evals (~56 of ~2,200/move).
2. **A per-child policy head can never cheapen search**: P and V share one forward, so *ranking* a child costs exactly what *evaluating* it costs. Filtering-by-policy saves nothing when the policy requires a per-child forward.

**Consequence (shapes T12 v1 + round-1):** the design freeze's "~34-way policy head" is not a nicety — it is THE eval-count lever. An **action-indexed head** (fixed slots: rotation × column × hold ≈ 80) lets ONE forward of the *parent* rank all its placements, so only the top-m ever get committed+evaluated → evals/move drops by the fan factor (~68× per node at full fan). This is what makes both the deployed vehicle's ~100 ms budget and self-play datagen cost feasible, independent of (and multiplicative with) the T13 per-eval fix.

The `guided:` arm remains valuable as a **strength-per-width** instrument (does the prior's top-m lose anything at matched width? — the Gate-0a question at system level; duel running). The action-head is the next net change: keep the per-child value path, add the 80-slot policy head trained from the same shards (children map to slots via their Placement (rot, col, hold)) — needs the Placement recorded per child in shards (currently NOT stored — shard schema addition required: a `child_slot` u8 tensor).

## Strength receipt (2026-07-08): the m12 restriction costs NOTHING at matched width

`guided:round0@m12w8d5` vs `beam:round0@w8d5` (same net leaf, seeds 850M, 6 CRN pairs): **6-6-0 dead even** (end reasons 7 topout / 5 escalation). The policy top-12 contains everything the full ~68-root beam needed — Gate-0a's coverage finding confirmed at system level, with the search in the loop. The guided vehicle loses no strength; the action-indexed head will make the SAME restriction ~fan-factor cheaper. Mechanism validated; the remaining vehicle work is the 104-slot head.
