---
id: T15
title: Completed-Q target transform + round-0 two-headed net training
labels: [wayfinder:prototype]
status: closed
assignee:
blocked-by: []
---

## Question

The actual "dodge fact (d)" fix (Python, `python/tetrnn`): transform stored per-root scores → completed-Q → `π' = softmax(logit + σ(completedQ))`, over all roots (smooth even at near-deterministic argmax). Plus round-0: regenerate the BC corpus with the F27 seat-alternation fix (folds T05), train the two-headed (value + ~34-way policy) net + SSL board-reconstruction aux head, export, verify PyTorch↔Rust goldens.

**The decisive test lives here:** does training on completed-Q targets *improve* the net (per-source policy CE down, held-out strength up) or *inject entropy* (the R1 trap that regressed the net even at G=0.900)? Pre-registered TARGET-EXTRACTION STOP. This is round-1's core question — the one Gate-0a/0b could not settle.

## Progress (2026-07-08, in flight)

**Python side BUILT + VERIFIED** (`python/tetrnn/{shards,targets,train}.py`):
- Shard reader: Rust→Python seam proven on the dev corpus (11,717 decisions / 710,108 children; checksummed round-trip; obs shapes verified).
- **Completed-Q transform** (`targets.py`): π' = softmax(logit + c·qnorm), qnorm = min-max over live roots, dead roots masked to 0, all-dead → uniform. 7 self-checks pass: **scale-free** (affine-invariant ⇒ kills the τ-non-transfer trap C6), **non-degenerate at near-deterministic argmax** (the fact-(d) dodge), monotone-in-Q, prior-logit responsive. **C_SCALE=12 calibrated ONCE** on 2,347 corpus decisions: softest c whose median N_eff (5.63) sits in the pre-registered band [2.5,6] (A5 discipline); saturation floor N_eff≈2 at c→∞ confirms real top-ties never one-hot.
- **Streaming trainer** (`train.py`): shard-by-shard (13 GB corpus never resident; game-aligned shards ⇒ shard split IS game-level split), grouped-softmax policy CE on π', WDL CE on the played child (z→{win,draw,loss}, net.rs head order), whitening from train children, per-epoch export via the proven contract. Velocity probes: MPS@64-decisions best (142s dev epoch); 256-batch and CPU both worse (508s/552s, contended).
- **Round-trip CLOSED**: dev net (40 games, 2 epochs) exports → Rust `Net::load` → `policy:` arm plays legal games (2-6 vs the prior 623k-decision fixture — sane for 40 games).

**Full round-0 training RUNNING** (2000-game corpus, 3 epochs, per-epoch exports). Trap re-learned: `| tail` swallows live stdout (the memory's pipe-swallow warning) — progress watched via the per-epoch export mtime instead. Evaluation battery pre-staged (`scratchpad/round0_battery.sh`): policy-vs-policy vs the prior round0 fixture, beam-vs-beam @w8d5, and fresh-seed G_π, seed regions 810M/820M/830M.

**The decisive read pending:** holdout pCE/top-1 trajectory + strength vs the prior BC net = the entropy-injection (R1) check for completed-Q targets at BC scale. The full R1 answer (does the loop COMPOUND) is round-1.

## Round-0 training complete (2026-07-08)

Full corpus (482 shards / 2000 games / ~585k decisions), 3 epochs, MPS: holdout pCE **2.117 → 2.086 → 2.071** (monotone, train/holdout track — no overfit, no entropy blow-up), top-1 vs search argmax ~0.35 (not comparable to the fixture's 0.639 — this net trains toward the SOFT completed-Q π', N_eff≈5.4, by design), **z_std 0.172 ≥ the 0.15 value gate** (marginal — epoch 1 dipped to 0.141 then recovered; watch: played-child-only WDL training may be value-starved vs the prior 50/50 replay). Targets in band (N_eff 5.42). Exported per epoch; final at `scratchpad/round0_v2`.

Strength battery (policy-vs-policy vs the prior fixture; beam-vs-beam @w8d5; fresh-seed G_π) = the decisive entropy-injection read — running.

## Pre-registered battery interpretation (written BEFORE results)

1. **BC-vs-BC (policy:round0_v2 vs policy:fixture, 32 pairs):** the fixture is the prior campaign's BC (same 2000-game corpus scale, argmax-ish targets + τ machinery). Win/even = completed-Q targets at least match the old pipeline at BC (entropy-injection refuted at BC scale). Clear loss = the soft-target risk is real → try sharper c or per-source weighting before round 1.
2. **Beam-vs-beam @w8d5:** the leaf+prior combination test. Even is acceptable (the fixture's value head had more training machinery); win = strict improvement.
3. **G_π (beam vs own policy):** healthy ≥ 0.55; this authorizes round-1 expert iteration on this net.
4. **Watch-items:** TrueCap draws (historically never observed) = passivity regression from soft targets/played-child-only value → round-1 must add value bootstrapping or attack-awareness via z; z_std 0.172 is marginal vs the fixture's 0.458.

## Resolution — ALL THREE BATTERY READS GREEN; round-1 authorized

Battery ran 2026-07-08 (pre-registered interpretation above; seed regions 810M/820M/830M, receipts in runs/):

1. **BC-vs-BC:** `policy:round0_v2` **35-29** over the prior campaign's round0 fixture (64 games, all topout-decisive — the TrueCap-passivity worry did NOT materialize). The completed-Q pipeline ≥ the old τ-machinery pipeline at BC scale. **Entropy-injection refuted at BC scale.**
2. **Beam-vs-beam @w8d5:** **11-5 (0.69)** — the new net is a strictly better leaf+prior.
3. **G_π (fresh seeds):** **16-0** — search massively improves on the new policy; expert-iteration fuel confirmed for THIS net.

**Verdict: T15's question (do completed-Q targets improve the net or inject entropy?) is answered — they improve it, at BC scale, over the prior pipeline.** The full compounding question (round-1: does training on SELF-generated completed-Q targets keep improving?) transfers to the round-1 ticket/driver.

**Round-1 trainer note (pre-staged):** round-1 targets must include the CURRENT net's logits (π' = softmax(logit + c·qnorm) — the reanalyze form). `targets.py` already takes `logits=`; the trainer computes child policy logits in-batch (`heads[:,3]`) — wire those in for round-1 (per-batch live targets instead of precomputed). Also: ε-sampling for datagen diversity still deferred (seed diversity carried round-0).

Remaining in-flight (T12 v1, not T15): round0_v3 retrain with the 104-slot action head on the regenerated parent+slot corpus → guided-vehicle throughput + strength measurements.
