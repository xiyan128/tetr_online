---
id: T15
title: Completed-Q target transform + round-0 two-headed net training
labels: [wayfinder:prototype]
status: open
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
