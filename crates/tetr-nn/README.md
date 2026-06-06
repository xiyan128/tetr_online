# tetr-nn

A small **Burn** value network for the Tetris bot — and, just as importantly, a
record of *why the learned-eval path has not (yet) beaten the hand-tuned CC2
evaluator*. Read the lessons below before reviving this.

## What it is

- A tiny MLP value net: `NUM_FEATURES (8) → 64 → 64 → 1`, regressing a board's
  value. `BurnEvaluator` wraps it behind tetr-core's `Evaluator` trait, so the same
  beam / best-first search can score with it.
- **Deterministic CPU inference** (Burn's `ndarray` backend) — matches the
  in-browser path and keeps replays/benches reproducible. Weights load from
  `assets/value_net.safetensors` via `include_bytes!`.
- **Feature-gated `nn`** on the game and **native-only** for tetr-research, so the
  wasm game never pulls it unless built `--features nn`. That keeps the heavy Burn
  dependency tree off the size budget — but it *is* heavy: it dominates `Cargo.lock`
  (~4k lines of transitive deps).

Wired into: two Watch-AI registry models, the tetr-embed surface, and
bench-marathon's NN comparison.

## The lesson (the whole point of this file)

**The shipped net is distilled from DT-20 — the *weak* linear survival eval — so it
can only ever mimic a weak bot.** This branch's eval ablation measured the linear
eval at **~0.05–0.30 APP** versus the CC2 eval's **~0.60**. A value net trained to
match DT-20 inherits DT-20's ceiling; it cannot reach CC2 by copying a weaker
teacher. That, not the net architecture, is why the value-net path stalled.

Two corollaries the ablation made concrete:

- **Eval ≫ search.** Swapping linear→CC2 moves APP far more than any search change.
  And **deep search *amplifies* a bad eval**: best-first + linear (0.05 APP) scored
  *below* greedy + linear (0.20) — more lookahead just optimizes harder toward the
  wrong objective. So a learned eval has to be genuinely good *before* search helps.
- A net distilled from a strong survivor's games sees **almost no dying boards**, so
  it never learns "danger ⇒ no future value" and tends to stack into a top-out (the
  death-coverage problem). Any future training set must cover near-death states.

## If you revive it

Do **not** distill DT-20 again. Options, roughly in order of promise:

1. **Distill a *stronger* teacher** — the CC2 / best-first bot — so the regression
   target is ~0.6 APP, not ~0.2. Cheapest path to a net that's actually worth running.
2. **Train on self-play returns** (discounted future attack) with deliberate
   death/danger coverage, so the net learns the survival signal a distillation can't.
3. **Raw-board input** (let the net learn its own features instead of the 8
   Dellacherie/BCTS hand-features) — highest ceiling, but needs far more data.

The JAX trainer (`../../training/train_value_net.py`) is the real training path;
`tetr-research`'s `distill` bin is the Rust bootstrap (DT-20 distillation) that
produced the current `value_net.safetensors`. (Two earlier attempts — a raw-board
"attack net" fed by self-play, and the self-play data generator — were removed as
dead code; this file is their post-mortem.)

## Known follow-up

`BurnEvaluator::evaluate_batch` reconstructs a dense `Board` per row via
`to_array2d()` just to extract features, undoing the bitboard hot-path win for the
NN beam. A `BoardFeatures::extract_from_cols(&[u64], _)` seam — mirroring how
`Cc2Evaluator` already scores straight off `columns()` — would remove the
reconstruction. See the perf-strike notes for the broader cols-path cleanup.
