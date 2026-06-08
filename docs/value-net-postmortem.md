# Value-net post-mortem (`tetr-nn`)

A record of *why the learned-eval path never beat the hand-tuned CC2 evaluator* —
and the playbook for reviving it. **The code was removed** in the value-net prune
(branch `prune/value-net`): the `tetr-nn` crate, the `nn` cargo feature on the game
and tetr-embed, the two Watch-AI registry models, the `distill` bin, the Marathon
NN comparison, and the JAX trainer under `training/`. To recover any of it, check
out the parent of that prune commit (the `feat/tetr-nn` tip) — nothing here needs
to be rebuilt from scratch.

Read the lesson below before reviving this.

## What it was

- A tiny MLP value net: `NUM_FEATURES (8) → 64 → 64 → 1`, regressing a board's
  value. `BurnEvaluator` wrapped it behind tetr-core's `Evaluator` trait, so the
  same beam / best-first search could score with it.
- **Deterministic CPU inference** (Burn's `ndarray` backend) — matched the
  in-browser path and kept replays/benches reproducible. Weights loaded from a
  baked-in `value_net.safetensors` via `include_bytes!`.
- **Feature-gated `nn`** on the game and **native-only** for tetr-research, so the
  wasm game never pulled it unless built `--features nn`. That kept the heavy Burn
  dependency tree off the size budget — but it *was* heavy: it dominated
  `Cargo.lock` (~4k lines of transitive deps). That weight, against a net that
  never beat the hand-tuned eval, is why the path was pruned.

## The lesson (the whole point of this file)

**The shipped net was distilled from DT-20 — the *weak* linear survival eval — so it
could only ever mimic a weak bot.** The eval ablation measured the linear eval at
**~0.05–0.30 APP** versus the CC2 eval's **~0.60**. A value net trained to match
DT-20 inherits DT-20's ceiling; it cannot reach CC2 by copying a weaker teacher.
That, not the net architecture, is why the value-net path stalled.

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

The training code that existed before the prune: the JAX trainer
(`training/train_value_net.py`) was the real training path, and `tetr-research`'s
`distill` bin was the Rust bootstrap (DT-20 distillation) that produced the shipped
`value_net.safetensors`. Both are recoverable from the pre-prune history. (Two even
earlier attempts — a raw-board "attack net" fed by self-play, and the self-play data
generator — were already removed as dead code before this; this file is their
post-mortem too.)
