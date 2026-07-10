---
title: Leapfrog — a learned bot that beats the hand-tuned champion
labels: [wayfinder:map]
created: 2026-07-07
updated: 2026-07-10
---

# The goal

Build a Tetris-versus bot whose strength comes from **learning**, not
hand-tuned evaluation, and race it against the repo's hand-tuned champion
(`probe-tp128d9`: the CC2 evaluator under a wide, deep beam — 0.8225
attack-per-piece on the held-out solo battery, dominant in versus). The bot
must also match the champion at single-player. CC2 may teach round 0 and serve
as a fixed opponent/yardstick afterwards — nothing more.

# The system (all of it)

One loop, four parts. Anything not listed here does not exist in the live
system.

1. **Net** — a small convnet: your own board + a few scalar features →
   win/draw/loss. Nothing else: no policy head, no auxiliary heads, no
   opponent input (bots are blind to the opponent's board by rule).
2. **Search** — the existing beam (`crates/tetr-core/src/ai/search/beam.rs`),
   ranking placements by the net's win probability. The SAME plain beam runs
   in datagen and in every duel (a same-seed driver-vs-harness test pins
   this). The beam enumerates every legal placement, so it needs no learned
   guidance. Width/depth is the compute knob.
3. **Data** — self-play games. Each decision stores one row: the board the
   mover chose (post-placement observation) and, once the game ends, who won
   (`z ∈ {+1, 0, −1}` from the mover's perspective). No search scores, no
   counterfactual children, no score-scale bookkeeping — those fields caused
   most of the historical defects and taught nothing the outcome doesn't.
4. **Gate** — a new net is kept only if it (a) beats the current best net in
   a seed-paired duel and (b) clears a fixed floor against the CC2 anchor
   (`beam:cc2@w8d5`; the floor is calibrated by round 0's baseline read).
   Both duels are one command each and reproduce from `(commit, seed)`.

A **round** is: play games → train on them (plus a replay of earlier rounds)
→ gate. One command, tens of minutes:

```
cargo build --release -p tetr-research           # the driver shells out to this
cd python
uv run python -m tetrnn.round --round 0 --scratch ~/tetr-rounds
uv run python -m tetrnn.round --round 1 --scratch ~/tetr-rounds     --incumbent ~/tetr-rounds/r0/net
```

Round 0 trains on CC2-vs-CC2 games (the only CC2 supervision) and its anchor
duel is recorded as the BASELINE (it calibrates the anchor floor; there is no
incumbent to beat yet). Every later round trains on the incumbent's own games
and passes `--incumbent` explicitly — the previous PROMOTE round's net. Code:
driver [python/tetrnn/round.py](../../python/tetrnn/round.py), trainer
[train.py](../../python/tetrnn/train.py), data plant
[datagen.rs](../../crates/tetr-research/src/datagen.rs).

# Why so simple

The previous, richer design (policy heads + action-slot heads + soft search
targets + vehicle grammars + provenance machinery) produced four
campaign-voiding silent defects in two days, and its central mathematical
object — the "completed-Q" policy target — was shown to be degenerate
(near-one-hot where scores separate, near-uniform at exactly the life-or-death
decisions). Post-mortem pattern: every defect came from two things sharing one
channel (two rankers behind one name, two score scales in one field, two
observation roles in one whitening, two index spaces for one list). This
design removes the dualities instead of policing them. The full forensic
record lives in [archive/](archive/) — read it before re-adding anything.

Key prior evidence the simple design leans on:
- A value-only net + this beam already **tied the champion's evaluator at
  matched search** in the 2026-06 campaign. The champion's moat is search
  throughput, not evaluation quality.
- The beam enumerates all placements — a policy prior steers nothing here
  (measured: root filtering does not cheapen the beam; the per-child filter is
  strictly more expensive than none).

# Status

- **2026-07-10**: simplification reset. No accepted incumbent; rounds 0-11 of
  the old design are quarantined history ([archive/](archive/) — hash-bearing
  index of every artifact and claim in
  `archive/assets/T19-evidence-index.v1.json`). The demolition removed the
  policy/slot machinery, the target-transform stack, and the vehicle grammar.
- **2026-07-10, round 0 ran**: the whole round took **4.1 minutes** (600
  CC2 games in 69s, 3 training epochs in 61s, 48 anchor games in 114s) —
  against the old design's 5+ hours. Baseline: holdout CE 0.679 / accuracy
  0.56; **anchor 0-48 vs CC2** — outcome-only value from 600 games is a thin
  evaluator, which is exactly the open question the loop now measures.
  Ledger: `<scratch>/rounds.jsonl`.
- **Round 1 (200 self-play games, from-scratch on mix): KEEP_INCUMBENT** —
  the candidate lost 6-42 to its own incumbent, replicating the 2026-06
  finding that from-scratch-on-mix regresses. The gate did its job on the
  first try.
- **Round 2 (same, + `--finetune`): KEEP_INCUMBENT, 3-45** — fine-tuning
  didn't rescue it and holdout CE rose: the weak self-play rows themselves
  degrade the evaluator.
- **The scale probe + the phase diagnostic (the night's finding):** a
  5000-game round-0 (8.3× data) moved holdout accuracy only 0.559→0.567,
  still anchored 0-48, and **lost 8-40 head-to-head to the 600-game net**.
  Why: measuring the net's cross-entropy by game phase shows the outcome
  label is a **pure coin flip for the first 60% of a mirror game**
  (CE ≈ 0.694/acc 0.50) and only becomes informative near death (acc 0.81
  in the last fifth). The net learns everything the labels contain — but
  mid-game board ranking, which is what the beam's evaluator does at move
  time, is exactly where balanced-mirror outcomes carry no information.
  **Outcome-only z from balanced mirror games is an information-starved
  target for a move-ranking evaluator.** This subsumes rounds 1-2's
  failures and closes the data-scaling path.
- Next levers, one at a time, both env-truth and machinery-free: (a) a more
  decisive venue for datagen (e.g. rain period 4 — every state sits closer
  to the outcome, and games get cheaper); (b) unbalanced game pairs (e.g.
  CC2@w8 vs CC2@w2 — a mid-game advantage then shows in the outcome, so
  mid-game boards become predictive). If neither lifts mid-game signal,
  the principled escalation is TD-style bootstrapped value targets.

# Open questions (one lever at a time)

- Does value-only training on played states compound round over round? (The
  2026-06 solo campaign compounded once, then stalled — replay buffers helped;
  that is the first lever if it stalls here.)
- ~~Purity endgame~~ RESOLVED by the reset: the net bot's search score is
  `z_scale · z_hat` with zero reward terms, which makes the beam's one
  remaining hand constant (`SPEC_DECAY`, a discount on speculative-ply
  *rewards*) inert on the learned path. The deployed net path carries no
  hand-tuned terms; width/depth remain the allowed compute knobs.
- Self-play datagen cost: CC2 datagen runs ~31k games/hr, net-leaf datagen
  ~90 games/hr at 6 workers (the net pays a batched forward per sibling
  group; CC2's linear eval is nearly free). Until that gap earns real perf
  work, the cadence knob is games-per-round (round 1 ran 200), not width.
- Single-player from the same net: planned as a same-seed self-race venue
  (two boards, no interaction, more attack wins) so the identical loop covers
  both modes; blocked until versus compounds.
