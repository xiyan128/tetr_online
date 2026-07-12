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
- **Lever (b) ran — unbalanced pairs (`datagen --opp-width`, CC2 w8 vs w2,
  2026-07-11): the signal lifted, the play didn't.** Wide seat wins 94.5%,
  labels became learnable everywhere (holdout acc 0.559→0.666; mid-game
  phase acc 0.49→0.62, late 0.81→0.92) — and the net still lost 0-48 to the
  anchor AND 0-48 to the balanced round-0 net. Diagnosis: unbalanced z is
  *across-game* signal ("does this board look like the stronger player's")
  — a beam needs *within-decision* contrast between sibling placements.
  Meanwhile the balanced net's one strong region (death recognition, acc
  0.81 late) acts as a survival heuristic — which is why it sweeps every
  net trained since.
- **The TD escalation ran (`train --td`, TD-Gammon-style bootstrapped
  targets, trainer-only): no play gain either** at α=0.5/4 epochs on the
  2k-game unbalanced corpus (0-48 anchor; 16-32 vs its plain sibling).
- **Scale × TD closed the outcome-only story**: 20k unbalanced games + TD
  = best classifier yet (holdout acc 0.693) and still 0-48 in play. Every
  outcome-only configuration failed identically: classifiers improve, play
  never does.
- **⭐ 2026-07-11 — ranking supervision works.** Shards (schema 3) now store
  ONE random non-best sibling per decision; training adds a logistic loss on
  `z_hat(played) − z_hat(sibling)` (`train --rank`). Unit-free, no scores,
  within-decision by construction. First run (2000 balanced CC2 games, CE +
  rank): **24-24 vs the CC2 anchor** (every prior net: 0-48) and **48-0 vs
  the best outcome-only net**. The beam's evaluator needs exactly this
  signal; outcomes alone measurably cannot supply it. Round 0's ranking
  comes from CC2's search (purity: round-0 teaching only); every later
  round's pairs are ranked by the net's OWN search — expert iteration with
  the search as the improvement operator.
- **The scale probe (20k games + rank pairs): the learned net BEATS the
  teacher — 38-10 vs `beam:cc2@w8d5`** (and 37-11 over the 2k-game rank
  net). Progression: 0-48 → 24-24 (2k) → 38-10 (20k). First decisive win
  of a fully-learned evaluator over the hand eval at matched search on the
  clean stack.
- **Campaign proper (fresh scratch `~/leapfrog-rounds`, schema 3, `--rank`).**
  r0 = 20k CC2 teacher games → anchor **35-13** (BASELINE, ledgered).
- **⭐ ROUND 1 PROMOTED — THE LOOP COMPOUNDS (2026-07-11).** The r0 net's own
  self-play (600 games, pairs ranked by its own search) trained an r1 net that
  **beats its parent 30-18** and holds the CC2 floor **35-13**. First honest
  promotion of a self-improving loop on the clean stack — the whole campaign's
  central question, answered yes. r1 is the new incumbent; round 2 chains from
  it. (Anchor flat vs r0, both 35-13, but head-to-head decisive: self-play
  sharpens net-vs-net, as expected.) Caveat: the driver crashed at the anchor
  step because an infra edit left the tree dirty (the enforced clean-tree
  reproducibility check fired); both duels ran for real and the ledger row was
  reconstructed from the actual results.
- **Infra: datagen work-steals now** (shared atomic game counter, not static
  round-robin) — round-1 datagen ran a 5h straggler (one worker on the long
  competitive games while nine idled); game i still uses seed seeds+i, so
  reproducibility is untouched. Champion ladder (`tp:cc2` upward) begins as the
  loop keeps compounding.

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

# Round-time perf (2026-07-11 audit)

Measured round-1 breakdown (600 games, ~11.4h wall): datagen ~4.9h (123
games/hr), train whitening+epoch0 ~5.6h (cold read + pure-Python checksum over
the 3.8GB 20k-game r0 replay, ×4 passes), warm epochs ~12min, duels ~32min.
A 44-agent adversarial cross-stack audit (all findings verified) ranked the
levers:

- **LANDED (pure-infra, byte-identical results):**
  - *Work-stealing datagen* (shared atomic game counter, not static
    round-robin) — killed a 5h straggler where one worker ground the long
    competitive games while nine idled.
  - *feats-only whitening + verify-once checksum* — the FNV checksum was
    99.6% of `read_shard` (85ms vs 0.3ms); it ran ×4 over the immutable
    replay. Now: whitening reads only feats, checksum verifies once (epoch 0).
    ~15% off each round.
- **QUEUED science-touching levers (need an A/B before adoption), by leverage:**
  1. *Cheaper datagen search* — decouple `datagen --wd` from the gate and run
     e.g. w4d3: ~2.6-3.1× datagen (~3h/round). A/B: does w4d3-ranked pair data
     still gate-beat the anchor / compound? Highest single lever.
  2. *Subsample the replay* (old design kept every-4th-shard = 25%): ~1-1.5h
     marginal after the read-once fix. A/B: same promotion verdict at 25%?
  3. *ANE/CoreML leaf backend revival* (~3-6× datagen) — large effort +
     the complexity we deliberately deleted; only if 1-2 aren't enough.
- **REJECTED as <1% or micro-churn:** per-parent Vec preallocs, SmallVec
  lock_clear, batched-per-generation net eval, im2col split, movegen bit-packs
  — real but sub-1%, not worth the churn (the audit's own verified estimates).

## Is there a pure-infra ORDER-OF-MAGNITUDE datagen speedup? Measured: NO (2026-07-12)

Direct question: can we 10× datagen purely-infra (incl. hidden bugs)? Answer,
by measurement — no, and there is no hidden pathology:
- **Datagen is 100% the net forward** (CC2 datagen 31k games/hr vs net 123 = a
  250× gap that IS the forward; the throughput model closes exactly).
- **No BLAS oversubscription**: `VECLIB_MAXIMUM_THREADS=1` vs default at 10
  workers = 29342 vs 29669 ev/s, identical. The 2.2× per-worker collapse at 10
  workers is genuine 12-core (8P+4E) saturation + memory bandwidth.
- **No redundant evals**: a w8d5 decision scores 2222 leaves — legitimate for
  the config, identical net-path vs cc2-path (board_only sharing saves nothing
  at d5/5-piece-queue). No bug (`tests/eval_count.rs`).
- **All three inference backends converge at a hardware floor of ~3-9k
  evals/s** at datagen's batch~67 (`tests/forward_bench.rs` + a throwaway ORT
  bench): Rust BLAS ~6500, ORT-CPU ~4000, ORT-CoreML/ANE ~4400-5200. **The ANE
  does NOT give the 5-15× the old memory promised** — that was a larger net at
  batch 480-3840; this tiny net at batch 67 is dispatch-overhead-bound and the
  ANE can't beat CPU BLAS. Even at batch 1024 no backend exceeds ~9k (≈1.4× over
  batch 67), so batching buys ~1.5×, not 10×.
- **Verdict:** the ~150-250µs/eval forward is a floor for this net on this
  hardware. Pure-infra realistic ceiling ~1.5-2× (fused conv/layout — risky,
  sub-1.3× per the audit, NOT pursued). The only ~3× lever is **science-touching
  cheaper search (w4d3, ⅓ the evals)** — the queued A/B. An order of magnitude
  would require a fundamentally cheaper evaluator (smaller net = science) or a
  different search paradigm, not an infra fix.

## Campaign-velocity dynamic (2026-07-12): datagen cost GROWS as the net improves

Round 2 datagen (r1 incumbent self-play) runs slower than round 1's (r0): the
stronger net plays longer, more competitive games that reach the 240-ply cap,
so evals/game rises. The gate duels lengthen too (r1's vs-incumbent duel was
22min/48 games). Since there is NO pure-infra 10× (measured above), and per-
round cost climbs with net strength, the **cheaper-search A/B (w4d3, ~3× fewer
evals, applied identically in datagen AND gate so driver≡harness holds) is now
the load-bearing velocity lever**, not an optional nicety — without a per-decision
cost reduction the campaign slows unboundedly as it compounds. Queued to run
once round 2 frees the machine.
