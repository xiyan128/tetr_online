# Lab note — adaptive (decision-criticality) budget allocation: a rigorous null

*2026-06-20. Built + ablated in a discarded worktree (branch `experiment/adaptive-allocation`,
never committed — so the code is gone from git). This note is the durable record; the design
below is enough to rebuild, and the full implementation lives only in the session transcript.*

## What was tried

Spend search compute where it changes the decision, not uniformly. The bot replans every piece
under a fixed node budget, but decision difficulty is non-uniform (E0: 60–77 % of ply-1 decisions
settle by ~d6, 10–18 % flip past d9). So: **stop a move's search early once its ply-1 decision is
stable, bank the unspent nodes, and spend the bank on still-undecided (hard) moves — at iso-budget.**

Implementation was clean and verified (kept here only as a design record): an eval-agnostic
`BudgetAllocator` (cross-move node bank + per-move stop monitor) consulted by `SearchPolicy`'s
single stop gate; the fixed baseline byte-identical (untouched arithmetic path; the full 283-test
suite stayed green); slice-invariant (node-checkpoint cadence, sliced ≡ blocking); F3-safe
(node-only — no width knob, so it can never drop below the survival-width floor). Gate-green.

## Result: NULL

| race (rain pair-GSPRT, iso-budget) | verdict |
|---|---|
| naive adaptive (`k3`, argmax-stop, reinvest→d20) vs fixed | **H0, loses −9.6** |
| efficiency-only (stop, no reinvest) vs fixed | **H0, loses −15** — the early-stop *alone* hurts |
| adaptive vs efficiency-only | ≈tie (+0.8) — **reinvesting into depth adds nothing** |
| best-tuned (`k5`+margin, reinvest capped at d12) vs fixed, **SPRT seeds** | H1, wins 93-69 |
| …same, **held-out CONFIRM seeds** | **H0, loses 91-96 — did not replicate** |

The tuned "win" was a seed artifact; the disjoint-region re-read reversed it (the held-out
confirmation discipline caught a false positive).

## Why — two failures, both eval-independent

1. **The reallocation target (depth) is saturated.** Depth pays only to ~d12 (E1); reinvesting
   past it is worthless-to-harmful (`adaptive ≈ efficiency-only`). Saturation is *game-structural*
   (the ~6-ply preview + 7-bag uncertainty past it), so a better leaf eval shifts it modestly at
   most — it does not open the headroom adaptive-depth needs.
2. **The cheap criticality signal is unreliable.** "Argmax stable for `k` generations" does not
   mean settled — moves re-flip — so the early-stop *costs* decisions the full-budget bot gets
   right (efficiency-only is the worst, −15).

## Bearing on the learned-eval (NNUE) blueprint

- The blueprint's compute-allocation unlock (#4: `gap = V(best) − V(2nd)` → reallocate budget) **is
  exactly this experiment.** The null shows it is gated by **depth saturation, not eval
  calibration** — a learned V does not deliver it as stated. (It might improve the *signal*;
  it cannot un-saturate the *target*.)
- The blueprint's "Bellman unlock" (#1) is moot in this codebase: `cc2.rs` already builds Value and
  Reward in the same `SCALE=256` space and the beam already composes `Σ reward + Value(leaf)`.
- The one composable bet the null leaves open: **V-guided *width* allocation** (width is the
  un-saturated survival lever) — but it is F3-dangerous (going narrow on easy moves risks the
  survival floor) and needs a learned criticality signal. Not a cheap win.

## Verdict on the lever

The cheap, ML-free **search-allocation** lever is closed. With E3/E11 (speculation not the ceiling)
and E8 (value net no compelling headroom), the in-paradigm levers are exhausted. The remaining
frontier is all a deliberate, expensive commitment the evidence does not yet green-light: a learned
eval (E8 leans STOP), adaptive width (F3-risky), or two-agent / timing search (research-only,
telemetry-blocked). The honest state: the bot is near its ceiling for cheap moves.
