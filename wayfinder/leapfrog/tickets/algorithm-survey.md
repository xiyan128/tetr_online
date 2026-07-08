---
id: T02
title: Algorithm-family survey — sample-efficient self-play menu
labels: [wayfinder:research]
status: closed
assignee: fable-lead
blocked-by: []
---

## Question

Which learning-system family fits Tetris versus from first principles? The game: placement-level actions (~34 × hold), stochastic 7-bag beyond the visible queue, seeded garbage holes, *simultaneous* moves approximated by the venue's alternating first-mover, delayed interaction through garbage, ~240-ply death-decisive games under sudden-death rain.

Survey and map onto this game:
- AlphaZero / Expert Iteration (the prior campaign's implicit choice, via beam-root targets);
- **Gumbel AlphaZero/MuZero** — policy improvement guaranteed at tiny simulation budgets (directly attacks the datagen-starvation failure mode);
- stochastic MCTS / chance nodes (bag + garbage as chance); Stochastic MuZero;
- league/population training for non-transitivity (AlphaStar, OpenAI Five) — is a league needed or is prev-gen pool mixing enough at this game's scale?
- sample-efficiency landscape (EfficientZero et al.) — what evals/game-of-data ratios are realistic;
- existing implementations to reuse rather than hand-roll (mctx, open_spiel, Rust MCTS crates) per the check-existing-tools rule.

Weigh against repo evidence (grain of salt, but priced): beam-root policy targets are near-deterministic at width (registered STOP); speculation/bag-belief is load-bearing (~90% spec-ON wins); width is a survival hedge; the champion's moat is brute width — a learned policy prior concentrating that hedge into few candidates is the leapfrog lever.

Output: a design-space memo (linked asset) with one recommended family and rejected alternatives with reasons.

## Resolution

Resolved 2026-07-07 via a six-lens adversarial design panel + red-team + synthesis (full memo: [T02 memo](../assets/T02-algorithm-family-memo.md)).

**Recommended family: Gumbel Expectimax-AlphaZero on the true model** — policy-*guided* search (Gumbel-top-m + Sequential Halving with **completed-Q**, not PUCT/visit-counts), **exact enumerated chance nodes** (bag + garbage) with a **survival-CVaR backup** replacing SPEC_DECAY, **terminal-WDL** value, **completed-Q policy targets**, factored two-board value (compressed opponent). Native Rust search on the existing SearchState/Mind seam; ANE eval; Python trainer. Not MuZero (we own the exact model), not a full league (champion-pinning + Elo tripwire instead).

**Key intellectual results:**
- Completed-Q **dodges fact (d)**: visit-count targets collapse to the starved near-one-hot beam root (Grill 2020); completed-Q reads Q over all placements so π' is non-degenerate at high visit concentration. This is the mechanism the prior beam-root ExIt lacked.
- Survival-CVaR over the **enumerable** bag is the principled replacement for the SPEC_DECAY=0.75 max-over-chance-edges category error, and the only concrete answer to "you can't distill a hedge into a point estimate" (tail-coverage lives in the value, so narrow search need not span the bag).
- The win is **honestly unproven** at matched wall-clock: facts (a)/(b) were measured at w128; the leapfrog operates shallow-with-strong-prior, an unmeasured regime.

**Structural change surfaced (R4):** the deployed net bot's vehicle must consume the policy prior — the value-only `compose()` path is retired for the net bot.

**Graduated to the frontier:** **Gate-0** — a zero-training, ~1-day falsification (survival-recall@k on existing champion near-death rollouts + clean-seed low-width G_π for the *actual* Gumbel operator) that gates all campaign spend. This is cheaper and more decisive than the round-0 retrain; it reframes T07 and becomes a blocker on the design freeze. Pre-registered kill criteria recorded in the memo.
