---
id: T11
title: Gate-0a — survival-recall@k on champion near-death rollouts
labels: [wayfinder:task]
status: closed
assignee: fable-lead
blocked-by: []
---

## Question

The cheapest, most decisive falsification of the whole leapfrog thesis (from the T02 panel). Zero training, ~1 day. Uses the existing `round0` net's policy head + champion (`tp:cc2@w128d9`) rollouts — does NOT wait on the round-0 retrain.

Thesis under test: a learned policy prior can *concentrate the survival hedge* into few candidates, making brute width unnecessary. If the prior's top-k cannot even cover the survival branches that width finds, the hedge is an irreducible breadth/coverage property (an information-theoretic moat) and the leapfrog is dead before any campaign.

Procedure:
1. Generate champion self-play rollouts; extract **near-death positions** (boards a few plies from a topout under rain — where the survival hedge actually bites).
2. At each, take the champion's w128 beam's set of survival / non-topout root placements (the "breadth-found hedge").
3. Take the `round0` net policy head's top-k (k≈12–24) at the same position.
4. Measure **recall@k**: does the prior's top-k cover the w128 survival slice?

**Kill criterion (pre-registered):** if recall@24 of the learned prior < the w128 beam's own top-24 survival slice → STOP the leapfrog; width is irreducible here. If it covers → the door is open, proceed to Gate-0b.

Output: the recall@k curve + verdict, and whether the round0 policy head is strong enough to be the probe or a better prior is needed first.

## Execution spec (API-grounded recon, 2026-07-07)

Confirmed feasible on existing APIs — no new engine surface needed. Build one instrument in `tetr-research`:

1. **Near-death corpus.** Run champion (`tp:cc2@w128d9`) self-play under the venue (heavy rain reaches near-death faster). For each game ending in topout, capture the `SearchState` at the last K≈5–10 plies before the topout ply. Target ~a few hundred near-death positions. (Reuse the versus loop / rollout path; positions are `SearchState`s.)
2. **Beam survival-root set.** At each position, run `BeamPlanner` at w128d9 and read `root_scores()` (`beam.rs:194`) → `(Placement, i32)` per root. Classify a root as **survival** iff its backed-up score is not `DEATH_SCORE`-dominated (all-dead lines back up to `super::DEATH_SCORE`; pick the threshold from the score histogram — there is a clean gap). This is the "breadth-found hedge."
3. **Net policy top-k.** At the same position, get the `round0` net's per-root-placement policy logits (the `policy:` arm / `PolicyMind` already ranks root children by policy head) → top-k, k∈{6,12,18,24}.
4. **Metric.** `recall@k = |survival_roots ∩ policy_topk| / |survival_roots|`, aligned by placement identity, averaged over the corpus. Also report the baseline (uniform top-k) and the count/fraction of survival roots per position (if ~all roots survive, the position wasn't near-death — filter).

**Verdict:** recall@24 ≥ the w128 beam's own top-24 survival slice → door open (proceed to Gate-0b). Below → width is irreducible, STOP the leapfrog.

Caveat to pre-register: the round0 net is weak (policy top-1 0.639); if it fails recall, re-run with a stronger prior (e.g. a BC net from champion rollouts) before concluding the moat holds — a weak prior failing is not the same as the hedge being unlearnable.

## Resolution

Built + ran 2026-07-07 (harness `crates/tetr-research/src/gate0a.rs`, `gate0a_smoke` test; findings: [T11 asset](../assets/T11-gate0a-findings.md)). **Gate-0a did NOT falsify the leapfrog thesis — it corrected the metric and returned a positive precondition signal.**

**Two findings:**
1. **The pre-registered binary-survival metric is uninformative (measurement corrected design).** In 72/72 near-death states, `n_survival == n_live` — the champion's w128d9 beam finds a surviving 9-ply line from *every* placement that doesn't immediately top out. Survivors are abundant, not a sparse breadth-found hedge; the root-score histogram is cleanly bimodal (−1e8 death vs ~−10⁴ live). The kill criterion ("recall@24 < beam's top-24 survival slice") is moot — there is no selective survival slice at d9. The hedge lives in *which survivors are safest* (the score), not binary survival.
2. **Redefined metric — agreement@k (net top-k ∩ champion-beam top-k-by-score)/k — is positive even for a WEAK prior.** round0 (policy top-1 0.639): agree@6 = 0.657 vs random 0.209 = **3.1× lift**, strongest at low k (the Gumbel-top-m regime). The champion's preferred near-death moves already sit in the weak net's top-6 at ~66%, ~3× over chance.

**Verdict:** the root-survival moat is not the barrier (survivors abundant; even a weak prior covers the champion's picks 3× over random). The barrier, if any, relocates to **value discrimination among survivors under chance** — exactly what Gate-0b (low-width Gumbel G_π + survival-CVaR backup) measures. Proceed. Caveats pre-registered: agreement is partly imitation (round0 was BC'd on champion-ish data) and agreement ≠ improvement; a stronger prior (T05) should raise the number and is the definitive read; Gate-0b (T07) tests whether the covered set actually yields search improvement.
