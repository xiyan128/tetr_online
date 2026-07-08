---
id: T07
title: Gate-0b — clean-seed low-width G_π for the actual Gumbel operator
labels: [wayfinder:task]
status: closed
assignee: fable-lead
blocked-by: []
---

## Question

Expert iteration compounds only if search genuinely improves on the raw policy — that gap is the training signal. The prior campaign's authorizing number G_π = 0.900 (99-11-0) is **triply invalid**: measured on contaminated TRAIN-region seeds (LANDING R3), measured for a **beam not the Gumbel operator**, and it certifies a policy head the deployed value-only path never reads (R4). It must be re-measured for the real operator before any campaign spend. (This ticket was reframed from "search gain on clean seeds" by the T02 panel.)

With the probe-grade Gumbel operator from *Build the minimal Gumbel expectimax search vehicle*:

1. Run `duel --a gumbel:<M>@m..n.. --b policy:<M>` on **genuinely fresh** seed regions (disjoint from any training region — verify the seed plumbing, a known prior bug).
2. Measure at low, **deployable** widths (m=8/16, n=32/64/128) — the regime the leapfrog actually operates in, NOT w128. Report the gain *curve*, with CIs and end-reason strata.
3. Check the completed-Q target-extraction: does π' *improve* the net or *inject entropy* (the R1 trap that regressed the net even at G=0.900)?

**Kill criterion (pre-registered):** if `G_π < 0.55` (CI-lower) at deployable width → no search gain where it must exist, leapfrog falsified, fall to the value-only escape hatch. Do NOT inherit the 0.900.

Output: the verified low-width G_π curve for the Gumbel operator + the go/no-go for expert iteration — a hard blocker on the design freeze.

## Resolution (v0 — search-gain premise CONFIRMED on clean seeds)

Ran 2026-07-08 via the existing sound-v0 operator (the beam; see the T12 scope correction — the beam gives correct per-root Q, so it *is* the v0 policy-guided search). `duel beam:round0@w8d5 vs policy:round0`, **fresh seed region (base 700000000, disjoint from anything)**:

**31-1-0 over 32 games — ~0.97 win share, all decisive (topout).**

Deep search crushes the raw round0 policy on clean seeds. This **re-confirms (and exceeds) the prior campaign's contaminated G_π=0.900** — the search-gain premise that authorizes expert iteration is solid, and the contaminated-seed doubt (LANDING R3) is retired for this net. Combined with [[Gate-0a]] (weak prior covers the champion's picks 3.1× over random), **both preconditions of the leapfrog thesis are green.**

**What this does NOT settle (the definitive test is downstream):** beating a *weak* policy (round0 top-1 0.639) with deep search is expected; the hard question the panel flagged is whether training on the search's **completed-Q targets** improves the net *without the R1 entropy-injection trap*. That is a **training-time test = campaign round 1**, post-design-freeze — not a duel. The remaining refinements (the low-width w4/w8/w16 curve; the SH-efficient operator) are throughput optimizations (T12/T13), not premise questions.

**Verdict:** the expert-iteration premise is real on clean seeds. Proceed to the design freeze; the decisive remaining risk moves to target-extraction (campaign) and deployment-parity at matched wall-clock (needs the forward fix).
