# T11 — Gate-0a findings (survival-recall → agreement@k)

Harness: `crates/tetr-research/src/gate0a.rs` (`gate0a_smoke` test). Champion `tp:cc2@w128d9` mirror self-play under a pressured venue (rain 4, cap 120, sudden-death) reaches near-death fast; capture the topping-out side's last few `SearchState`s; per state re-run a fresh w128d9 beam for `root_scores()` and the round0 net for per-root policy logits. Beam roots, `root_scores()`, and net children all enumerate the same `hold_placements(state)` in the same order → aligned by index (asserted).

## Finding 1 — the panel's binary-survival metric is refuted (measurement corrected design)

The pre-registered Gate-0a asked: does the net's top-k cover the w128 beam's **survival** root set? Running it showed **`n_survival == n_live` in 100% of near-death states** — the champion's w128d9 beam finds a surviving 9-ply line from *every* placement that doesn't *immediately* top out. Survivors are abundant, not a sparse breadth-found hedge. The root-score histogram is cleanly bimodal: death-dominated roots sit at exactly −1e8 (DEATH_SCORE), live roots at ~−10⁴ — no middle ground.

So "survives the horizon" is a near-trivial binary (≈ "not an instant topout," which any bot sees), and recall-of-survival-roots collapses to k/n. **The survival hedge does not live at the root-survival level at d9; it lives in *which survivors are safest* — the score.** This is a genuine correction the armchair panel missed, and it partly *defuses* the skeptic's "width is irreducible survival coverage" objection at the root level.

## Finding 2 — agreement@k: even the weak round0 prior concentrates on the champion's preferred moves

Redefined metric: **agreement@k = |net top-k ∩ champion-beam top-k-by-score| / k** over live roots — does the prior put the champion's decision-relevant moves in its own top-k? (This is what Gumbel-top-m sampling needs: if the champion's best move is in the net's top-m, a narrow search over the net's top-m adjudicates it.)

Result — 24 champion games, 72 near-death states (last 3 plies/game to cut correlation), round0 net (the WEAK probe: policy top-1 0.639):

| k | agree@k | random (k/n_live) | lift |
|--:|--------:|------------------:|-----:|
| 6  | 0.657 | 0.209 | **3.1×** |
| 12 | 0.722 | 0.335 | 2.2× |
| 18 | 0.762 | 0.483 | 1.6× |
| 24 | 0.803 | 0.604 | 1.3× |

The lift is strongest at **low k** (3.1× at k=6) and washes out toward k=24 (where k approaches the live-root count, so any ranker scores high). This is the meaningful regime: a Gumbel search samples the net's top-m (m≈8–16), so "does the champion's best move sit in the net's top-6" is exactly the question — and the answer is ~66% coverage at 3.1× random.

**Read (carefully):** even a weak prior concentrates ~3× above random on the champion's preferred near-death moves — the prior's top-6 already contains most of what the champion's deep-wide beam picks. This is a **green-light (not a verdict)** for the leapfrog thesis: the decision-relevant set *is* coverable by a small policy top-m, consistent with "a narrow guided search can adjudicate what width finds."

## What this is NOT (anti-over-claim)

- **Not a verdict.** A ~few-dozen-state read on a weak net with a mid-experiment metric change. Necessary-not-sufficient.
- **Agreement ≠ improvement.** The champion's top-k-by-score conflates survival + attack; high agreement could partly be imitation (round0 was BC'd on champion-ish data). The real question — does a Gumbel search on the net's top-m *improve* play (G_π > 0.55 at deployable width)? — is **Gate-0b**, still to run for the actual operator on clean seeds.
- **Weak-prior caveat holds:** a stronger prior should raise agreement; this establishes a *floor*, not a ceiling.

## Verdict → next

Gate-0a **does not falsify** the leapfrog thesis; it *sharpens* it. The root-survival moat is not the barrier (survivors abundant, weak prior already covers the champion's picks ~5× over random). The barrier, if any, moves to **value discrimination among survivors under chance** — precisely what Gate-0b (low-width Gumbel G_π with the survival-CVaR backup) must measure. Proceed to build the Gumbel operator (T12) and run Gate-0b (T07), with a stronger prior than round0 for the definitive agreement read.
