"""Completed-Q policy targets (leapfrog T15) — the fact-(d) dodge.

The beam backs up a per-root Q (integer score, arbitrary scale). Its *argmax*
is near-deterministic (the prior campaign's registered STOP: visit/argmax
targets collapse to one-hot and the policy loss starves). The Gumbel-MuZero
improved policy reads Q over ALL roots instead:

    pi'(a) = softmax( logit_theta(a) + c * qnorm(a) )     over live roots
    qnorm  = (Q - min_live Q) / (max_live Q - min_live Q)  in [0, 1]
    dead roots (death-dominated backup) get -inf logit -> pi' = 0

Adaptation from mctx: the beam expands every root at full width (uniform
"visits"), so mctx's visit-scaled sigma(q) = (c_visit + max_b N(b)) * c_scale * q
collapses to one fixed constant c. Two properties carried by construction:

  * SCALE-FREE: qnorm is invariant to affine rescaling of the beam score, so c
    transfers across rounds/nets — no tau recalibration (the C6 trap).
  * NON-DEGENERATE: pi' is a smooth function of Q gaps even when argmax is
    near-deterministic; entropy is controlled by c, not by how sharply the
    search happened to concentrate.

Round-0 BC uses logits=None (untrained prior ~ uniform): pi' = softmax(c*qnorm).
Round-1+ recomputes logits from the CURRENT net at training time (the shards
store the served children obs, so this is free reanalyze).
"""

from __future__ import annotations

import numpy as np

# A root whose backup is below this is death-dominated (DEATH_SCORE = -1e8;
# real scores are O(1e4) — the corpus histogram is cleanly bimodal).
DEATH_THRESH = -1_000_000

# The sigma scale: how strongly a full-width beam's Q ranking bends the prior.
# Fixed by design (scale-free qnorm makes it transferable across rounds/nets).
# Calibrated ONCE (2026-07-08, 2347 decisions of the CC2 w8d5 corpus): c-sweep
# gave median N_eff 41→2.0 over c 4→96; c=12 is the softest value whose median
# (5.63, p10 2.25, p90 15.68) sits in the pre-registered sharpness band
# [2.5, 6]. Flat-Q decisions stay soft (no signal → no invented sharpness);
# top-tied decisions floor at N_eff≈2 even as c→∞ (min-max keeps real ties).
C_SCALE = 12.0


def completed_q_target(
    scores: np.ndarray,
    logits: np.ndarray | None = None,
    c: float = C_SCALE,
) -> np.ndarray:
    """pi' over one decision's children from the beam's per-root scores.

    scores: [n] i32/f — the beam's backed-up root scores (DEATH-coded).
    logits: [n] f32 or None — the current net's policy logits (None = uniform).
    Returns [n] f64 probabilities summing to 1 over live roots (0 on dead).
    All-dead decisions return the uniform distribution (a forced loss carries
    no ranking signal; the value head owns it via z).
    """
    scores = np.asarray(scores, dtype=np.float64)
    n = scores.shape[0]
    live = scores > DEATH_THRESH
    if not live.any():
        return np.full(n, 1.0 / n)

    lo, hi = scores[live].min(), scores[live].max()
    qnorm = np.zeros(n)
    if hi > lo:
        qnorm[live] = (scores[live] - lo) / (hi - lo)
    else:
        qnorm[live] = 0.5  # all live roots tied: prior (or uniform) decides

    z = c * qnorm
    if logits is not None:
        z = z + np.asarray(logits, dtype=np.float64)
    z[~live] = -np.inf
    z -= z[live].max()
    e = np.exp(z)
    return e / e.sum()


def n_eff(pi: np.ndarray) -> float:
    """Effective support size exp(H(pi)) — the target-sharpness read."""
    p = pi[pi > 0]
    return float(np.exp(-(p * np.log(p)).sum()))


# ---------------------------------------------------------------------------
# Self-checks: the two load-bearing properties + edge cases.

def _tests() -> None:
    rng = np.random.default_rng(0)

    # 1. SCALE-FREE: affine rescaling of scores leaves pi' identical.
    s = rng.integers(-40_000, -1_000, size=30).astype(np.float64)
    a = completed_q_target(s)
    b = completed_q_target(s * 7.3 + 123_456)  # affine, stays above DEATH_THRESH
    assert np.allclose(a, b, atol=1e-12), "affine rescale changed the target"

    # 2. NON-DEGENERATE at near-deterministic argmax: one root clearly best
    #    still leaves real mass elsewhere (this is the fact-(d) dodge).
    s = np.full(30, -20_000.0)
    s[7] = -1_500.0  # dominant
    pi = completed_q_target(s)
    assert pi[7] == pi.max()
    assert n_eff(pi) > 2.0, f"target collapsed: N_eff={n_eff(pi):.2f}"
    assert pi[7] < 0.9, "argmax hoarded the mass"

    # 3. Dead masking: death-coded roots get exactly zero.
    s = np.array([-5_000.0, -100_000_000.0, -3_000.0])
    pi = completed_q_target(s)
    assert pi[1] == 0.0 and np.isclose(pi.sum(), 1.0)

    # 4. All-dead -> uniform (forced loss, no ranking signal).
    s = np.full(5, -100_000_000.0)
    assert np.allclose(completed_q_target(s), 0.2)

    # 5. All live tied -> uniform (no Q signal, prior decides; None = uniform).
    s = np.full(6, -9_000.0)
    assert np.allclose(completed_q_target(s), 1 / 6)

    # 6. Logits shift the target: a strong prior on a mediocre root moves mass.
    s = np.linspace(-30_000, -2_000, 10)
    logit = np.zeros(10)
    logit[0] = 5.0  # prior loves the worst live root
    pi0 = completed_q_target(s)
    pi1 = completed_q_target(s, logits=logit)
    assert pi1[0] > pi0[0] * 10, "prior logit had no effect"

    # 7. Monotone in Q among live roots (equal logits).
    s = rng.integers(-40_000, -1_000, size=20).astype(np.float64)
    pi = completed_q_target(s)
    order_q = np.argsort(s)
    order_p = np.argsort(pi)
    assert (order_q == order_p).all(), "target not monotone in Q"

    print("targets.py: all 7 self-checks pass")


if __name__ == "__main__":
    _tests()

    # N_eff read over a real corpus, if one is given.
    import sys

    if len(sys.argv) > 1:
        from .shards import read_shard, shard_paths

        effs, top1 = [], []
        for p in shard_paths(sys.argv[1]):
            sh = read_shard(p)
            for k in range(sh.n_decisions):
                ch = sh.children_of(k)
                pi = completed_q_target(sh.child_score[ch])
                effs.append(n_eff(pi))
                top1.append(pi.max())
        effs = np.array(effs)
        top1 = np.array(top1)
        print(
            f"corpus N_eff: median={np.median(effs):.2f} "
            f"p10={np.percentile(effs, 10):.2f} p90={np.percentile(effs, 90):.2f} | "
            f"target top-1 mass: median={np.median(top1):.2f}"
        )
