"""Reference completed-Q targets and the beam rank-distillation fallback.

These are deliberately separate contracts.

``mctx_completed_q_target`` is a NumPy port of DeepMind Mctx's Gumbel MuZero
target construction: complete unvisited legal actions with the mixed value,
min-max rescale the completed values, add the transformed values to the
*frozen generator* logits, and mask only genuinely invalid actions.

``rank_distillation_target`` is the narrower contract supported by legacy
Tetr shards.  Those shards contain a backed-up score for every *stored* root,
but no generator logits, visit counts, raw value, invalid-action mask, or
discarded roots.  A caller may use this helper only when it can independently
prove that the stored roots are the complete legal action set and can supply
the frozen generator logits (an explicitly uniform teacher uses zero logits).
It is rank distillation, not theorem-backed Gumbel policy improvement.

The old implementation silently treated missing logits as uniform, removed
legal terminal-loss moves as though they were invalid, and used ``c=12``.
That constant makes the module's own 30-way isolated-best example essentially
one-hot (top mass 0.99982, effective support 1.002).  Until beam search records
meaningful MCTS-like counts, the fallback uses beta=5: Mctx's default
``maxvisit_init * value_scale`` with no invented visits.
"""

from __future__ import annotations

import numpy as np

MCTX_VALUE_SCALE = 0.1
MCTX_MAXVISIT_INIT = 50.0
MCTX_EPSILON = 1e-8

# Beam root_best values are max-backed-up heuristic ranks, not MCTS Q means,
# and legacy shards have no meaningful visit counts.  Five is the Mctx default
# transform scale before any visits: 50 * 0.1.
BEAM_RANK_BETA = MCTX_MAXVISIT_INIT * MCTX_VALUE_SCALE

# Legacy beam shards encode a legal immediate terminal loss with this
# out-of-domain sentinel.  It is not a Q on the evaluator's ordinary scale.
LEGACY_TERMINAL_THRESHOLD = -1_000_000.0


def _vector(name: str, value: np.ndarray, dtype: np.dtype) -> np.ndarray:
    out = np.asarray(value, dtype=dtype)
    if out.ndim != 1 or out.size == 0:
        raise ValueError(f"{name} must be a non-empty rank-1 array")
    return out


def _invalid_mask(invalid_actions: np.ndarray | None, n: int) -> np.ndarray:
    if invalid_actions is None:
        return np.zeros(n, dtype=np.bool_)
    invalid = _vector("invalid_actions", invalid_actions, np.dtype(np.bool_))
    if invalid.shape != (n,):
        raise ValueError(f"invalid_actions has shape {invalid.shape}, expected {(n,)}")
    return invalid


def masked_softmax(logits: np.ndarray, invalid_actions: np.ndarray | None = None) -> np.ndarray:
    """Stable softmax with Mctx's invalid-action behavior.

    At least one valid action gives invalid actions exactly zero probability.
    If every action is invalid, Mctx replaces every logit with the same finite
    minimum and therefore returns uniform; this port preserves that edge case.
    """
    z = _vector("logits", logits, np.dtype(np.float64))
    if not np.isfinite(z).all():
        raise ValueError("logits must be finite")
    invalid = _invalid_mask(invalid_actions, z.size)
    if invalid.all():
        return np.full(z.size, 1.0 / z.size, dtype=np.float64)
    valid = ~invalid
    shifted = np.full(z.size, -np.inf, dtype=np.float64)
    shifted[valid] = z[valid] - z[valid].max()
    weights = np.exp(shifted)
    return weights / weights.sum()


def mctx_completed_q_transform(
    qvalues: np.ndarray,
    visit_counts: np.ndarray,
    prior_logits: np.ndarray,
    raw_value: float,
    *,
    invalid_actions: np.ndarray | None = None,
    value_scale: float = MCTX_VALUE_SCALE,
    maxvisit_init: float = MCTX_MAXVISIT_INIT,
    rescale_values: bool = True,
    use_mixed_value: bool = True,
    epsilon: float = MCTX_EPSILON,
) -> np.ndarray:
    """Port of Mctx ``qtransform_completed_by_mix_value``.

    ``visit_counts > 0`` means an action has an observed Q.  Unvisited legal
    actions are completed with Mctx's prior-weighted mixed-value estimator (or
    ``raw_value`` when ``use_mixed_value`` is false).  Invalidity is orthogonal
    to visitation: a legal terminal-loss action must have a finite Q and must
    not be marked invalid. This function preserves Mctx's fixed-vocabulary
    normalization, in which unvisited masked slots receive the completion value;
    Tetr's variable legal-action contract uses
    :func:`legal_mctx_completed_q_target` to remove padding first.
    """
    q = _vector("qvalues", qvalues, np.dtype(np.float64))
    visits = _vector("visit_counts", visit_counts, np.dtype(np.float64))
    logits = _vector("prior_logits", prior_logits, np.dtype(np.float64))
    if visits.shape != q.shape or logits.shape != q.shape:
        raise ValueError("qvalues, visit_counts, and prior_logits must have equal shape")
    if not np.isfinite(visits).all() or (visits < 0).any():
        raise ValueError("visit_counts must be finite and non-negative")
    if not np.isfinite(logits).all() or not np.isfinite(raw_value):
        raise ValueError("prior_logits and raw_value must be finite")
    if not np.isfinite([value_scale, maxvisit_init, epsilon]).all():
        raise ValueError("scales and epsilon must be finite")
    if value_scale < 0 or maxvisit_init < 0 or epsilon <= 0:
        raise ValueError("scales must be non-negative and epsilon must be positive")
    invalid = _invalid_mask(invalid_actions, q.size)
    visited = visits > 0
    if np.any(invalid & visited):
        raise ValueError("an invalid action cannot have a positive visit count")
    if not np.isfinite(q[visited]).all():
        raise ValueError("visited qvalues must be finite")

    prior = masked_softmax(logits, invalid)
    if use_mixed_value:
        # Mctx clamps probabilities to tiny so a visited action with a masked
        # or underflowed prior cannot create 0/0 in the conditional average.
        safe_prior = np.maximum(np.finfo(np.float64).tiny, prior)
        if visited.any():
            prob_visited = safe_prior[visited].sum()
            weighted_q = float(np.sum(safe_prior[visited] * q[visited] / prob_visited))
        else:
            weighted_q = 0.0
        total_visits = float(visits.sum())
        completion = (float(raw_value) + total_visits * weighted_q) / (total_visits + 1.0)
    else:
        completion = float(raw_value)

    completed = np.where(visited, q, completion)
    if rescale_values:
        lo = float(completed.min())
        span = float(completed.max() - lo)
        transformed = (completed - lo) / max(span, epsilon)
    else:
        transformed = completed
    beta = (maxvisit_init + float(visits.max())) * value_scale
    return beta * transformed


def mctx_completed_q_target(
    qvalues: np.ndarray,
    visit_counts: np.ndarray,
    prior_logits: np.ndarray,
    raw_value: float,
    *,
    invalid_actions: np.ndarray | None = None,
    value_scale: float = MCTX_VALUE_SCALE,
    maxvisit_init: float = MCTX_MAXVISIT_INIT,
    rescale_values: bool = True,
    use_mixed_value: bool = True,
    epsilon: float = MCTX_EPSILON,
) -> np.ndarray:
    """Mctx-compatible policy target ``softmax(prior + completed-Q)``."""
    logits = _vector("prior_logits", prior_logits, np.dtype(np.float64))
    transformed = mctx_completed_q_transform(
        qvalues,
        visit_counts,
        logits,
        raw_value,
        invalid_actions=invalid_actions,
        value_scale=value_scale,
        maxvisit_init=maxvisit_init,
        rescale_values=rescale_values,
        use_mixed_value=use_mixed_value,
        epsilon=epsilon,
    )
    return masked_softmax(logits + transformed, invalid_actions)


def legal_mctx_completed_q_target(
    qvalues: np.ndarray,
    visit_counts: np.ndarray,
    prior_logits: np.ndarray,
    raw_value: float,
    *,
    invalid_actions: np.ndarray | None = None,
    value_scale: float = MCTX_VALUE_SCALE,
    maxvisit_init: float = MCTX_MAXVISIT_INIT,
    rescale_values: bool = True,
    use_mixed_value: bool = True,
    epsilon: float = MCTX_EPSILON,
) -> np.ndarray:
    """Completed-Q target whose legal distribution is invariant to padding.

    Mctx operates on a fixed action vocabulary, so its reference transform
    completes masked slots before rescaling. Tetr's schema stores an explicit
    variable legal-action set. For that contract, invalid padding must not
    change the resolution or odds among legal actions: remove invalid entries,
    apply the exact reference transform to the legal set, then scatter zeros
    back. An all-invalid row retains Mctx's uniform defensive edge case.
    """
    q = _vector("qvalues", qvalues, np.dtype(np.float64))
    visits = _vector("visit_counts", visit_counts, np.dtype(np.float64))
    logits = _vector("prior_logits", prior_logits, np.dtype(np.float64))
    if visits.shape != q.shape or logits.shape != q.shape:
        raise ValueError("qvalues, visit_counts, and prior_logits must have equal shape")
    invalid = _invalid_mask(invalid_actions, q.size)
    if np.any((visits > 0) & invalid):
        raise ValueError("an invalid action cannot have a positive visit count")
    if invalid.all():
        return np.full(q.size, 1.0 / q.size, dtype=np.float64)
    legal = ~invalid
    legal_target = mctx_completed_q_target(
        q[legal],
        visits[legal],
        logits[legal],
        raw_value,
        value_scale=value_scale,
        maxvisit_init=maxvisit_init,
        rescale_values=rescale_values,
        use_mixed_value=use_mixed_value,
        epsilon=epsilon,
    )
    target = np.zeros(q.size, dtype=np.float64)
    target[legal] = legal_target
    return target


def rank_distillation_target(
    scores: np.ndarray,
    generator_logits: np.ndarray,
    *,
    beta: float = BEAM_RANK_BETA,
    epsilon: float = MCTX_EPSILON,
) -> np.ndarray:
    """Search-rank target for a proven-complete legal beam-root set.

    Every score must be a meaningful Q on one finite scale. A group that mixes
    the legacy ``-1e8`` terminal sentinel with ordinary evaluator scores is
    rejected: global min-max would otherwise erase the ranking among survivors.
    The caller must supply logits from the frozen data-generating policy. There
    is intentionally no ``None => uniform`` shortcut: a uniform teacher must
    be explicit as ``np.zeros_like(scores)`` and learned top-m legacy shards
    must not pass through this API.
    """
    q = _vector("scores", scores, np.dtype(np.float64))
    logits = _vector("generator_logits", generator_logits, np.dtype(np.float64))
    if logits.shape != q.shape:
        raise ValueError("scores and generator_logits must have equal shape")
    if not np.isfinite(q).all() or not np.isfinite(logits).all():
        raise ValueError("scores and generator_logits must be finite")
    if not np.isfinite([beta, epsilon]).all():
        raise ValueError("beta and epsilon must be finite")
    if beta < 0 or epsilon <= 0:
        raise ValueError("beta must be non-negative and epsilon must be positive")
    terminal = q <= LEGACY_TERMINAL_THRESHOLD
    if terminal.any() and not terminal.all():
        raise ValueError(
            "partial legacy terminal sentinels are not evaluator-scale Q values; "
            "store a same-scale terminal Q or omit this decision's policy loss"
        )
    if terminal.all():
        # Sentinel magnitudes encode no preference among forced terminal moves.
        return masked_softmax(logits)
    lo = float(q.min())
    span = float(q.max() - lo)
    qnorm = (q - lo) / max(span, epsilon)
    return masked_softmax(logits + beta * qnorm)


def n_eff(pi: np.ndarray) -> float:
    """Effective support size ``exp(H(pi))``."""
    p = np.asarray(pi, dtype=np.float64)
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))


def _tests() -> None:
    """Fast executable smoke checks; pytest carries the full contract."""
    # Official Mctx policies-test fixture (invalid actions are 0 and 3).
    target = mctx_completed_q_target(
        np.array([20.0, 3.0, -1.0, 10.0]),
        np.array([0.0, 9.0, 8.0, 0.0]),
        np.array([0.0, -1.0, 2.0, 3.0]),
        -5.0,
        invalid_actions=np.array([True, False, False, True]),
        maxvisit_init=60.0,
        value_scale=0.05,
    )
    assert np.allclose(target, [0.0, 0.60186886, 0.39813114, 0.0], atol=1e-7)

    scores = np.zeros(30)
    scores[7] = 1.0
    pi = rank_distillation_target(scores, np.zeros_like(scores))
    assert pi[7] == pi.max() and pi[7] < 0.85
    assert n_eff(pi) > 2.5

    tied = np.full(6, -100_000_000.0)
    prior_logits = np.arange(6, dtype=np.float64)
    assert np.allclose(rank_distillation_target(tied, prior_logits), masked_softmax(prior_logits))
    print("targets.py: reference parity and rank-target smoke checks pass")


if __name__ == "__main__":
    _tests()
