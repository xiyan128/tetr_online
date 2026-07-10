from __future__ import annotations

import numpy as np
import pytest

from tetrnn.fit_slots import cache_embeddings
from tetrnn.round import require_validity_stack
from tetrnn.targets import (
    BEAM_RANK_BETA,
    legal_mctx_completed_q_target,
    masked_softmax,
    mctx_completed_q_target,
    mctx_completed_q_transform,
    n_eff,
    rank_distillation_target,
)
from tetrnn.train import require_target_contract


def test_matches_official_mctx_policy_fixture() -> None:
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


def test_mctx_mixed_value_formula_and_scale() -> None:
    transformed = mctx_completed_q_transform(
        np.array([1.0, 2.0, np.nan]),
        np.array([2.0, 1.0, 0.0]),
        np.log(np.array([0.2, 0.3, 0.5])),
        0.4,
    )
    # Prior-conditional visited Q = 1.6; v_mix=(.4 + 3*1.6)/4=1.3.
    # Completed [1,2,1.3] -> normalized [0,1,.3], beta=(50+2)*.1=5.2.
    assert np.allclose(transformed, [0.0, 5.2, 1.56])


def test_zero_visits_and_equal_q_preserve_generator_prior() -> None:
    logits = np.array([-2.0, 0.5, 1.0])
    zero_visit = mctx_completed_q_target(np.full(3, np.nan), np.zeros(3), logits, raw_value=-7.0)
    tied_rank = rank_distillation_target(np.full(3, -100_000_000.0), logits)
    prior = masked_softmax(logits)
    assert np.allclose(zero_visit, prior)
    assert np.allclose(tied_rank, prior)


def test_legacy_partial_terminal_sentinel_is_rejected() -> None:
    with pytest.raises(ValueError, match="not evaluator-scale Q values"):
        rank_distillation_target(np.array([-100_000_000.0, -4_000.0, 2_000.0]), np.zeros(3))


def test_all_legacy_terminal_sentinels_preserve_prior() -> None:
    logits = np.array([-2.0, 0.5, 1.0])
    scores = np.array([-100_000_000.0, -100_000_001.0, -100_000_100.0])
    assert np.allclose(rank_distillation_target(scores, logits), masked_softmax(logits))


def test_invalid_is_not_the_same_as_legal_terminal_loss() -> None:
    # Terminal loss is represented on the same finite Q scale, not by the
    # legacy out-of-domain -1e8 search sentinel.
    q = np.array([-10.0, 0.0, 1.0])
    visits = np.ones(3)
    logits = np.zeros(3)
    legal = mctx_completed_q_target(q, visits, logits, raw_value=0.0)
    masked = legal_mctx_completed_q_target(
        q,
        np.array([0.0, 1.0, 1.0]),
        logits,
        raw_value=0.0,
        invalid_actions=np.array([True, False, False]),
    )
    assert legal[0] > 0.0
    assert masked[0] == 0.0
    assert np.isclose(legal.sum(), 1.0) and np.isclose(masked.sum(), 1.0)


def test_invalid_padding_cannot_change_legal_target_resolution() -> None:
    q = np.array([-2.0, 3.0, 0.5])
    visits = np.array([4.0, 2.0, 0.0])
    logits = np.array([0.2, -0.5, 1.0])
    base = legal_mctx_completed_q_target(q, visits, logits, raw_value=0.7)
    padded = legal_mctx_completed_q_target(
        np.append(q, -1e200),
        np.append(visits, 0.0),
        np.append(logits, 1e6),
        raw_value=0.7,
        invalid_actions=np.array([False, False, False, True]),
    )
    assert np.array_equal(padded[:3], base)
    assert padded[3] == 0.0


def test_invalid_action_with_visits_is_rejected() -> None:
    with pytest.raises(ValueError, match="invalid action cannot have a positive visit"):
        legal_mctx_completed_q_target(
            np.array([1.0, 2.0]),
            np.array([1.0, 1.0]),
            np.zeros(2),
            raw_value=0.0,
            invalid_actions=np.array([False, True]),
        )
    with pytest.raises(ValueError, match="invalid action cannot have a positive visit"):
        legal_mctx_completed_q_target(
            np.array([1.0, 2.0]),
            np.array([1.0, 1.0]),
            np.zeros(2),
            raw_value=0.0,
            invalid_actions=np.array([True, True]),
        )


def test_all_invalid_matches_mctx_uniform_edge_case() -> None:
    assert np.allclose(masked_softmax(np.array([-4.0, 8.0]), np.ones(2, dtype=bool)), 0.5)


def test_rank_target_affine_invariance_above_epsilon_and_tie_order() -> None:
    rng = np.random.default_rng(7)
    scores = rng.integers(-40_000, -1_000, size=31).astype(np.float64)
    logits = rng.normal(size=31)
    a = rank_distillation_target(scores, logits)
    b = rank_distillation_target(scores * 7.3 + 123_456.0, logits)
    assert np.allclose(a, b, atol=1e-12)
    tied = rank_distillation_target(np.array([0.0, 1.0, 1.0]), np.zeros(3))
    assert tied[1] == tied[2] > tied[0]


def test_rank_target_pairwise_odds_shift() -> None:
    scores = np.array([-3.0, 1.0, 5.0, 9.0])
    logits = np.array([0.2, -1.3, 0.7, 2.1])
    prior = masked_softmax(logits)
    target = rank_distillation_target(scores, logits)
    qnorm = (scores - scores.min()) / (scores.max() - scores.min())
    for i, j in [(0, 1), (1, 3), (2, 0)]:
        shift = np.log(target[i] / target[j]) - np.log(prior[i] / prior[j])
        assert np.isclose(shift, BEAM_RANK_BETA * (qnorm[i] - qnorm[j]))


def test_rank_tilt_never_lowers_expected_score() -> None:
    rng = np.random.default_rng(11)
    for _ in range(500):
        n = int(rng.integers(2, 80))
        scores = rng.normal(size=n)
        logits = rng.normal(size=n)
        prior = masked_softmax(logits)
        target = rank_distillation_target(scores, logits, beta=float(rng.uniform(0, 12)))
        assert float(target @ scores) >= float(prior @ scores) - 1e-12


def test_reference_scale_adversarial_envelope() -> None:
    scores = np.zeros(30)
    scores[4] = 1.0
    target = rank_distillation_target(scores, np.zeros(30))
    assert 0.83 < target[4] < 0.85
    assert n_eff(target) > 2.5


def test_epsilon_prevents_microscopic_gap_from_becoming_decisive() -> None:
    target = rank_distillation_target(np.array([0.0, 1e-12]), np.zeros(2))
    assert np.allclose(target, 0.5, atol=2e-4)


def test_target_is_deterministic_and_normalized() -> None:
    q = np.array([2.0, -1.0, 8.0, 3.0])
    visits = np.array([2.0, 0.0, 1.0, 0.0])
    logits = np.array([0.1, 0.2, -0.4, 1.3])
    first = mctx_completed_q_target(q, visits, logits, raw_value=0.7)
    second = mctx_completed_q_target(q, visits, logits, raw_value=0.7)
    assert first.tobytes() == second.tobytes()
    assert np.isfinite(first).all() and np.isclose(first.sum(), 1.0)


@pytest.mark.parametrize(
    ("scores", "logits"),
    [
        (np.array([]), np.array([])),
        (np.zeros((2, 2)), np.zeros(4)),
        (np.array([0.0, np.nan]), np.zeros(2)),
        (np.zeros(2), np.zeros(3)),
    ],
)
def test_rank_target_rejects_malformed_inputs(scores: np.ndarray, logits: np.ndarray) -> None:
    with pytest.raises(ValueError):
        rank_distillation_target(scores, logits)


def test_training_fails_closed_on_missing_or_live_generator_logits() -> None:
    with pytest.raises(RuntimeError, match="policy training is paused"):
        require_target_contract([])
    with pytest.raises(RuntimeError, match="no finite fixed point"):
        require_target_contract(["live"])
    with pytest.raises(RuntimeError, match="legacy corpora are audit-only"):
        require_target_contract(["--legacy-uniform-complete"])


def test_round_driver_is_paused_before_more_invalid_datagen() -> None:
    with pytest.raises(RuntimeError, match="campaign paused by the validity reset"):
        require_validity_stack()


def test_programmatic_legacy_slot_fitting_is_paused() -> None:
    with pytest.raises(RuntimeError, match="legacy slot targets are audit-only"):
        cache_embeddings(object(), [])
