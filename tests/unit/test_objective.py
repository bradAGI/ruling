"""The training objective: plain cross-entropy by default, and each calibration term doing only its job."""

import mlx.core as mx
import numpy as np
import pytest

from ruling.train import objective


def batch(probabilities, targets):
    """Log-probabilities, a mask and targets for rows of different widths, padded to the widest."""
    width = max(len(p) for p in probabilities)
    log_probs = mx.array([[float(np.log(v)) for v in p] + [-np.inf] * (width - len(p)) for p in probabilities])
    mask = mx.array([[i < len(p) for i in range(width)] for p in probabilities])
    target = mx.array([t + [0.0] * (width - len(t)) for t in targets], dtype=mx.float32)
    return log_probs, mask, target


def value(*args, **kwargs):
    return float(objective(*args, **kwargs))


def test_with_every_term_off_it_is_the_original_cross_entropy():
    log_probs, mask, targets = batch([[0.7, 0.2, 0.1], [0.4, 0.6]], [[1, 0, 0], [0, 1]])
    expected = -(np.log(0.7) + np.log(0.6)) / 2
    assert value(log_probs, mask, targets) == pytest.approx(expected, rel=1e-5)


def test_label_smoothing_moves_the_target_toward_uniform_over_each_rows_own_options():
    log_probs, mask, targets = batch([[0.5, 0.5], [1 / 3, 1 / 3, 1 / 3]], [[1, 0], [1, 0, 0]])
    # A uniform prediction is the optimum of a fully smoothed target, whatever the row width.
    assert value(log_probs, mask, targets, label_smoothing=1.0) == pytest.approx((np.log(2) + np.log(3)) / 2, rel=1e-5)
    # Smoothing makes a confident correct answer cost more, because it is now partly wrong.
    confident, mask2, target2 = batch([[0.99, 0.01]], [[1, 0]])
    assert value(confident, mask2, target2, label_smoothing=0.1) > value(confident, mask2, target2)


def test_brier_adds_nothing_for_a_perfect_prediction_and_punishes_a_confident_error():
    perfect, mask, targets = batch([[1 - 1e-9, 1e-9]], [[1, 0]])
    assert value(perfect, mask, targets, brier_weight=1.0) == pytest.approx(value(perfect, mask, targets), abs=1e-6)
    wrong, mask, targets = batch([[0.9, 0.1]], [[0, 1]])
    assert value(wrong, mask, targets, brier_weight=1.0) - value(wrong, mask, targets) == pytest.approx(0.81 + 0.81, rel=1e-4)


def test_ranking_penalizes_only_a_wrong_answer_more_confident_than_a_right_one():
    # Row 0 right at 0.6, row 1 wrong at 0.9: an inversion.
    inverted, mask, targets = batch([[0.6, 0.4], [0.9, 0.1]], [[1, 0], [0, 1]])
    # Row 0 right at 0.9, row 1 wrong at 0.6: correctly ordered by more than the margin.
    ordered, mask2, targets2 = batch([[0.9, 0.1], [0.6, 0.4]], [[1, 0], [0, 1]])
    assert value(inverted, mask, targets, ranking_weight=1.0) - value(inverted, mask, targets) == pytest.approx(0.3 + 0.05, rel=1e-4)
    assert value(ordered, mask2, targets2, ranking_weight=1.0) == pytest.approx(value(ordered, mask2, targets2), abs=1e-7)


def test_ranking_is_silent_when_a_batch_is_all_right_or_all_wrong():
    right, mask, targets = batch([[0.9, 0.1], [0.6, 0.4]], [[1, 0], [1, 0]])
    assert value(right, mask, targets, ranking_weight=1.0) == pytest.approx(value(right, mask, targets), abs=1e-7)


def test_the_ranking_gradient_lowers_the_wrong_answers_confidence():
    targets = mx.array([[1.0, 0.0], [0.0, 1.0]])
    mask = mx.array([[True, True], [True, True]])

    def loss(logits):
        return objective(logits - mx.logsumexp(logits, axis=-1, keepdims=True), mask, targets, ranking_weight=1.0)

    logits = mx.array([[0.4, 0.0], [2.2, 0.0]])        # right at ~0.6, wrong at ~0.9
    grad = mx.grad(loss)(logits)
    # Descending the gradient lowers the wrong row's top logit, so its confidence falls.
    assert float(grad[1, 0]) > 0
