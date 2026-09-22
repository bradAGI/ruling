import math

import numpy as np
import pytest

from ruling.calibration import (
    Calibration,
    choice_confidence,
    score_confidence,
    expected_calibration_error,
    fit_temperature,
    softmax,
)


def test_softmax_temperature_flattens_and_sharpens():
    logits = np.array([2.0, 0.0, -1.0])
    assert softmax(logits, 10.0).max() < softmax(logits, 1.0).max() < softmax(logits, 0.1).max()
    assert math.isclose(softmax(logits, 3.0).sum(), 1.0)


@pytest.mark.parametrize("n", [2, 3, 7])
def test_choice_confidence_runs_from_uniform_to_certain(n):
    assert choice_confidence(np.full(n, 1.0 / n)) == pytest.approx(0.0)
    assert choice_confidence(np.eye(n)[0]) == pytest.approx(1.0)
    rng = np.random.default_rng(n)
    for _ in range(50):
        assert 0.0 <= choice_confidence(rng.dirichlet(np.ones(n))) <= 1.0


def test_choice_confidence_matches_typesafes_published_example():
    # three options with a peak of 0.8: (0.8 - 1/3) / (1 - 1/3)
    assert choice_confidence(np.array([0.8, 0.1, 0.1])) == pytest.approx(0.7)
    assert choice_confidence(np.array([1.0])) == 1.0


def test_score_confidence_measures_concentration_around_the_modal_level():
    assert score_confidence(np.array([0.0, 1.0, 0.0])) == pytest.approx(1.0)
    assert score_confidence(np.full(3, 1 / 3)) == pytest.approx(0.0)
    assert score_confidence(np.array([1.0])) == 1.0
    # mass on a neighbouring level costs less than mass two levels away
    assert score_confidence(np.array([0.0, 0.7, 0.3, 0.0])) > score_confidence(np.array([0.0, 0.7, 0.0, 0.3]))
    assert 0.0 <= score_confidence(np.array([0.4, 0.2, 0.4])) <= 1.0


def test_fit_temperature_recovers_generating_temperature():
    rng = np.random.default_rng(0)
    true_t = 2.5
    logits, labels = [], []
    for _ in range(3000):
        row = rng.normal(size=4) * 3
        logits.append(row)
        labels.append(int(rng.choice(4, p=softmax(row, true_t))))
    fitted = fit_temperature(logits, labels)
    assert abs(fitted - true_t) / true_t < 0.15


def test_ece_is_zero_when_bins_are_honest():
    probs = [0.9] * 10 + [0.6] * 10
    correct = [True] * 9 + [False] + [True] * 6 + [False] * 4
    assert expected_calibration_error(probs, correct) == pytest.approx(0.0)
    assert expected_calibration_error([0.9] * 10, [False] * 10) == pytest.approx(0.9)


def test_calibration_roundtrip_and_validation(tmp_path):
    from ruling.calibration import Provenance

    path = tmp_path / "c.json"
    Calibration({"choice": 1.7}, provenance=Provenance("m", None, 1, False)).save(path)
    loaded = Calibration.load(path)
    assert loaded.temperature("choice") == 1.7
    assert loaded.temperature("noul") == 1.0
    path.write_text('{"temperatures": {"choice": -1}, "provenance": {"model": "m", "revision": null, "rotations": 1, "prior_debias": false}}')
    with pytest.raises(ValueError):
        Calibration.load(path)


def test_calibration_refuses_a_different_engine(tmp_path):
    from ruling.calibration import Provenance

    fitted = Provenance(model="m", revision="abc", rotations=3, prior_debias=True)
    path = tmp_path / "c.json"
    Calibration({"choice": 1.2}, provenance=fitted).save(path)
    loaded = Calibration.load(path)
    loaded.check(fitted)
    with pytest.raises(ValueError, match="rotations"):
        loaded.check(Provenance(model="m", revision="abc", rotations=1, prior_debias=True))
    with pytest.raises(ValueError, match="revision"):
        loaded.check(Provenance(model="m", revision="def", rotations=3, prior_debias=True))
    with pytest.raises(ValueError, match="provenance"):
        Calibration({"choice": 1.2}).save(path)


def test_cli_bounds_the_mlx_buffer_cache(tmp_path, monkeypatch):
    import mlx.core as mx

    from ruling.cli import main

    monkeypatch.setenv("RULING_CACHE_LIMIT_GB", "1.5")
    monkeypatch.setenv("RULING_MEMORY_FRACTION", "0.4")
    main(["build-corpus", "states", "--output", str(tmp_path / "s.jsonl"), "--count", "4"])
    assert mx.set_cache_limit(0) == int(1.5 * 2**30)
    assert mx.set_memory_limit(mx.device_info()["memory_size"]) == int(mx.device_info()["memory_size"] * 0.4)


def test_coverage_admits_confident_correct_rows_and_stops_at_the_budget():
    from ruling.calibration import confident_error_rate, coverage_at_error

    confidence = [0.99, 0.95, 0.90, 0.80, 0.70, 0.60]
    correct = [True, True, False, True, True, True]
    # Three accepted rows carry one error: 33% > 5%, so only the first two are covered.
    assert coverage_at_error(confidence, correct, 0.05) == pytest.approx(2 / 6)
    # A generous budget covers everything.
    assert coverage_at_error(confidence, correct, 0.5) == 1.0
    # Honest probabilities cover everything; confidently wrong ones cover nothing.
    assert coverage_at_error([0.9, 0.9], [True, True], 0.05) == 1.0
    assert coverage_at_error([0.99, 0.5], [False, True], 0.05) == 0.0
    assert confident_error_rate(confidence, correct) == pytest.approx(1 / 6)


def test_coverage_treats_tied_confidences_as_one_group():
    from ruling.calibration import coverage_at_error

    # Two identical rows, one wrong: admitting them together fails the budget, so a
    # permutation of the tie cannot sneak the correct one in alone.
    assert coverage_at_error([0.8, 0.8], [True, False], 0.05) == 0.0
    assert coverage_at_error([0.8, 0.8], [False, True], 0.05) == 0.0
