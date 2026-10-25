"""Decision metrics: invariants over random inputs, and one example checked against the reference numbers."""

import numpy as np
import pytest

from ruling.calibration import Calibration
from ruling.evals import Observation, report
from ruling.metrics import THRESHOLDS, decision_metrics

THRESHOLD_KEYS = ["0.5", "0.7", "0.9", "0.95"]


def random_pairs(rng, count, soft=False):
    probs, targets = [], []
    for _ in range(count):
        width = int(rng.integers(2, 6))
        probs.append(rng.dirichlet(np.ones(width) * rng.choice([0.2, 1.0, 5.0])))
        if soft:
            targets.append(rng.dirichlet(np.ones(width)))
        else:
            targets.append(np.eye(width)[rng.integers(width)])
    return probs, targets


@pytest.mark.parametrize("soft", [False, True])
def test_metrics_hold_their_ranges_and_selective_is_consistent_for_any_input(soft):
    rng = np.random.default_rng(7)
    for trial in range(300):
        probs, targets = random_pairs(rng, int(rng.integers(1, 12)), soft)
        m = decision_metrics(probs, targets)
        assert m["n"] == len(probs)
        assert 0.0 <= m["accuracy"] <= 1.0 and 0.0 <= m["ece"] <= 1.0 and 0.0 <= m["mean_confidence"] <= 1.0
        assert m["nll"] >= 0.0 and 0.0 <= m["brier"] <= 2.0
        assert list(m["selective"]) == THRESHOLD_KEYS == [str(t) for t in THRESHOLDS]
        coverages = [m["selective"][k]["coverage"] for k in THRESHOLD_KEYS]
        assert all(0.0 <= c <= 1.0 for c in coverages) and coverages == sorted(coverages, reverse=True)
        confidence = np.array([p.max() for p in probs])
        correct = np.array([p.argmax() == y.argmax() for p, y in zip(probs, targets)])
        for threshold, key in zip(THRESHOLDS, THRESHOLD_KEYS):
            covered = confidence >= threshold
            cell = m["selective"][key]
            assert cell["coverage"] == pytest.approx(covered.mean())
            if covered.any():
                assert cell["accuracy"] == pytest.approx(correct[covered].mean())
            else:
                assert cell["accuracy"] is None


def test_hand_computed_example_matches_the_reference_implementation_to_the_digit():
    # Four decisions, worked through the metric definitions by hand:
    #   0: p=[.7,.2,.1] y=[1,0,0]  right, conf .7, nll -ln .7,        brier .09+.04+.01=.14
    #   1: p=[.4,.6]    y=[1,0]    wrong, conf .6, nll -ln .4,        brier .36+.36=.72
    #   2: p=[.95,.05]  y=[.8,.2]  right, conf .95, nll -(.8 ln .95 + .2 ln .05), brier .0225+.0225=.045
    #   3: p=[.5,.5]    y=[0,1]    half right (a tie of two), conf .5, nll -ln .5, brier .25+.25=.5
    probs = [np.array([0.7, 0.2, 0.1]), np.array([0.4, 0.6]), np.array([0.95, 0.05]), np.array([0.5, 0.5])]
    targets = [np.array([1.0, 0, 0]), np.array([1.0, 0]), np.array([0.8, 0.2]), np.array([0.0, 1.0])]
    m = decision_metrics(probs, targets)
    assert m["n"] == 4
    assert m["accuracy"] == pytest.approx(0.625)
    assert m["nll"] == pytest.approx((-np.log(0.7) - np.log(0.4) - (0.8 * np.log(0.95) + 0.2 * np.log(0.05)) - np.log(0.5)) / 4)
    assert m["brier"] == pytest.approx((0.14 + 0.72 + 0.045 + 0.5) / 4)
    assert m["mean_confidence"] == pytest.approx((0.7 + 0.6 + 0.95 + 0.5) / 4)
    # Bins (lo, hi]: 0.5 falls in (0.4, 0.5], 0.6 in (0.5, 0.6], 0.7 in (0.6, 0.7], 0.95 in (0.9, 1.0].
    # Every bin holds one decision, so each contributes |conf - correct| / 4.
    assert m["ece"] == pytest.approx((abs(0.5 - 0.5) + abs(0.6 - 0) + abs(0.7 - 1) + abs(0.95 - 1)) / 4)
    assert m["selective"] == {
        "0.5": {"coverage": 1.0, "accuracy": pytest.approx(0.625)},
        "0.7": {"coverage": 0.5, "accuracy": 1.0},
        "0.9": {"coverage": 0.25, "accuracy": 1.0},
        "0.95": {"coverage": 0.25, "accuracy": 1.0},
    }


def test_a_zero_confidence_decision_lands_in_the_first_bin():
    # The bin membership rule is lo < c <= hi, with c == 0 placed in bin 0 instead of falling out.
    m = decision_metrics([np.array([0.0, 0.0, 0.0])], [np.array([0.0, 0.0, 1.0])])
    assert m["ece"] == pytest.approx(abs(0.0 - 1 / 3))


def test_metrics_do_not_depend_on_the_order_options_are_listed_in():
    """Options tied for the top used to credit whichever came first, so reordering them changed accuracy."""
    rng = np.random.default_rng(11)
    for trial in range(300):
        width = int(rng.integers(2, 6))
        count = int(rng.integers(1, 10))
        # Probabilities on a coarse grid so exact ties are common, in the model's answer and in soft targets.
        probs = [np.bincount(rng.integers(0, width, 4), minlength=width) / 4 for _ in range(count)]
        targets = [np.bincount(rng.integers(0, width, 2), minlength=width) / 2 for _ in range(count)]
        orders = [rng.permutation(width) for _ in range(count)]
        before = decision_metrics(probs, targets)
        after = decision_metrics([p[o] for p, o in zip(probs, orders)], [y[o] for y, o in zip(targets, orders)])
        assert flat(after) == pytest.approx(flat(before))


def flat(metrics: dict) -> dict:
    """The metrics with the selective block spread into top-level keys, which `pytest.approx` can compare."""
    selective = metrics.pop("selective")
    return metrics | {f"{t}/{k}": v for t, cell in selective.items() for k, v in cell.items()}


def test_empty_input_and_mismatched_lengths():
    assert decision_metrics([], []) == {"n": 0}
    with pytest.raises(ValueError, match="same length"):
        decision_metrics([np.array([1.0, 0.0])], [])


def test_report_carries_overall_by_type_and_by_question_blocks():
    def obs(i, qid, qtype, keys, logits, label, references=None):
        return Observation(record=i, question_id=qid, question_type=qtype, family=None, keys=keys,
                           logits=None if logits is None else np.array(logits), label=label,
                           references={k: np.array(v) for k, v in (references or {}).items()}, reversed_argmax=None)

    observations = [
        obs(0, "team", "choice", ["a", "b"], [2.0, 0.0], 0),
        obs(0, "ok", "noul", ["yes", "no"], [0.0, 2.0], 0),
        obs(1, "team", "choice", ["a", "b"], [0.0, 2.0], 1, {"target": [0.3, 0.7]}),
        obs(1, "ok", "noul", ["yes", "no"], [1.0, 0.0], None),       # unlabeled: excluded
        obs(2, "team", "choice", ["a", "b"], None, 0),               # unscored: excluded
    ]
    result = report(observations, Calibration())
    assert result["overall"]["n"] == 3
    assert result["overall"]["accuracy"] == pytest.approx(2 / 3)
    assert set(result["by_type"]) == {"choice", "noul"}
    assert result["by_type"]["choice"]["n"] == 2 and result["by_type"]["noul"]["n"] == 1
    assert result["by_question"]["team"]["n"] == 2 and result["by_question"]["ok"]["n"] == 1
    # The soft target is the y vector where present: record 1's nll is -0.7 log p(b), not -log p(b).
    p_b = 1 / (1 + np.exp(-2.0))
    assert result["by_type"]["choice"]["nll"] == pytest.approx((-np.log(p_b) - (0.3 * np.log(1 - p_b) + 0.7 * np.log(p_b))) / 2)
    assert set(result["overall"]["selective"]) == set(THRESHOLD_KEYS)
    assert report([obs(2, "team", "choice", ["a", "b"], None, 0)], Calibration())["overall"] == {"n": 0}
