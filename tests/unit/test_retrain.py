"""Retraining promotes a new adapter only when it is better on held-out data, and repoints atomically."""

import itertools
import json

import numpy as np

from ruling.calibration import Calibration, Provenance
from ruling.evals import Observation, Record, decision_blocks
from ruling.retrain import CURRENT, REPORT, history, next_version, promote, score, should_promote, trainable

QUESTION = {"q": {"type": "noul", "instructions": "It is about money."}}


def test_promotion_needs_lower_log_loss_and_no_lost_accuracy():
    grid = [0.6, 0.7, 0.8]
    for (acc_now, nll_now), (acc_new, nll_new) in itertools.product(itertools.product(grid, grid), repeat=2):
        promoted, reason = should_promote({"n": 10, "accuracy": acc_now, "nll": nll_now},
                                          {"n": 10, "accuracy": acc_new, "nll": nll_new})
        assert promoted == (nll_new < nll_now and acc_new >= acc_now) and reason


def test_a_more_accurate_but_overconfident_candidate_is_judged_on_its_calibrated_log_loss():
    """Training sharpens an adapter faster than it corrects it, so a candidate that is right more
    often can still lose on raw log loss; fitting each version's temperature first removes the
    confidence scale from the verdict."""
    rng = np.random.default_rng(0)

    def observations(accuracy: float, scale: float) -> list[Observation]:
        rows = []
        for i in range(200):
            label = int(rng.integers(2))
            logits = np.zeros(2)
            logits[label if rng.random() < accuracy else 1 - label] = rng.uniform(0.5, 2.0) * scale
            rows.append(Observation(i, "q", "noul", None, ["no", "yes"], logits, label, {}, None))
        return rows

    versions = {"current": observations(0.8, 1.0), "candidate": observations(0.9, 8.0)}
    raw = {name: decision_blocks(rows, Calibration())["overall"] for name, rows in versions.items()}
    assert raw["candidate"]["accuracy"] > raw["current"]["accuracy"] and raw["candidate"]["nll"] > raw["current"]["nll"]
    assert not should_promote(raw["current"], raw["candidate"])[0]

    provenance = Provenance("m", None, 1, True)
    scored = {name: score(rows, provenance) for name, rows in versions.items()}
    assert should_promote(scored["current"][0], scored["candidate"][0])[0]
    assert scored["candidate"][1].temperatures["noul"] > scored["current"][1].temperatures["noul"] > 0
    assert scored["candidate"][1].provenance == provenance
    assert scored["candidate"][0]["accuracy"] == raw["candidate"]["accuracy"]


def test_versions_count_up_past_every_existing_one(tmp_path):
    assert next_version(tmp_path) == tmp_path / "v1"
    for name in ("v1", "v3", "notes"):
        (tmp_path / name).mkdir()
    promote(tmp_path, tmp_path / "v3")
    assert next_version(tmp_path) == tmp_path / "v4"


def test_promotion_repoints_current_and_leaves_nothing_behind(tmp_path):
    for name in ("v1", "v2"):
        (tmp_path / name).mkdir()
    promote(tmp_path, tmp_path / "v1")
    promote(tmp_path, tmp_path / "v2")
    assert (tmp_path / CURRENT).resolve() == (tmp_path / "v2").resolve()
    # A relative link survives the adapters directory being moved or mounted elsewhere.
    assert str((tmp_path / CURRENT).readlink()) == "v2"
    assert sorted(p.name for p in tmp_path.iterdir()) == [CURRENT, "v1", "v2"]


def test_decisions_on_held_out_states_never_reach_training():
    held = [Record(state="A benchmark state.", questions=QUESTION)]
    served = [Record(state="A  benchmark   state.", questions=QUESTION, labels={"q": True}),
              Record(state="A customer's state.", questions=QUESTION, labels={"q": False})]
    kept, dropped = trainable(served, held)
    assert [r.state for r in kept] == ["A customer's state."] and dropped == 1


def test_history_lists_finished_versions_in_number_order_and_names_the_served_one(tmp_path):
    for number in (1, 2, 10):
        version = tmp_path / f"v{number}"
        version.mkdir()
        (version / REPORT).write_text(json.dumps({"version": version.name, "promoted": number != 10}))
    (tmp_path / "v11").mkdir()  # still training: no report yet
    promote(tmp_path, tmp_path / "v2")
    listed = history(tmp_path)
    assert listed["current"] == "v2"
    assert [v["version"] for v in listed["versions"]] == ["v1", "v2", "v10"]
    assert history(tmp_path / "missing") == {"current": None, "versions": []}
