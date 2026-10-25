"""Drift: whether a served question's recent answers moved away from the ones before them, read from the ledger."""

import numpy as np

from ruling.drift import drift
from ruling.ledger import DecisionLedger
from ruling.questions import (Choice, ChoiceAnswer, Noul, NoulAnswer, SystemOneRequest, SystemOneResponse, Usage)

TEAM = {"team": Choice(instructions="Which team?", criteria={"billing": None, "shipping": None, "sales": None})}
REFUND = {"refund": Noul(instructions="Does the customer want a refund?")}


def log(ledger, rng, count, weights, model="m", confidence=0.8):
    """`count` decisions on the team question whose answers are drawn from `weights` over the three teams."""
    keys = list(TEAM["team"].criteria)
    for pick in rng.choice(3, size=count, p=weights):
        probabilities = {k: confidence if i == pick else (1 - confidence) / 2 for i, k in enumerate(keys)}
        ledger.record(SystemOneRequest(state="s", questions=TEAM),
                      SystemOneResponse(model=model, usage=Usage(input_tokens=1), answers={
                          "team": ChoiceAnswer(choice=keys[pick], probabilities=probabilities, confidence=confidence)}))


def test_a_steady_question_has_not_drifted_and_a_moved_one_has(tmp_path):
    rng = np.random.default_rng(3)
    steady, moved = DecisionLedger(tmp_path / "steady.sqlite"), DecisionLedger(tmp_path / "moved.sqlite")
    log(steady, rng, 400, [0.6, 0.3, 0.1])
    log(moved, rng, 200, [0.6, 0.3, 0.1])
    log(moved, rng, 200, [0.2, 0.3, 0.5], confidence=0.6)
    [calm] = drift(steady, window=200)
    [shift] = drift(moved, window=200)
    assert (calm["model"], calm["question"], calm["decisions"]) == ("m", "team", [200, 200])
    assert not calm["changed"] and calm["answer_shift"] < 0.1
    # Mean distributions go from about [.54, .33, .19] to [.28, .32, .40]: a shift of about .24.
    assert shift["changed"] and 0.15 < shift["answer_shift"] < 0.35
    assert shift["confidence_change"] < -0.1


def test_drift_is_reported_per_model_and_question_and_only_with_two_full_windows(tmp_path):
    rng = np.random.default_rng(4)
    ledger = DecisionLedger(tmp_path / "ledger.sqlite")
    log(ledger, rng, 120, [0.5, 0.3, 0.2], model="small")
    log(ledger, rng, 40, [0.5, 0.3, 0.2], model="large")
    for _ in range(120):
        ledger.record(SystemOneRequest(state="s", questions=REFUND),
                      SystemOneResponse(model="small", usage=Usage(input_tokens=1), answers={"refund": NoulAnswer(noul=0.9)}))
    reported = {(d["model"], d["question"]): d for d in drift(ledger, window=50)}
    assert set(reported) == {("small", "team"), ("small", "refund")}       # "large" has 40, short of two windows
    assert reported[("small", "refund")]["answer_shift"] == 0 and not reported[("small", "refund")]["changed"]
