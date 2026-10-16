"""The decision ledger: what was asked and answered, the outcomes reported later, and the training records they make."""

import pytest

from ruling.ledger import DecisionLedger, UnknownDecision
from ruling.questions import Choice, ChoiceAnswer, Noul, NoulAnswer, SystemOneRequest, SystemOneResponse, Usage

REQUEST = SystemOneRequest(
    state={"message": "I was charged twice."},
    questions={"team": Choice(instructions="Which team?", criteria={"billing": None, "technical": None}),
               "refund": Noul(instructions="Does the customer want a refund?")})
RESPONSE = SystemOneResponse(
    model="m", usage=Usage(input_tokens=10),
    answers={"team": ChoiceAnswer(choice="billing", probabilities={"billing": 0.9, "technical": 0.1}, confidence=0.8),
             "refund": NoulAnswer(noul=0.7)})


def test_outcomes_become_training_records_and_survive_a_restart(tmp_path):
    path = tmp_path / "ledger.sqlite"
    ledger = DecisionLedger(path)
    labeled = ledger.record(REQUEST, RESPONSE)
    ledger.record(REQUEST, RESPONSE)          # never resolved, so it teaches nothing
    ledger.outcome(labeled, {"team": "billing"})
    ledger.outcome(labeled, {"team": "technical", "refund": True})   # a later report corrects an earlier one

    records = list(DecisionLedger(path).records())
    assert len(records) == 1
    assert records[0].id == labeled
    assert records[0].state == REQUEST.state and set(records[0].questions) == {"team", "refund"}
    assert records[0].labels == {"team": "technical", "refund": True}


def test_an_outcome_must_name_a_decision_and_fit_its_question(tmp_path):
    ledger = DecisionLedger(tmp_path / "ledger.sqlite")
    with pytest.raises(UnknownDecision):
        ledger.outcome("no-such-id", {"team": "billing"})
    decision = ledger.record(REQUEST, RESPONSE)
    with pytest.raises(ValueError):
        ledger.outcome(decision, {"team": "sales"})
    with pytest.raises(ValueError):
        ledger.outcome(decision, {"urgency": 2})
    assert list(ledger.records()) == []


def test_unresolved_decisions_older_than_the_retention_are_dropped_on_the_next_record(tmp_path):
    path = tmp_path / "ledger.sqlite"
    clock = {"now": 1_000_000.0}
    ledger = DecisionLedger(path, retain_seconds=3600, now=lambda: clock["now"])
    resolved, stale = ledger.record(REQUEST, RESPONSE), ledger.record(REQUEST, RESPONSE)
    ledger.outcome(resolved, {"team": "billing"})
    clock["now"] += 3601
    fresh = ledger.record(REQUEST, RESPONSE)          # this write is what prunes
    assert ledger.ids() == [resolved, fresh]
    with pytest.raises(UnknownDecision):
        ledger.outcome(stale, {"team": "billing"})
    # Resolved decisions are training data and are kept however old; without a retention nothing is pruned.
    clock["now"] += 10 * 3600
    later = ledger.record(REQUEST, RESPONSE)
    assert ledger.ids() == [resolved, later]
    forever = DecisionLedger(path, now=lambda: clock["now"] + 10 * 3600)
    kept = forever.record(REQUEST, RESPONSE)
    assert forever.ids() == [resolved, later, kept]
