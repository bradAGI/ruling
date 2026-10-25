"""The decision ledger: what was asked and answered, the outcomes reported later, and the training records they make."""

import pytest

from pydantic import ValidationError

from ruling.ledger import Correction, DecisionLedger, Outcome, OutcomeRefused, UnknownDecision
from ruling.questions import (Choice, ChoiceAnswer, Noul, NoulAnswer, Score, ScoreAnswer, SystemOneRequest,
                              SystemOneResponse, Usage)

REQUEST = SystemOneRequest(
    state={"message": "I was charged twice."},
    questions={"team": Choice(instructions="Which team?", criteria={"billing": None, "technical": None}),
               "refund": Noul(instructions="Does the customer want a refund?")})
WIDE = SystemOneRequest(
    state="A parcel arrived crushed.",
    questions={"team": Choice(instructions="Which team?", criteria={"billing": None, "shipping": None, "sales": None}),
               "urgency": Score(instructions="How urgent?", criteria=["low", "medium", "high"])})
WIDE_RESPONSE = SystemOneResponse(
    model="m", usage=Usage(input_tokens=10),
    answers={"team": ChoiceAnswer(choice="shipping", probabilities={"billing": 0.1, "shipping": 0.8, "sales": 0.1},
                                  confidence=0.7),
             "urgency": ScoreAnswer(score=1.2, legend={"0": "low", "1": "medium", "2": "high"},
                                    probabilities={"0": 0.2, "1": 0.4, "2": 0.4 - 1e-9}, confidence=0.3)})
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


def right(answer=None):
    return Correction(correct=True) if answer is None else Correction(correct=True, answer=answer)


def wrong(answer=None):
    return Correction(correct=False) if answer is None else Correction(correct=False, answer=answer)


def test_a_correction_becomes_the_label_it_implies(tmp_path):
    """Marking an answer right labels it with what was served; marking it wrong labels it with the answer given,
    or, for a question of two options, with the other one."""
    ledger = DecisionLedger(tmp_path / "ledger.sqlite")
    two, wide = ledger.record(REQUEST, RESPONSE), ledger.record(WIDE, WIDE_RESPONSE)
    assert ledger.outcome(two, corrections={"team": right(), "refund": wrong()}).labels == \
        {"team": "billing", "refund": False}
    assert ledger.outcome(two, corrections={"team": wrong()}).labels == {"team": "technical", "refund": False}
    assert ledger.outcome(wide, corrections={"team": wrong("sales"), "urgency": right()}).labels == \
        {"team": "sales", "urgency": 1}
    assert ledger.outcome(wide, {"urgency": 0}, {"team": right()}).labels == {"team": "shipping", "urgency": 0}


def test_a_correction_that_does_not_say_what_was_right_is_refused_and_stores_nothing(tmp_path):
    ledger = DecisionLedger(tmp_path / "ledger.sqlite")
    wide = ledger.record(WIDE, WIDE_RESPONSE)
    with pytest.raises(OutcomeRefused, match="right answer") as refused:
        ledger.outcome(wide, corrections={"urgency": right(), "team": wrong()})      # one of three: which?
    assert refused.value.param == "corrections"
    with pytest.raises(ValueError, match="served"):
        ledger.outcome(wide, corrections={"team": wrong("shipping")})                 # wrong, yet what was served
    with pytest.raises(ValueError):
        ledger.outcome(wide, corrections={"team": wrong("returns")})                 # not an option
    with pytest.raises(ValueError):
        ledger.outcome(wide, corrections={"refund": right()})                        # not a question asked
    assert list(ledger.records()) == []


def test_an_outcome_body_says_something_once_and_only_a_wrong_mark_carries_an_answer():
    with pytest.raises(ValidationError):
        Outcome()
    with pytest.raises(ValidationError):
        Outcome(labels={"team": "billing"}, corrections={"team": right()})
    with pytest.raises(ValidationError):
        Correction(correct=True, answer="billing")
    assert Outcome(labels={"team": "billing"}).corrections == {}


def test_a_batch_of_outcomes_is_stored_whole_or_not_at_all(tmp_path):
    ledger = DecisionLedger(tmp_path / "ledger.sqlite")
    first, second = ledger.record(REQUEST, RESPONSE), ledger.record(REQUEST, RESPONSE)
    with pytest.raises(UnknownDecision):
        ledger.outcomes([(first, Outcome(labels={"team": "billing"})), ("no-such-id", Outcome(labels={"refund": True}))])
    with pytest.raises(OutcomeRefused) as refused:
        ledger.outcomes([(first, Outcome(labels={"team": "billing"})), (second, Outcome(labels={"team": "sales"}))])
    assert refused.value.param == "outcomes.1.labels"
    assert list(ledger.records()) == []
    stored = ledger.outcomes([(first, Outcome(labels={"team": "billing"})),
                              (second, Outcome(corrections={"refund": wrong()}))])
    assert [r.labels for r in stored] == [{"team": "billing"}, {"refund": False}]
    assert {r.id for r in ledger.records()} == {first, second}
