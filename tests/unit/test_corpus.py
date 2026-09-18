import numpy as np
import pytest

from ruling.corpus import civil_record, clinc_record, emotions_records, nli_record, boolq_record
from ruling.evals import Record
from ruling.train import target_vector


def test_nli_record_asks_verdict_and_entailment():
    r = nli_record("snli", "A dog runs.", "An animal moves.", "entailment")
    assert r.labels == {"verdict": "supported", "supported": True}
    assert set(r.questions["verdict"].criteria) == {"supported", "insufficient", "contradicted"}


def test_clinc_record_offers_a_none_option_and_sometimes_hides_the_gold_intent():
    intents = [f"intent_{i}" for i in range(150)]
    rng = np.random.default_rng(3)
    hidden = shown = 0
    for _ in range(300):
        r = clinc_record("how do I pay my bill", "intent_7", intents, rng)
        keys = list(r.questions["intent"].criteria)
        assert "none_of_these" in keys and 4 <= len(keys) <= 13
        if r.labels["intent"] == "none_of_these":
            assert "intent_7" not in keys
            hidden += 1
        else:
            assert r.labels["intent"] == "intent_7"
            shown += 1
    assert hidden > 30 and shown > 150
    oos = clinc_record("what is the meaning of life", None, intents, rng)
    assert oos.labels["intent"] == "none_of_these"


def test_emotions_become_soft_targets_from_rater_fractions():
    raters = [{"id": "x", "text": "I can't believe you did this", "anger": 1, "joy": 0, "surprise": 1},
              {"id": "x", "text": "I can't believe you did this", "anger": 1, "joy": 0, "surprise": 0},
              {"id": "x", "text": "I can't believe you did this", "anger": 0, "joy": 0, "surprise": 0}]
    rng = np.random.default_rng(0)
    (r,) = emotions_records(raters, ["anger", "joy", "surprise"], rng, questions_per_text=3)
    assert target_vector(r, "anger") == pytest.approx({"yes": 2 / 3, "no": 1 / 3})
    assert target_vector(r, "joy") == {"yes": 0.0, "no": 1.0}
    assert r.labels["anger"] is True and r.labels["surprise"] is False


def test_civil_comment_targets_are_rater_fractions():
    r = civil_record({"text": "you are an idiot", "toxicity": 0.8, "insult": 0.9, "threat": 0.0})
    assert target_vector(r, "toxic") == pytest.approx({"yes": 0.8, "no": 0.2})
    assert r.labels["insult"] is True and r.labels["threat"] is False


def test_boolq_record():
    r = boolq_record("Tides are caused by the moon.", "are tides caused by the moon", True)
    assert r.labels == {"answer": True} and r.questions["answer"].instructions.endswith("?")
    assert isinstance(r, Record)
