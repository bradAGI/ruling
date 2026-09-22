"""The cascade sends only doubtful questions onward, and applies each engine's own temperature."""

import numpy as np
import pytest

from ruling.calibration import Calibration, softmax
from ruling.cascade import CascadeEngine
from ruling.engine import RawScore, Scored
from ruling.questions import Choice, Noul, SystemOneRequest


class FakeEngine:
    """Answers from a fixed table of logits and remembers what it was asked."""

    prior_debias = False
    max_options = 255

    def __init__(self, model_id, logits, temperature=1.0, tokens=10):
        self.model_id, self.logits, self.tokens = model_id, logits, tokens
        self.revision = self.adapter = None
        self.rotations = 1
        self.max_input_tokens = self.max_branch_tokens = 1000
        self.calibration = Calibration(temperatures={"choice": temperature, "noul": temperature, "score": temperature})
        self.asked = []

    def score(self, state, questions):
        self.asked.append(sorted(questions))
        return Scored(raw={qid: RawScore(keys=list(self.logits[qid].keys()),
                                         logits=np.array(list(self.logits[qid].values()), dtype=np.float64))
                           for qid in questions},
                      input_tokens=self.tokens * len(questions))


QUESTIONS = {
    "sure": Choice(instructions="?", criteria={"a": None, "b": None, "c": None}),
    "unsure": Choice(instructions="?", criteria={"a": None, "b": None, "c": None}),
    "claim": Noul(instructions="It is so."),
}
SMALL = FakeEngine("small", {"sure": {"a": 6.0, "b": 0.0, "c": 0.0},
                             "unsure": {"a": 0.4, "b": 0.0, "c": 0.0},
                             "claim": {"yes": 0.2, "no": 0.0}})
LARGE = FakeEngine("large", {"sure": {"a": 0.0, "b": 6.0, "c": 0.0},
                             "unsure": {"a": 0.0, "b": 0.0, "c": 6.0},
                             "claim": {"yes": 0.0, "no": 6.0}}, tokens=100)


def test_only_doubtful_questions_reach_the_second_engine():
    small, large = FakeEngine("small", SMALL.logits), FakeEngine("large", LARGE.logits, tokens=100)
    cascade = CascadeEngine(small, large, threshold=0.8)
    scored = cascade.score("state", QUESTIONS)

    assert large.asked == [["claim", "unsure"]]
    assert scored.raw["sure"].keys[int(scored.raw["sure"].logits.argmax())] == "a"      # kept from small
    assert scored.raw["unsure"].keys[int(scored.raw["unsure"].logits.argmax())] == "c"  # replaced by large
    assert scored.raw["claim"].keys[int(scored.raw["claim"].logits.argmax())] == "no"
    assert scored.input_tokens == 3 * 10 + 2 * 100


@pytest.mark.parametrize("threshold, escalated", [(0.0, []), (1.0, [["claim", "sure", "unsure"]])])
def test_threshold_extremes_send_nothing_or_everything(threshold, escalated):
    small, large = FakeEngine("small", SMALL.logits), FakeEngine("large", LARGE.logits)
    CascadeEngine(small, large, threshold).score("state", QUESTIONS)
    assert large.asked == escalated


def test_each_engines_temperature_is_applied_before_the_decision_and_in_the_answer():
    # At T=3 the small engine's "sure" answer softens below the threshold, so it escalates
    # where the same logits at T=1 would not; and the returned logits carry that division.
    small = FakeEngine("small", SMALL.logits, temperature=3.0)
    large = FakeEngine("large", LARGE.logits, temperature=2.0)
    cascade = CascadeEngine(small, large, threshold=0.8)
    scored = cascade.score("state", QUESTIONS)

    assert large.asked == [["claim", "sure", "unsure"]]
    np.testing.assert_allclose(scored.raw["sure"].logits, np.array([0.0, 6.0, 0.0]) / 2.0)
    response = cascade.evaluate(SystemOneRequest(state="state", questions=QUESTIONS))
    expected = softmax(np.array([0.0, 6.0, 0.0]) / 2.0)
    assert response.answers["sure"].probabilities["b"] == pytest.approx(float(expected[1]))
    assert response.model.startswith("cascade:small>large")


def test_limits_are_the_tighter_of_the_pair():
    small, large = FakeEngine("small", SMALL.logits), FakeEngine("large", LARGE.logits)
    large.max_input_tokens, large.max_options = 500, 20
    cascade = CascadeEngine(small, large, threshold=0.5)
    assert (cascade.max_input_tokens, cascade.max_options) == (500, 20)
    with pytest.raises(ValueError):
        CascadeEngine(small, large, threshold=1.5)
