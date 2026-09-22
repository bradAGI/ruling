"""A real engine cascading to itself must answer exactly as it does alone, whatever the threshold."""

import numpy as np
import pytest

from ruling.calibration import softmax
from ruling.cascade import CascadeEngine
from ruling.questions import Choice, Noul, Score, SystemOneRequest

pytestmark = pytest.mark.integration

STATE = "The forecast says the sky is clear and blue all day with no clouds."
QUESTIONS = {
    "color": Choice(instructions="What color is the sky in the state?", criteria={"red": None, "blue": None, "green": None}),
    "clouds": Score(instructions="How cloudy is it?", criteria=["clear", "partly cloudy", "overcast"]),
    "blue": Noul(instructions="The sky is described as blue."),
}


@pytest.mark.parametrize("threshold", [0.0, 1.0])
def test_self_cascade_is_the_identity(single_engine, threshold):
    alone = single_engine.score(STATE, QUESTIONS)
    cascade = CascadeEngine(single_engine, single_engine, threshold).score(STATE, QUESTIONS)
    for qid in QUESTIONS:
        assert cascade.raw[qid].keys == alone.raw[qid].keys
        np.testing.assert_allclose(softmax(cascade.raw[qid].logits), softmax(alone.raw[qid].logits), atol=1e-6)
    # Escalating everything costs the second pass; escalating nothing costs only the first.
    assert cascade.input_tokens == alone.input_tokens * (2 if threshold == 1.0 else 1)


def test_cascade_serves_typed_answers(single_engine):
    cascade = CascadeEngine(single_engine, single_engine, 0.5)
    response = cascade.evaluate(SystemOneRequest(state=STATE, questions=QUESTIONS))
    assert response.answers["color"].choice == "blue"
    assert response.answers["blue"].noul > 0.5
    assert response.usage.output_tokens == 0 and response.model.startswith("cascade:")
