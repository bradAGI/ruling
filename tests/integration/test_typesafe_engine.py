"""A ruling server speaks TypeSafe's protocol, so it stands in for the hosted API to prove the engine end to end."""

import numpy as np
import pytest

from ruling.calibration import Calibration, softmax
from ruling.cascade import CascadeEngine
from ruling.hosted import TYPESAFE_PREFIX, TypeSafeEngine
from ruling.questions import Choice, Noul, Score

pytestmark = pytest.mark.integration

STATE = "The forecast says the sky is clear and blue all day with no clouds."
QUESTIONS = {
    "color": Choice(instructions="What color is the sky in the state?", criteria={"red": None, "blue": None, "green": None}),
    "clouds": Score(instructions="How cloudy is it?", criteria=["clear", "partly cloudy", "overcast"]),
    "blue": Noul(instructions="The sky is described as blue."),
}


def test_engine_over_the_protocol_matches_the_engine_behind_it(live_server_with_key, engine):
    remote = TypeSafeEngine(TYPESAFE_PREFIX, Calibration(), live_server_with_key, "s3cret",
                            max_input_tokens=65_536, max_branch_tokens=32_768)
    over_http = remote.score(STATE, QUESTIONS)
    direct = engine.score(STATE, QUESTIONS)
    for qid in QUESTIONS:
        assert over_http.raw[qid].keys == direct.raw[qid].keys
        # The wire carries probabilities rounded by JSON, so agreement is to the published precision.
        np.testing.assert_allclose(softmax(over_http.raw[qid].logits), softmax(direct.raw[qid].logits), atol=1e-6)
    assert over_http.input_tokens == direct.input_tokens


def test_a_hosted_model_can_be_the_second_stage_of_a_cascade(live_server_with_key, single_engine):
    hosted = TypeSafeEngine(TYPESAFE_PREFIX, Calibration(), live_server_with_key, "s3cret",
                            max_input_tokens=65_536, max_branch_tokens=32_768)
    everything_escalates = CascadeEngine(single_engine, hosted, threshold=1.0).score(STATE, QUESTIONS)
    hosted_alone = hosted.score(STATE, QUESTIONS)
    for qid in QUESTIONS:
        np.testing.assert_allclose(softmax(everything_escalates.raw[qid].logits),
                                   softmax(hosted_alone.raw[qid].logits), atol=1e-6)
