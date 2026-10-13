"""Against a real vLLM, when one is reachable: set RULING_VLLM_BASE_URL and RULING_VLLM_TEST_MODEL."""

import os

import pytest

from ruling.calibration import Calibration
from ruling.hosted import VLLM_PREFIX, VllmEngine
from ruling.questions import Choice, Noul, Score, SystemOneRequest

pytestmark = pytest.mark.integration

BASE_URL = os.environ.get("RULING_VLLM_BASE_URL")
MODEL = os.environ.get("RULING_VLLM_TEST_MODEL")


@pytest.mark.skipif(not (BASE_URL and MODEL), reason="no vLLM server named in RULING_VLLM_BASE_URL and RULING_VLLM_TEST_MODEL")
def test_vllm_answers_every_question_type_with_distributions_that_sum_to_one():
    engine = VllmEngine(VLLM_PREFIX + MODEL, Calibration(), BASE_URL, os.environ.get("RULING_VLLM_API_KEY"),
                        max_input_tokens=65_536, max_branch_tokens=32_768)
    response = engine.evaluate(SystemOneRequest(
        state={"message": "My card was charged twice for one order. Refund the duplicate."},
        questions={
            "team": Choice(instructions="Which team should handle this?",
                           criteria={"billing": "Charges and refunds", "technical": "Bugs", "sales": "Pricing"}),
            "urgency": Score(instructions="How urgent is this?", criteria=["Can wait", "This week", "Today"]),
            "refund": Noul(instructions="The customer asks for a refund."),
        }))
    assert response.model == VLLM_PREFIX + MODEL and response.usage.input_tokens > 0
    assert response.answers["team"].choice == "billing"
    assert abs(sum(response.answers["team"].probabilities.values()) - 1) < 1e-6
    assert abs(sum(response.answers["urgency"].probabilities.values()) - 1) < 1e-6
    assert 0 <= response.answers["refund"].noul <= 1
