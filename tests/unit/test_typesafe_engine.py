"""A System One API as an engine: its probabilities come back as ruling logits, keys aligned, tokens counted."""

import json

import httpx
import numpy as np
import pytest

from ruling.calibration import Calibration, softmax
from ruling.hosted import TYPESAFE_PREFIX, TypeSafeEngine
from ruling.questions import Choice, Noul, Score, SystemOneRequest

QUESTIONS = {
    "team": Choice(instructions="Which team?", criteria={"billing": None, "technical": None, "sales": None}),
    "urgency": Score(instructions="How urgent?", criteria=["can wait", "this week", "today"]),
    "refund": Noul(instructions="The customer asks for a refund."),
}
PUBLISHED = {
    "model": "jev-latest",
    "answers": {
        "team": {"type": "choice", "choice": "technical", "confidence": 0.4,
                 "probabilities": {"sales": 0.1, "technical": 0.6, "billing": 0.3}},   # host's own key order
        "urgency": {"type": "score", "score": 1.5, "confidence": 0.5, "legend": {"0": "can wait", "1": "this week", "2": "today"},
                    "probabilities": {"0": 0.1, "1": 0.3, "2": 0.6}},
        "refund": {"type": "noul", "noul": 0.9},
    },
    "usage": {"input_tokens": 321, "output_tokens": 0},
}


def engine_with(handler):
    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="https://host.invalid")
    return TypeSafeEngine(TYPESAFE_PREFIX + "jev-latest", Calibration(), "https://host.invalid", "k",
                          max_input_tokens=65_536, max_branch_tokens=32_768, client=client)


def test_published_probabilities_round_trip_as_logits_in_canonical_key_order():
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        seen["auth"] = request.headers.get("authorization")
        return httpx.Response(200, json=PUBLISHED)

    engine = engine_with(handler)
    scored = engine.score("state", QUESTIONS)

    assert seen["body"]["model"] == "jev-latest" and set(seen["body"]["questions"]) == set(QUESTIONS)
    assert scored.raw["team"].keys == ["billing", "technical", "sales"]
    np.testing.assert_allclose(softmax(scored.raw["team"].logits), [0.3, 0.6, 0.1], atol=1e-9)
    np.testing.assert_allclose(softmax(scored.raw["urgency"].logits), [0.1, 0.3, 0.6], atol=1e-9)
    np.testing.assert_allclose(softmax(scored.raw["refund"].logits), [0.9, 0.1], atol=1e-9)
    assert scored.input_tokens == 321

    response = engine.evaluate(SystemOneRequest(state="state", questions=QUESTIONS))
    assert response.answers["team"].choice == "technical"
    assert response.answers["refund"].noul == pytest.approx(0.9)
    assert response.model == "typesafe:jev-latest" and response.usage.output_tokens == 0


def test_host_errors_surface_with_their_body():
    engine = engine_with(lambda request: httpx.Response(402, text="Payment Required"))
    with pytest.raises(RuntimeError, match="402.*Payment Required"):
        engine.score("state", QUESTIONS)


def test_an_answer_missing_an_option_is_an_error_not_a_guess():
    broken = json.loads(json.dumps(PUBLISHED))
    del broken["answers"]["team"]["probabilities"]["sales"]
    engine = engine_with(lambda request: httpx.Response(200, json=broken))
    with pytest.raises(ValueError, match="lacks options"):
        engine.score("state", QUESTIONS)
