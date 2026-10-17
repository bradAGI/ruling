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


def test_a_published_zero_stays_exactly_zero():
    rounded = json.loads(json.dumps(PUBLISHED))
    rounded["answers"]["team"]["probabilities"] = {"billing": 1.0, "technical": 0.0, "sales": 0.0}
    engine = engine_with(lambda request: httpx.Response(200, json=rounded))
    response = engine.evaluate(SystemOneRequest(state="state", questions=QUESTIONS))
    assert response.answers["team"].probabilities == {"billing": 1.0, "technical": 0.0, "sales": 0.0}
    assert response.answers["team"].confidence == 1.0


def test_a_hosts_context_limit_is_a_capacity_rejection_not_a_failure():
    from ruling.engine import InputTooLong

    engine = engine_with(lambda request: httpx.Response(
        413, json={"error": {"message": "maximum context length exceeded: request needs 20141 tokens; maximum is 16384"}}))
    with pytest.raises(InputTooLong, match="maximum context length"):
        engine.score("state", QUESTIONS)


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


def test_a_hosts_per_question_context_limit_is_a_capacity_rejection_too():
    """A newer host says 422 context_too_long for one question instead of 413 for the request."""
    from ruling.engine import InputTooLong

    engine = engine_with(lambda request: httpx.Response(
        422, json={"error": {"code": "context_too_long", "param": "questions.team",
                             "message": "questions.team renders to 20141 tokens; the limit is 16384 per question."}}))
    with pytest.raises(InputTooLong, match="maximum context length") as caught:
        engine.score("state", QUESTIONS)
    assert caught.value.question_id == "team"


def test_other_422s_from_a_host_are_refusals_not_capacity():
    engine = engine_with(lambda request: httpx.Response(422, json={"error": {"code": "invalid_input", "message": "x"}}))
    with pytest.raises(RuntimeError):
        engine.score("state", QUESTIONS)


def test_boxes_go_to_the_service_with_its_images_header_and_come_back_as_they_are():
    from ruling.hosted import IMAGES_HEADER, LONG_CONTEXT_HEADER
    from ruling.questions import Box, BoxAnswer

    seen = {}

    def handler(request):
        seen["headers"] = dict(request.headers)
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={**PUBLISHED, "answers": {**PUBLISHED["answers"], "where": {
            "type": "box", "box": {"top": 0.1, "bottom": 0.4, "left": 0.2, "right": 0.9}, "confidence": 0.7}},
            "usage": {"input_tokens": 321, "output_tokens": 12}})

    engine = engine_with(handler)
    assert engine.locates
    response = engine.evaluate(SystemOneRequest(state="A photo <image 1>",
                                                questions={**QUESTIONS, "where": Box(instructions="the stamp")}))
    assert seen["body"]["questions"]["where"] == {"type": "box", "instructions": "the stamp"}
    assert seen["headers"][IMAGES_HEADER.lower()] == "true" and LONG_CONTEXT_HEADER.lower() not in seen["headers"]
    assert response.answers["where"] == BoxAnswer(box={"top": 0.1, "bottom": 0.4, "left": 0.2, "right": 0.9}, confidence=0.7)
    assert response.answers["refund"].noul == pytest.approx(0.9)
    assert response.usage.output_tokens == 12


def test_a_request_asking_for_long_context_forwards_the_services_header():
    from ruling.engine import extended_context
    from ruling.hosted import IMAGES_HEADER, LONG_CONTEXT_HEADER

    seen = []

    def handler(request):
        seen.append(dict(request.headers))
        return httpx.Response(200, json=PUBLISHED)

    engine = engine_with(handler)
    engine.score("plain text", QUESTIONS)
    with extended_context():
        engine.score("plain text", QUESTIONS)
    assert LONG_CONTEXT_HEADER.lower() not in seen[0] and IMAGES_HEADER.lower() not in seen[0]
    assert seen[1][LONG_CONTEXT_HEADER.lower()] == "true"
