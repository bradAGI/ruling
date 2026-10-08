"""Outages route around: a cascade keeps its primary's answers, a failover hands the request to the fallback."""

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient

from ruling.calibration import Calibration
from ruling.cascade import CascadeEngine
from ruling.engine import RawScore, Scored
from ruling.failover import FailoverEngine
from ruling.hosted import TYPESAFE_PREFIX, HostUnavailable, TypeSafeEngine
from ruling.questions import Choice, Noul, SystemOneRequest
from ruling.server import create_app

QUESTIONS = {
    "team": Choice(instructions="Which team?", criteria={"billing": None, "technical": None}),
    "refund": Noul(instructions="A refund is requested."),
}


class FakeEngine:
    prior_debias = False
    max_options = 255

    def __init__(self, model_id, top, failing=None):
        self.model_id, self.top, self.failing = model_id, top, failing
        self.revision = self.adapter = None
        self.rotations = 1
        self.max_input_tokens = self.max_branch_tokens = 1000
        self.calibration = Calibration()
        self.calls = 0

    def score(self, state, questions):
        self.calls += 1
        if self.failing:
            raise self.failing
        return Scored(raw={qid: RawScore(keys=["billing", "technical"] if qid == "team" else ["yes", "no"],
                                         logits=np.array([self.top, 0.0] if qid == "team" else [self.top, 0.0]))
                           for qid in questions}, input_tokens=7)

    def answer(self, question, raw):
        from ruling.engine import answer
        return answer(question, raw, self.calibration)

    def evaluate(self, request):
        scored = self.score(request.state, request.questions)
        from ruling.questions import SystemOneResponse, Usage
        return SystemOneResponse(model=self.model_id,
                                 answers={qid: self.answer(q, scored.raw[qid]) for qid, q in request.questions.items()},
                                 usage=Usage(input_tokens=scored.input_tokens))


def typesafe_engine(handler):
    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="https://host.invalid")
    return TypeSafeEngine(TYPESAFE_PREFIX + "jev-latest", Calibration(), "https://host.invalid", "k",
                          max_input_tokens=65_536, max_branch_tokens=32_768, client=client)


@pytest.mark.parametrize("status", [429, 502, 503, 504])
def test_an_overloaded_host_is_unavailable_not_a_failure(status):
    engine = typesafe_engine(lambda request: httpx.Response(status, text="overloaded"))
    with pytest.raises(HostUnavailable, match=str(status)):
        engine.score("state", QUESTIONS)


def test_a_timeout_is_unavailable():
    def handler(request):
        raise httpx.ReadTimeout("slow", request=request)

    with pytest.raises(HostUnavailable, match="unreachable"):
        typesafe_engine(handler).score("state", QUESTIONS)


def test_a_refusal_is_still_an_error():
    engine = typesafe_engine(lambda request: httpx.Response(402, text="Payment Required"))
    with pytest.raises(RuntimeError, match="402") as info:
        engine.score("state", QUESTIONS)
    assert not isinstance(info.value, HostUnavailable)


def test_cascade_keeps_the_primary_answers_when_the_secondary_is_down():
    small = FakeEngine("small", top=0.2)                                   # unsure about everything
    down = FakeEngine("large", top=6.0, failing=HostUnavailable("503"))
    scored = CascadeEngine(small, down, threshold=0.9).score("state", QUESTIONS)
    assert down.calls == 1
    assert scored.input_tokens == 7
    np.testing.assert_allclose(scored.raw["team"].logits, [0.2, 0.0])


def test_cascade_does_not_hide_other_failures_of_the_secondary():
    small = FakeEngine("small", top=0.2)
    broken = FakeEngine("large", top=6.0, failing=RuntimeError("bug"))
    with pytest.raises(RuntimeError, match="bug"):
        CascadeEngine(small, broken, threshold=0.9).score("state", QUESTIONS)


def test_failover_answers_from_the_fallback_and_says_so():
    jev = FakeEngine("typesafe:jev-latest", top=6.0, failing=HostUnavailable("503"))
    local = FakeEngine("local", top=-6.0)
    response = FailoverEngine(jev, local).evaluate(SystemOneRequest(state="state", questions=QUESTIONS))
    assert response.model == "local"
    assert response.answers["team"].choice == "technical"
    assert local.calls == 1


def test_failover_leaves_the_primary_alone_when_it_answers():
    jev = FakeEngine("typesafe:jev-latest", top=6.0)
    local = FakeEngine("local", top=-6.0)
    response = FailoverEngine(jev, local).evaluate(SystemOneRequest(state="state", questions=QUESTIONS))
    assert response.model == "typesafe:jev-latest" and response.answers["team"].choice == "billing"
    assert local.calls == 0


def test_failover_raises_when_both_are_down():
    jev = FakeEngine("jev", top=6.0, failing=HostUnavailable("503"))
    local = FakeEngine("local", top=-6.0, failing=HostUnavailable("timeout"))
    with pytest.raises(HostUnavailable, match="timeout"):
        FailoverEngine(jev, local).score("state", QUESTIONS)


def test_server_reports_an_unavailable_host_as_503():
    app = create_app(FakeEngine("jev", top=6.0, failing=HostUnavailable("overloaded")))
    with TestClient(app) as client:
        response = client.post("/v1/systemone", json={"state": "s", "questions": {
            "refund": {"type": "noul", "instructions": "A refund is requested."}}})
    assert response.status_code == 503 and "overloaded" in response.json()["detail"]
