"""vLLM's own `/v1/systemone` as an engine: same answers, vLLM's limits and refusals, picked by adapter name."""

import json

import httpx
import numpy as np
import pytest

from ruling.calibration import Calibration, Provenance, softmax
from ruling.config import Settings
from ruling.engine import InputTooLong
from ruling.hosted import VLLM_PREFIX, HostUnavailable, VllmEngine
from ruling.questions import Choice, InvalidOptions, Noul, Score
from ruling.registry import build_engine

QUESTIONS = {
    "team": Choice(instructions="Which team?", criteria={"billing": None, "technical": None, "sales": None}),
    "tone": Score(instructions="How upset?", criteria=["calm", "annoyed", "furious"]),
    "urgent": Noul(instructions="Needs a reply within the hour."),
}
# The documented response of vLLM's structured decisions endpoint, extra fields included.
VLLM_RESPONSE = {
    "id": "decision-1", "object": "structured_decision", "created": 1790000000, "model": "tickets-lora",
    "answers": {
        "team": {"type": "choice", "choice": "billing", "probabilities": {"billing": 0.97, "sales": 0.02, "technical": 0.01},
                 "confidence": 0.90},
        "tone": {"type": "score", "score": 0.4, "legend": {"0": "calm", "1": "annoyed", "2": "furious"},
                 "probabilities": {"0": 0.7, "1": 0.2, "2": 0.1}, "confidence": 0.65},
        "urgent": {"type": "noul", "noul": 0.02, "probabilities": {"yes": 0.02, "no": 0.98}, "confidence": 0.95},
    },
    "usage": {"input_tokens": 142, "output_tokens": 3},
    "diagnostics": {"team": {"label_mass": 0.93, "argmax_is_label": True}},
}


def engine_with(handler, calibration=None):
    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://vllm.invalid")
    return VllmEngine(VLLM_PREFIX + "tickets-lora", calibration or Calibration(), "http://vllm.invalid", None,
                      max_input_tokens=65_536, max_branch_tokens=32_768, client=client)


def test_an_adapter_is_asked_by_name_and_its_answers_come_back_in_canonical_order():
    seen = {}

    def handler(request):
        seen["path"], seen["body"] = request.url.path, json.loads(request.content)
        seen["auth"] = request.headers.get("authorization")
        return httpx.Response(200, json=VLLM_RESPONSE)

    engine = engine_with(handler)
    scored = engine.score({"ticket": "charged twice"}, QUESTIONS)
    assert seen["path"] == "/v1/systemone" and seen["body"]["model"] == "tickets-lora" and seen["auth"] is None
    assert engine.model_id == "vllm:tickets-lora" and engine.provenance.model == "vllm:tickets-lora"
    assert scored.raw["team"].keys == ["billing", "technical", "sales"]
    np.testing.assert_allclose(softmax(scored.raw["team"].logits), [0.97, 0.01, 0.02], atol=1e-9)
    np.testing.assert_allclose(softmax(scored.raw["urgent"].logits), [0.02, 0.98], atol=1e-9)
    assert scored.input_tokens == 142


def test_a_calibration_fit_for_the_vllm_model_is_accepted_and_one_for_another_is_not():
    fit = Calibration({"choice": 1.3}, Provenance("vllm:tickets-lora", None, 1, False, None))
    assert engine_with(lambda r: httpx.Response(200, json=VLLM_RESPONSE), fit).calibration.temperature("choice") == 1.3
    other = Calibration({"choice": 1.3}, Provenance("typesafe:tickets-lora", None, 1, False, None))
    with pytest.raises(ValueError, match="model"):
        engine_with(lambda r: httpx.Response(200, json=VLLM_RESPONSE), other)


def test_a_choice_wider_than_vllms_alphabet_runs_in_two_stages_of_at_most_64_questions():
    """vLLM labels options A to Z and takes 64 questions a request, so a 100-way choice is first asked as
    100 yes/no questions, in two requests, and then as a choice over the strongest few."""
    requests = []

    def handler(request):
        body = json.loads(request.content)
        requests.append(body)
        answers = {}
        for qid, question in body["questions"].items():
            if question["type"] == "noul":
                # Only option o42 is a confident yes; the rest are all equally unlikely.
                yes = 0.9 if "o42" in json.dumps(question["instructions"]) else 0.1
                answers[qid] = {"type": "noul", "noul": yes, "probabilities": {"yes": yes, "no": 1 - yes}, "confidence": 0.8}
            else:
                keys = list(question["criteria"])
                probabilities = {k: 0.0 for k in keys}
                probabilities[keys[0]] = 1.0
                answers[qid] = {"type": "choice", "choice": keys[0], "probabilities": probabilities, "confidence": 1.0}
        return httpx.Response(200, json={"answers": answers, "usage": {"input_tokens": len(body["questions"])}})

    engine = engine_with(handler)
    wide = {"pick": Choice(instructions="Which one?", criteria={f"o{i}": None for i in range(100)})}
    scored = engine.score("s", wide)
    assert engine.max_options == 255 and engine.pass_options == 26
    stage_one = [r for r in requests if all(q["type"] == "noul" for q in r["questions"].values())]
    final = [r for r in requests if "pick" in r["questions"]]
    assert len(stage_one) == 2 and [len(r["questions"]) for r in stage_one] == [64, 36]
    assert len(final) == 1 and list(final[0]["questions"]["pick"]["criteria"])[0] == "o42"
    assert len(final[0]["questions"]["pick"]["criteria"]) <= 26
    probabilities = softmax(scored.raw["pick"].logits)
    assert scored.raw["pick"].keys[int(np.argmax(probabilities))] == "o42"
    assert sum(1 for p in probabilities if p > 0) <= 26
    assert scored.input_tokens == 64 + 36 + 1  # the fake bills one token per question


def test_a_choice_over_the_platform_limit_is_still_refused_before_any_request():
    calls = []
    engine = engine_with(lambda r: calls.append(r) or httpx.Response(200, json=VLLM_RESPONSE))
    wide = {"pick": Choice.model_construct(instructions="?", criteria={f"o{i}": None for i in range(256)})}
    with pytest.raises(InvalidOptions, match="2 to 255 options") as caught:
        engine.score("s", wide)
    assert caught.value.question_id == "pick" and calls == []


@pytest.mark.parametrize("status, text, expected, fragment", [
    (400, "label 'Q' is not one distinct token", ValueError, "refused"),
    (501, "no read strategy", RuntimeError, "no read strategy"),
    (413, "too long", InputTooLong, "maximum context length"),
    (503, "loading", HostUnavailable, "unavailable"),
    (500, "boom", RuntimeError, "returned 500"),
])
def test_vllm_refusals_map_to_ruling_errors(status, text, expected, fragment):
    engine = engine_with(lambda r: httpx.Response(status, text=text))
    with pytest.raises(expected, match=fragment):
        engine.score("s", QUESTIONS)


def test_build_engine_needs_the_vllm_url_and_reads_it_from_the_environment(monkeypatch):
    with pytest.raises(ValueError, match="RULING_VLLM_BASE_URL"):
        build_engine(Settings(model="vllm:tickets-lora"))
    monkeypatch.setenv("RULING_VLLM_BASE_URL", "http://gpu-box:8000")
    monkeypatch.setenv("RULING_VLLM_API_KEY", "k")
    monkeypatch.setenv("RULING_MODEL", "vllm:tickets-lora")
    settings = Settings.from_env()
    assert settings.vllm_base_url == "http://gpu-box:8000" and settings.vllm_api_key == "k"
    engine = build_engine(settings)
    assert isinstance(engine, VllmEngine) and engine.model_id == "vllm:tickets-lora"
