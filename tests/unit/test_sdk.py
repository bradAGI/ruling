"""The Python client: typed batch answers, server errors as one exception, retries that wait as told,
gzipped bodies above the threshold, and image blocks built from files."""

import base64
import gzip
import json

import httpx
import pytest
from fastapi.testclient import TestClient

from ruling import Box, Image, RetryPolicy, RulingClient, RulingError
from ruling.ask import AskResponse, FACTS
from ruling.questions import ChoiceAnswer, NoulAnswer
from ruling.sdk import GZIP_THRESHOLD

from .test_batch_api import NOUL, ROW, SINGLE, ScoringEngine, app_with, no_hub_lookups  # noqa: F401


def client_for(app, **kwargs) -> RulingClient:
    """The SDK unchanged, its requests relayed into the app in-process."""
    served = TestClient(app)

    def relay(request: httpx.Request) -> httpx.Response:
        headers = {k: v for k, v in request.headers.items() if k not in ("host", "content-length")}
        response = served.request(request.method, request.url.raw_path.decode(), content=request.content, headers=headers)
        return httpx.Response(response.status_code, headers=response.headers, content=response.content)

    return RulingClient(base_url="http://t", transport=httpx.MockTransport(relay), **kwargs)


def test_predict_parses_answers_into_models_and_leaves_errors_as_dicts():
    with client_for(app_with()) as client:
        rows = client.predict([ROW, {"id": "long", "state": "x" * 2000, "questions": {"q": NOUL}}, SINGLE])
    assert [row.get("id") for row in rows] == ["r1", "long", "r2"]
    assert isinstance(rows[0]["answers"]["refund"], NoulAnswer)
    assert rows[1]["error"]["code"] == "context_too_long"
    assert isinstance(rows[2]["answers"]["decision"], ChoiceAnswer) and rows[2]["answers"]["decision"].choice == "billing"


def test_evaluate_returns_the_report_and_usage_the_buckets():
    with client_for(app_with()) as client:
        report = client.evaluate([ROW, SINGLE])
        usage = client.usage()
    assert report["overall"]["n"] == 2 and set(report["by_question"]) == {"refund", "decision"}
    assert [(b["model"], b["decisions"]) for b in usage] == [("m", 2)]


def test_a_server_error_is_one_exception_carrying_the_body():
    with client_for(app_with(api_key="k")) as client, pytest.raises(RulingError) as refused:
        client.models()
    assert (refused.value.status, refused.value.code) == (401, "unauthorized")
    with client_for(app_with(), api_key="k") as client, pytest.raises(RulingError) as missing:
        client.predict([ROW], model="nope")
    assert (missing.value.status, missing.value.code, missing.value.param) == (404, "unknown_model", "model")
    assert "nope" in str(missing.value)


def test_retries_wait_as_long_as_the_server_says_then_back_off(monkeypatch):
    attempts, slept = [], []
    monkeypatch.setattr("ruling.sdk.sleep", slept.append)

    def handler(request):
        attempts.append(request)
        if len(attempts) == 1:
            return httpx.Response(429, headers={"retry-after": "3"}, json={"error": {"code": "queue_full", "message": "full"}})
        if len(attempts) == 2:
            return httpx.Response(503, json={"error": {"code": "upstream_unavailable", "message": "down"}})
        if len(attempts) == 3:
            raise httpx.ConnectError("refused", request=request)
        return httpx.Response(200, json={"models": []})

    with RulingClient(base_url="http://t", transport=httpx.MockTransport(handler),
                      retry=RetryPolicy(max_retries=3, backoff_initial=0.5, backoff_max=5.0)) as client:
        assert client.models() == []
    assert len(attempts) == 4 and slept == [3.0, 1.0, 2.0]


def test_retries_give_up_after_the_policy_and_raise_the_last_error(monkeypatch):
    monkeypatch.setattr("ruling.sdk.sleep", lambda seconds: None)
    attempts = []

    def handler(request):
        attempts.append(request)
        return httpx.Response(429, headers={"retry-after": "1"}, json={"error": {"code": "queue_full", "message": "full"}})

    with RulingClient(base_url="http://t", transport=httpx.MockTransport(handler)) as client, \
            pytest.raises(RulingError) as refused:
        client.models()
    assert len(attempts) == 3 and refused.value.code == "queue_full"
    with RulingClient(base_url="http://t", transport=httpx.MockTransport(handler), retry=RetryPolicy(max_retries=0)) as client, \
            pytest.raises(RulingError):
        client.models()
    assert len(attempts) == 4


def test_large_bodies_go_gzipped_unless_turned_off():
    seen = []

    def handler(request):
        seen.append((request.headers.get("content-encoding"), request.content))
        return httpx.Response(200, json={"model": "m", "answers": [], "usage": {"input_tokens": 0, "output_tokens": 0}})

    big = [{"state": "x" * GZIP_THRESHOLD, "questions": {"q": NOUL}}]
    with RulingClient(base_url="http://t", transport=httpx.MockTransport(handler)) as client:
        client.predict([ROW])
        client.predict(big)
    with RulingClient(base_url="http://t", transport=httpx.MockTransport(handler), gzip=False) as client:
        client.predict(big)
    assert seen[0][0] is None and json.loads(seen[0][1])["rows"] == [ROW]
    assert seen[1][0] == "gzip" and json.loads(gzip.decompress(seen[1][1]))["rows"] == big
    assert seen[2][0] is None and json.loads(seen[2][1])["rows"] == big


def test_images_are_built_from_files_and_bytes_in_the_wire_shape(tmp_path):
    photo = tmp_path / "receipt.JPG"
    photo.write_bytes(b"\xff\xd8 not really a jpeg")
    assert Image.from_path(photo) == {"type": "image", "media_type": "image/jpeg",
                                      "data": base64.b64encode(photo.read_bytes()).decode()}
    assert Image.from_bytes(b"RIFF", "image/webp")["media_type"] == "image/webp"
    with pytest.raises(ValueError, match="gif"):
        Image.from_path(tmp_path / "animated.gif")
    with pytest.raises(ValueError):
        Image.from_bytes(b"GIF89a", "image/gif")


def test_ask_returns_the_typed_reading_of_a_question_in_words():
    with client_for(app_with()) as client:
        response = client.ask("is water wet?")
    assert isinstance(response, AskResponse) and set(response.facts) == set(FACTS)
    assert response.input == "is water wet?" and response.usage.input_tokens > 0


def test_assets_and_box_questions_go_on_the_wire_as_the_server_reads_them(tmp_path):
    from .test_images import png

    path = tmp_path / "photo.png"
    path.write_bytes(png(2, 2))
    seen = {}

    def handler(request):
        seen.update(json.loads(request.content))
        return httpx.Response(200, json={"model": "m", "usage": {"input_tokens": 1, "output_tokens": 3}, "answers": {
            "where": {"type": "box", "box": {"top": 0.1, "bottom": 0.5, "left": 0.2, "right": 0.9}, "confidence": 0.8}}})

    with RulingClient(base_url="http://t", transport=httpx.MockTransport(handler)) as client:
        response = client.system_one(state="Find the stamp on <photo>", questions={"where": Box(instructions="the stamp on <photo>")},
                                     assets={"<photo>": Image.from_path(path)})
    assert seen["assets"] == {"<photo>": Image.from_path(path)} and seen["questions"]["where"]["type"] == "box"
    assert response.answers["where"].box.right == 0.9 and response.usage.output_tokens == 3
