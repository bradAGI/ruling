"""Errors leave the server in the hosted API's shape, and a full queue turns requests away with a wait."""

import json
import threading

import anyio
import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient

from ruling.calibration import Calibration
from ruling.engine import InputTooLong, RawScore, Scored, answer
from ruling.hosted import HostUnavailable
from ruling.questions import SystemOneResponse, Usage, state_images
from ruling import registry as registry_module
from ruling.registry import Registry
from ruling.server import Admission, create_app

PAYLOAD = {"state": "The invoice was paid twice.",
           "questions": {"refund": {"type": "noul", "instructions": "A refund is due."}}}


class FakeEngine:
    prior_debias = False
    max_options = 255
    reads_images = False
    locates = False
    rotations = 1
    max_input_tokens = max_branch_tokens = 1000
    revision = adapter = None

    def __init__(self, model_id, failing=None, gate: threading.Event | None = None):
        self.model_id, self.failing, self.gate = model_id, failing, gate
        self.calibration = Calibration()

    def evaluate(self, request):
        if self.gate is not None:
            self.gate.wait()
        if self.failing:
            raise self.failing
        raw = RawScore(keys=["yes", "no"], logits=np.array([2.0, 0.0]))
        return SystemOneResponse(model=self.model_id,
                                 answers={qid: answer(q, raw, self.calibration) for qid, q in request.questions.items()},
                                 usage=Usage(input_tokens=5))


@pytest.fixture(autouse=True)
def no_hub_lookups(monkeypatch):
    """Fake model ids have no Hugging Face repository to ask for a release date."""
    monkeypatch.setattr(registry_module, "model_released", lambda model_id: "2026-01-01")


def run_apart(main) -> None:
    """Run an async main on a thread of its own. The browser tests leave an event loop running
    on the main thread, and anyio refuses to start another one there."""
    failure = []

    def runner():
        try:
            anyio.run(main)
        except BaseException as error:  # re-raised below, so the test fails where it should
            failure.append(error)

    thread = threading.Thread(target=runner)
    thread.start()
    thread.join()
    if failure:
        raise failure[0]


def app_with(engines: dict, default: str, api_key=None, max_pending=4):
    return create_app(Registry(default, engines), api_key, max_pending)


def test_validation_errors_name_the_field():
    with TestClient(app_with({"m": FakeEngine("m")}, "m")) as client:
        response = client.post("/v1/systemone", json={"state": "x", "questions": {}})
    assert response.status_code == 422
    body = response.json()["error"]
    assert body["code"] == "invalid_input" and body["param"] == "questions" and body["message"]


def test_unknown_model_lists_what_the_server_answers_for():
    with TestClient(app_with({"m": FakeEngine("m"), "tuned": FakeEngine("m")}, "m")) as client:
        response = client.post("/v1/systemone", json={**PAYLOAD, "model": "nope"})
    assert response.status_code == 404
    body = response.json()["error"]
    assert body == {"code": "unknown_model", "message": body["message"], "param": "model"}
    assert "'m'" in body["message"] and "'tuned'" in body["message"]


@pytest.mark.parametrize("failing, status, code", [
    (InputTooLong("request is 9 tokens, over the maximum context length of 8"), 422, "context_too_long"),
    (ValueError("state contains the reserved split marker"), 422, "invalid_input"),
    (HostUnavailable("jev is unavailable, returned 503"), 503, "upstream_unavailable"),
])
def test_engine_failures_map_to_codes(failing, status, code):
    with TestClient(app_with({"m": FakeEngine("m", failing=failing)}, "m")) as client:
        response = client.post("/v1/systemone", json=PAYLOAD)
    assert response.status_code == status
    assert response.json()["error"] == {"code": code, "message": str(failing)}


def test_a_question_too_long_for_the_model_is_named_in_the_error():
    too_long = InputTooLong("question 'refund' makes a branch of 9 tokens, over the maximum context length of 8",
                            question_id="refund")
    with TestClient(app_with({"m": FakeEngine("m", failing=too_long)}, "m")) as client:
        response = client.post("/v1/systemone", json=PAYLOAD)
    assert response.status_code == 422
    assert response.json()["error"] == {"code": "context_too_long", "message": str(too_long), "param": "questions.refund"}


@pytest.mark.parametrize("question, detail", [
    ({"type": "choice", "instructions": "pick", "criteria": {"only": None}}, "got 1"),
    ({"type": "choice", "instructions": "pick", "criteria": {f"o{i}": None for i in range(27)}}, "2 to 26"),
    ({"type": "score", "criteria": ["lo", "hi"]}, "needs instructions"),
    ({"type": "noul", "instructions": " "}, "needs instructions"),
])
def test_a_question_the_model_cannot_answer_is_refused_by_name(question, detail):
    engine = FakeEngine("m")
    engine.max_options = 26
    with TestClient(app_with({"m": engine}, "m")) as client:
        response = client.post("/v1/systemone", json={"state": "x", "questions": {"q": question}})
    assert response.status_code == 422
    error = response.json()["error"]
    assert error["code"] == "invalid_options" and error["param"] == "questions.q" and detail in error["message"]


def test_a_noul_described_by_its_criteria_alone_is_answered():
    question = {"type": "noul", "criteria": {"true": "the customer is angry", "false": "the customer is calm"}}
    with TestClient(app_with({"m": FakeEngine("m")}, "m")) as client:
        response = client.post("/v1/systemone", json={"state": "I hate this", "questions": {"q": question}})
    assert response.status_code == 200 and 0 <= response.json()["answers"]["q"]["noul"] <= 1


def test_a_hosted_engines_option_limit_is_refused_by_name():
    from ruling.questions import InvalidOptions

    engine = FakeEngine("m", failing=InvalidOptions("q", "a choice needs 2 to 20 options, got 21"))
    with TestClient(app_with({"m": engine}, "m")) as client:
        response = client.post("/v1/systemone", json=PAYLOAD)
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_options" and response.json()["error"]["param"] == "questions.q"


def test_missing_key_is_unauthorized_in_the_same_shape():
    with TestClient(app_with({"m": FakeEngine("m")}, "m", api_key="k")) as client:
        assert client.get("/v1/models").json()["error"]["code"] == "unauthorized"
        assert client.get("/v1/models", headers={"authorization": "Bearer k"}).status_code == 200


def test_entries_answer_under_their_own_name_and_the_primary_under_the_engines():
    with TestClient(app_with({"m": FakeEngine("m"), "tuned": FakeEngine("m")}, "m")) as client:
        assert client.post("/v1/systemone", json={**PAYLOAD, "model": "tuned"}).json()["model"] == "tuned"
        assert client.post("/v1/systemone", json={**PAYLOAD, "model": "default"}).json()["model"] == "m"
        names = [m["name"] for m in client.get("/v1/models").json()["models"]]
        assert names == ["m", "tuned"]


def test_admission_counts_in_flight_requests_and_estimates_the_wait():
    admission = Admission(2)
    assert admission.enter() and admission.enter() and not admission.enter()
    assert admission.retry_after() == 1
    admission.leave(3.0)
    assert admission.enter() and admission.retry_after() == 6
    admission.leave(1.0)
    admission.leave(1.0)
    assert admission.pending == 0
    with pytest.raises(ValueError):
        Admission(0)


def test_a_full_queue_turns_the_next_request_away_with_retry_after():
    gate = threading.Event()
    app = app_with({"m": FakeEngine("m", gate=gate)}, "m", max_pending=2)
    statuses = {}

    async def run():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as client:
            async def post(i):
                statuses[i] = await client.post("/v1/systemone", json=PAYLOAD)
            async with anyio.create_task_group() as group:
                for i in range(2):
                    group.start_soon(post, i)
                # Give the two admitted requests time to park in the engine; the third then finds the queue full.
                await anyio.sleep(0.2)
                statuses[2] = await client.post("/v1/systemone", json=PAYLOAD)
                gate.set()

    run_apart(run)
    assert statuses[2].status_code == 429
    assert statuses[2].json()["error"]["code"] == "queue_full" and statuses[2].headers["retry-after"] == "1"
    assert statuses[0].status_code == 200 and statuses[1].status_code == 200


PHOTO = {"type": "image", "media_type": "image/png", "data": "iVBORw0KGgo="}


def test_images_reach_only_a_model_that_reads_them():
    sighted = FakeEngine("vision")
    sighted.reads_images = True
    request = {**PAYLOAD, "state": {"photo": PHOTO, "note": "Is it red?"}}
    with TestClient(app_with({"m": FakeEngine("m"), "vision": sighted}, "m")) as client:
        refused = client.post("/v1/systemone", json=request)
        answered = client.post("/v1/systemone", json={**request, "model": "vision"})
        listed = {m["name"]: m["modalities"] for m in client.get("/v1/models").json()["models"]}
    assert refused.status_code == 422
    assert refused.json()["error"]["code"] == "unsupported_modality" and refused.json()["error"]["param"] == "state"
    assert answered.status_code == 200
    assert listed == {"m": ["text"], "vision": ["text", "image"]}


def test_a_malformed_image_is_invalid_input_on_the_state():
    request = {**PAYLOAD, "state": {"photo": {**PHOTO, "media_type": "image/gif"}}}
    with TestClient(app_with({"m": FakeEngine("m")}, "m")) as client:
        response = client.post("/v1/systemone", json=request)
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_input" and response.json()["error"]["param"].startswith("state")


def test_retraining_history_covers_only_retrain_adapters_and_needs_the_key(tmp_path):
    from ruling.retrain import REPORT, promote

    adapters = tmp_path / "tickets"
    (adapters / "v1").mkdir(parents=True)
    (adapters / "v1" / REPORT).write_text('{"version": "v1", "promoted": true}')
    promote(adapters, adapters / "v1")
    tuned, plain, pinned = FakeEngine("m"), FakeEngine("m"), FakeEngine("m")
    tuned.adapter_path, pinned.adapter_path = adapters / "current", adapters / "v1"
    app = app_with({"tuned": tuned, "plain": plain, "pinned": pinned}, "plain", api_key="k")
    with TestClient(app) as client:
        refused = client.get("/v1/retrain")
        listed = client.get("/v1/retrain", headers={"authorization": "Bearer k"}).json()
    assert refused.status_code == 401
    assert listed == {"models": [{"name": "tuned", "current": "v1", "versions": [{"version": "v1", "promoted": True}]}]}


class SightedEngine(FakeEngine):
    """Reads images and remembers the last request, so a test can see what the server handed it."""

    reads_images = True

    def evaluate(self, request):
        self.seen = request
        return super().evaluate(request)


def png_bytes() -> bytes:
    from .test_images import png
    return png(2, 2)


def test_assets_arrive_as_multipart_parts_or_inline_and_reach_the_engine_as_images():
    import base64

    sighted = SightedEngine("vision")
    body = {"state": "Inspect <product> against <listing>", "model": "vision",
            "questions": {"same": {"type": "noul", "instructions": "Is <product> the item in <listing>?"}},
            "assets": {"<product>": {"type": "image", "part": "asset-0"}, "<listing>": {"type": "image", "part": "asset-1"}}}
    with TestClient(app_with({"vision": sighted}, "vision")) as client:
        multipart = client.post("/v1/systemone", data={"request": json.dumps(body)},
                                files=[("asset-0", ("asset-0", png_bytes(), "image/png")),
                                       ("asset-1", ("asset-1", png_bytes(), "image/png"))])
        assert multipart.status_code == 200, multipart.text
        assert sighted.seen.questions["same"].instructions == "Is <image 1> the item in <image 2>?"
        assert sighted.seen.state["text"] == "Inspect <image 1> against <image 2>"
        assert [i.media_type for i in state_images(sighted.seen.state)] == ["image/png", "image/png"]
        inline = client.post("/v1/systemone", json={**body, "assets": {
            "<product>": {"type": "image", "data_base64": base64.b64encode(png_bytes()).decode()},
            "<listing>": {"type": "image", "data_base64": base64.b64encode(png_bytes()).decode()}}})
        assert inline.status_code == 200
        missing = client.post("/v1/systemone", data={"request": json.dumps(body)},
                              files=[("asset-0", ("asset-0", png_bytes(), "image/png"))])
        no_request = client.post("/v1/systemone", files=[("asset-0", ("asset-0", png_bytes(), "image/png"))])
        bad_json = client.post("/v1/systemone", data={"request": "{not json"})
    assert missing.status_code == 422 and missing.json()["error"]["param"] == "assets"
    assert "asset-1" in missing.json()["error"]["message"]
    assert no_request.status_code == 422 and no_request.json()["error"]["param"] == "request"
    assert bad_json.status_code == 422 and "JSON" in bad_json.json()["error"]["message"]


def test_a_malformed_json_body_still_names_the_field():
    with TestClient(app_with({"m": FakeEngine("m")}, "m")) as client:
        response = client.post("/v1/systemone", json={**PAYLOAD, "questions": {"q": {"type": "noul", "instruction": "x"}}})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_input" and response.json()["error"]["param"].startswith("questions.q")


def test_a_box_question_is_refused_by_a_model_that_does_not_locate_and_listed_where_one_does():
    locating = SightedEngine("vision")
    locating.locates = True
    request = {"state": {"photo": PHOTO}, "questions": {"where": {"type": "box", "instructions": "the stamp"}}}
    with TestClient(app_with({"m": FakeEngine("m"), "vision": locating}, "m")) as client:
        refused = client.post("/v1/systemone", json=request)
        no_image = client.post("/v1/systemone", json={**request, "model": "vision", "state": "no picture"})
        listed = {m["name"]: m["question_types"] for m in client.get("/v1/models").json()["models"]}
    assert refused.status_code == 422 and refused.json()["error"]["code"] == "unsupported_modality"
    assert refused.json()["error"]["param"] == "questions"
    assert no_image.status_code == 422 and no_image.json()["error"]["code"] == "invalid_options"
    assert listed == {"m": ["choice", "score", "noul"], "vision": ["choice", "score", "noul", "box"]}


def test_the_long_context_header_reaches_the_engine_as_the_extended_limits():
    from ruling.engine import long_context

    class Watching(FakeEngine):
        def evaluate(self, request):
            self.extended = long_context.get()
            return super().evaluate(request)

    watching = Watching("m")
    with TestClient(app_with({"m": watching}, "m")) as client:
        client.post("/v1/systemone", json=PAYLOAD)
        plain = watching.extended
        client.post("/v1/systemone", json=PAYLOAD, headers={"TypeSafe-Danger-Zone-Long-Context": "true"})
        asked = watching.extended
        client.post("/v1/systemone", json=PAYLOAD)
        after = watching.extended
    assert (plain, asked, after) == (False, True, False)
