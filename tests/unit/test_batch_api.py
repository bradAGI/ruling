"""Batches of rows are answered or scored in one request, one bad row never fails the batch,
usage is counted per hour, and a gzipped body is read as if it were plain."""

import gzip
import json
import threading
from datetime import UTC, datetime

import anyio
import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient

from ruling.calibration import Calibration
from ruling.engine import InputTooLong, RawScore, Scored, answer
from ruling.hosted import HostUnavailable
from ruling.questions import MAX_PREDICT_ROWS, SystemOneResponse, Usage
from ruling import registry as registry_module
from ruling.registry import Registry
from ruling.server import create_app
from ruling.usage import UsageMeter

from .test_server_errors import run_apart

NOUL = {"type": "noul", "instructions": "A refund is due."}
CHOICE = {"type": "choice", "instructions": "Which team?", "criteria": {"billing": None, "technical": None}}
ROW = {"id": "r1", "state": "The invoice was paid twice.", "questions": {"refund": NOUL}, "labels": {"refund": True}}
SINGLE = {"id": "r2", "state": "A bug report.", "question": CHOICE, "label": "technical"}
PHOTO = {"type": "image", "media_type": "image/png", "data": "iVBORw0KGgo="}


class ScoringEngine:
    """Scores through `score`, as a real engine does, so `evals.collect` and `evals.predict` both work.
    A state over `limit` characters is too long; `yes` and the first option always win."""

    prior_debias = False
    max_options = 255
    reads_images = False
    locates = False
    rotations = 1
    max_input_tokens = max_branch_tokens = 1000
    revision = adapter = None

    def __init__(self, model_id="m", limit=1000, failing=None, gate: threading.Event | None = None):
        self.model_id, self.limit, self.failing, self.gate = model_id, limit, failing, gate
        self.calibration = Calibration()

    def score(self, state, questions):
        if self.gate is not None:
            self.gate.wait()
        if self.failing:
            raise self.failing
        if len(json.dumps(state)) > self.limit:
            raise InputTooLong("request is over the maximum context length")
        raw = {}
        for qid, q in questions.items():
            keys = ["yes", "no"] if q.type == "noul" else list(q.criteria) if q.type == "choice" else \
                [str(i) for i in range(len(q.criteria))]
            raw[qid] = RawScore(keys=keys, logits=np.array([2.0] + [0.0] * (len(keys) - 1)))
        return Scored(raw=raw, input_tokens=len(questions))

    def evaluate(self, request):
        scored = self.score(request.state, request.questions)
        return SystemOneResponse(model=self.model_id, usage=Usage(input_tokens=scored.input_tokens),
                                 answers={qid: answer(q, scored.raw[qid], self.calibration)
                                          for qid, q in request.questions.items()})


@pytest.fixture(autouse=True)
def no_hub_lookups(monkeypatch):
    monkeypatch.setattr(registry_module, "model_released", lambda model_id: "2026-01-01")


def app_with(engines=None, default="m", api_key=None, max_pending=4, usage=None):
    engines = engines or {"m": ScoringEngine("m")}
    return create_app(Registry(default, engines), api_key, max_pending, usage=usage)


# ---- predict ---------------------------------------------------------------

def test_predict_answers_every_row_in_order_and_keeps_a_bad_row_in_its_place():
    wordy = {"id": "long", "state": "x" * 2000, "questions": {"q": NOUL}}
    wide = {"id": "wide", "state": "s", "questions": {"q": {**CHOICE, "criteria": {"only": None}}}}
    shaped = {"id": "shape", "state": "s"}
    pictured = {"id": "pic", "state": {"photo": PHOTO}, "questions": {"q": NOUL}}
    anonymous = {"state": "s", "questions": {"q": NOUL}}
    rows = [ROW, wordy, SINGLE, wide, shaped, "not a row", pictured, anonymous]
    with TestClient(app_with()) as client:
        response = client.post("/v1/predict", json={"rows": rows})
        unnamed = client.post("/v1/predict", json={"rows": [{"state": "s"}]}).json()["answers"][0]
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["model"] == "m" and body["usage"] == {"input_tokens": 3, "output_tokens": 0}
    entries = body["answers"]
    assert [e.get("id") for e in entries] == ["r1", "long", "r2", "wide", "shape", None, "pic", None]
    assert entries[0]["answers"]["refund"]["type"] == "noul" and 0 < entries[0]["answers"]["refund"]["noul"] < 1
    assert entries[2]["answers"]["decision"]["choice"] == "billing"
    assert "id" not in entries[7] and entries[7]["answers"]["q"]["type"] == "noul"
    assert "id" not in entries[5] and unnamed == {"error": unnamed["error"]}
    codes = [e["error"]["code"] for e in entries if "error" in e]
    assert codes == ["context_too_long", "invalid_options", "invalid_input", "invalid_input", "unsupported_modality"]
    assert all(e["error"]["message"] for e in entries if "error" in e)


def test_predict_resolves_the_model_like_a_decision_does():
    with TestClient(app_with({"m": ScoringEngine("m"), "tuned": ScoringEngine("m")})) as client:
        assert client.post("/v1/predict", json={"rows": [ROW], "model": "tuned"}).json()["model"] == "tuned"
        assert client.post("/v1/predict", json={"rows": [ROW], "model": "default"}).json()["model"] == "m"
        missing = client.post("/v1/predict", json={"rows": [ROW], "model": "nope"})
    assert missing.status_code == 404 and missing.json()["error"]["code"] == "unknown_model"


@pytest.mark.parametrize("path", ["/v1/predict", "/v1/evaluations"])
def test_a_batch_needs_between_one_and_the_limit_of_rows(path):
    with TestClient(app_with()) as client:
        empty = client.post(path, json={"rows": []})
        over = client.post(path, json={"rows": [ROW] * (MAX_PREDICT_ROWS + 1)})
    for response in (empty, over):
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "invalid_input" and response.json()["error"]["param"] == "rows"


@pytest.mark.parametrize("path", ["/v1/predict", "/v1/evaluations", "/v1/usage"])
def test_batch_and_usage_routes_need_the_key(path):
    with TestClient(app_with(api_key="k")) as client:
        refused = client.get(path) if path == "/v1/usage" else client.post(path, json={"rows": [ROW]})
        assert refused.status_code == 401 and refused.json()["error"]["code"] == "unauthorized"


def test_a_host_outage_fails_the_whole_batch():
    engine = ScoringEngine("m", failing=HostUnavailable("down"))
    with TestClient(app_with({"m": engine})) as client:
        response = client.post("/v1/predict", json={"rows": [ROW]})
    assert response.status_code == 503 and response.json()["error"]["code"] == "upstream_unavailable"


def test_a_batch_holds_one_admission_slot_for_its_whole_duration():
    gate = threading.Event()
    app = app_with({"m": ScoringEngine("m", gate=gate)}, max_pending=1)
    statuses = {}

    async def run():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as client:
            async def post():
                statuses["batch"] = await client.post("/v1/predict", json={"rows": [ROW, ROW]})
            async with anyio.create_task_group() as group:
                group.start_soon(post)
                await anyio.sleep(0.2)
                statuses["turned_away"] = await client.post("/v1/systemone", json={k: ROW[k] for k in ("state", "questions")})
                gate.set()

    run_apart(run)
    assert statuses["turned_away"].status_code == 429
    assert statuses["turned_away"].json()["error"]["code"] == "queue_full"
    assert statuses["batch"].status_code == 200


# ---- evaluations -----------------------------------------------------------

def test_evaluations_return_the_eval_report_for_the_rows():
    rows = [ROW, SINGLE, {"state": "unlabeled", "questions": {"q": NOUL}}]
    with TestClient(app_with()) as client:
        response = client.post("/v1/evaluations", json={"rows": rows})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["model"] == "m"
    assert set(body) >= {"model", "question_types", "overall", "by_type", "by_question"}
    assert body["overall"]["n"] == 2 and 0 <= body["overall"]["accuracy"] <= 1
    assert set(body["overall"]["selective"]) == {"0.5", "0.7", "0.9", "0.95"}
    assert set(body["by_type"]) == {"noul", "choice"} and set(body["by_question"]) == {"refund", "decision"}
    assert body["question_types"]["noul"]["count"] == 2 and body["question_types"]["noul"]["labeled_count"] == 1


def test_evaluations_count_a_row_too_long_for_the_model_as_unscored():
    rows = [ROW, {"state": "x" * 2000, "questions": {"refund": NOUL}, "labels": {"refund": False}}]
    with TestClient(app_with()) as client:
        body = client.post("/v1/evaluations", json={"rows": rows}).json()
    assert body["question_types"]["noul"] | {"count": 1, "unscored": 1} == body["question_types"]["noul"]
    assert body["overall"]["n"] == 1


def test_evaluations_refuse_rows_without_any_labels_or_a_row_the_model_cannot_read():
    with TestClient(app_with()) as client:
        unlabeled = client.post("/v1/evaluations", json={"rows": [{"state": "s", "questions": {"q": NOUL}}]})
        malformed = client.post("/v1/evaluations", json={"rows": [ROW, {"state": "s"}]})
        pictured = client.post("/v1/evaluations", json={"rows": [ROW, {"state": {"p": PHOTO}, "questions": {"q": NOUL}}]})
    assert unlabeled.status_code == 422
    assert unlabeled.json()["error"]["code"] == "invalid_input" and unlabeled.json()["error"]["param"] == "rows"
    assert malformed.status_code == 422
    assert malformed.json()["error"]["code"] == "invalid_input" and malformed.json()["error"]["param"] == "rows.1"
    assert pictured.status_code == 422
    assert pictured.json()["error"]["code"] == "unsupported_modality" and pictured.json()["error"]["param"] == "rows.1"


# ---- usage ---------------------------------------------------------------

def test_usage_counts_decisions_and_batch_rows_per_hour_and_model():
    clock = {"now": datetime(2026, 10, 16, 14, 20, tzinfo=UTC).timestamp()}
    meter = UsageMeter(now=lambda: clock["now"])
    engines = {"m": ScoringEngine("m"), "tuned": ScoringEngine("m")}
    decision = {k: ROW[k] for k in ("state", "questions")}
    wordy = {"state": "x" * 2000, "questions": {"q": NOUL}, "labels": {"q": True}}
    with TestClient(app_with(engines, usage=meter)) as client:
        assert client.post("/v1/systemone", json=decision).status_code == 200
        assert client.post("/v1/systemone", json={**decision, "model": "tuned"}).status_code == 200
        assert client.post("/v1/predict", json={"rows": [ROW, wordy, SINGLE]}).status_code == 200
        clock["now"] += 3600
        assert client.post("/v1/evaluations", json={"rows": [ROW, wordy]}).status_code == 200
        assert client.post("/v1/systemone", json={**decision, "model": "nope"}).status_code == 404
        everything = client.get("/v1/usage").json()["usage"]
        later = client.get("/v1/usage", params={"starting_on": "2026-10-16T15:00:00Z"}).json()["usage"]
        bad = client.get("/v1/usage", params={"starting_on": "yesterday"})
    assert everything == [
        {"hour": "2026-10-16T14:00:00+00:00", "model": "m", "decisions": 3, "input_tokens": 3},
        {"hour": "2026-10-16T14:00:00+00:00", "model": "tuned", "decisions": 1, "input_tokens": 1},
        {"hour": "2026-10-16T15:00:00+00:00", "model": "m", "decisions": 1, "input_tokens": 1},
    ]
    assert later == everything[2:]
    assert bad.status_code == 422 and bad.json()["error"]["param"] == "starting_on"


# ---- compressed bodies -------------------------------------------------------

def test_a_gzipped_body_is_read_as_plain_json_and_a_broken_one_is_refused():
    decision = {k: ROW[k] for k in ("state", "questions")}
    headers = {"content-encoding": "gzip", "content-type": "application/json"}
    with TestClient(app_with()) as client:
        packed = client.post("/v1/systemone", content=gzip.compress(json.dumps(decision).encode()), headers=headers)
        broken = client.post("/v1/systemone", content=b"this is not gzip", headers=headers)
        batch = client.post("/v1/predict", content=gzip.compress(json.dumps({"rows": [ROW]}).encode()), headers=headers)
    assert packed.status_code == 200 and packed.json()["answers"]["refund"]["type"] == "noul"
    assert batch.status_code == 200 and batch.json()["answers"][0]["id"] == "r1"
    assert broken.status_code == 422
    assert broken.json()["error"]["code"] == "invalid_input" and "gzip" in broken.json()["error"]["message"]


def test_a_zstd_body_is_read_as_the_hosted_sdk_sends_it_and_other_encodings_are_415():
    """TypeSafe's SDKs compress every body over 1 KiB with zstd on their own; a server that only inflated
    gzip would refuse their ordinary requests."""
    from backports.zstd import compress

    decision = {k: ROW[k] for k in ("state", "questions")}
    with TestClient(app_with()) as client:
        packed = client.post("/v1/systemone", content=compress(json.dumps(decision).encode()),
                             headers={"content-encoding": "zstd", "content-type": "application/json"})
        broken = client.post("/v1/systemone", content=b"not zstd", headers={"content-encoding": "zstd"})
        unknown = client.post("/v1/systemone", content=b"x", headers={"content-encoding": "br"})
    assert packed.status_code == 200 and packed.json()["answers"]["refund"]["type"] == "noul"
    assert broken.status_code == 422 and "zstd" in broken.json()["error"]["message"]
    assert unknown.status_code == 415 and unknown.json()["error"]["code"] == "unsupported_encoding"


def test_a_body_over_the_hosted_limit_is_413_whether_declared_or_inflated():
    from ruling.server import MAX_BODY_BYTES

    bomb = gzip.compress(b"0" * (MAX_BODY_BYTES + 1))
    with TestClient(app_with()) as client:
        declared = client.post("/v1/systemone", content=b"{}",
                               headers={"content-type": "application/json", "content-length": str(MAX_BODY_BYTES + 1)})
        inflated = client.post("/v1/systemone", content=bomb, headers={"content-encoding": "gzip"})
    assert declared.status_code == 413 and declared.json()["error"]["code"] == "body_too_large"
    assert inflated.status_code == 413 and "inflates" in inflated.json()["error"]["message"]
