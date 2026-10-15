import json
import shutil
import subprocess
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from ruling.evals import load_dataset
from ruling.ledger import DecisionLedger
from ruling.questions import Choice, Noul
from ruling.sdk import RulingClient
from ruling.registry import Registry
from ruling.server import create_app

from .conftest import _serve

pytestmark = pytest.mark.integration

TS_SDK = Path(__file__).resolve().parents[2] / "sdks" / "typescript" / "index.js"
PAYLOAD = {
    "state": {"message": "My card was charged twice for one order. Refund the duplicate."},
    "questions": {
        "team": {"type": "choice", "instructions": "Which team should handle this?",
                 "criteria": {"billing": "Charges and refunds", "technical": "Bugs and errors", "sales": "Pricing"}},
        "refund": {"type": "noul", "instructions": "The customer asks for a refund."},
    },
}


@pytest.fixture(scope="module")
def app(engine):
    return create_app(Registry.single(engine), None, 4)


def test_endpoint_returns_typed_answers(app, engine):
    with TestClient(app) as client:
        response = client.post("/v1/systemone", json=PAYLOAD)
    assert response.status_code == 200
    body = response.json()
    assert body["model"] == engine.model_id
    assert body["answers"]["team"]["choice"] == "billing"
    assert set(body["answers"]["team"]["probabilities"]) == {"billing", "technical", "sales"}
    assert body["answers"]["refund"]["noul"] > 0.5
    assert body["usage"]["output_tokens"] == 0


def test_validation_and_model_errors(app):
    def code(body):
        response = client.post("/v1/systemone", json=body)
        return response.status_code, response.json()["error"]["code"], response.json()["error"].get("param")

    with TestClient(app) as client:
        assert code({"state": "x", "questions": {}}) == (422, "invalid_input", "questions")
        one = {"state": "x", "questions": {"q": {"type": "choice", "instructions": "q", "criteria": {"one": None}}}}
        assert code(one) == (422, "invalid_options", "questions.q")
        assert code({**PAYLOAD, "model": "nope"})[:2] == (404, "unknown_model")
        too_many = {"state": "x", "questions": {"q": {"type": "choice", "instructions": "q",
                    "criteria": {f"o{i}": None for i in range(256)}}}}
        assert code(too_many) == (422, "invalid_input", "questions.q.choice.criteria")
        noul = {"type": "noul", "instructions": "x"}
        assert code({"state": "x", "questions": {f"q{i}": noul for i in range(65)}}) == (422, "invalid_input", "questions")
        wordy = {"state": " ".join(f"w{i}" for i in range(40000)), "questions": {"q": noul}}
        assert code(wordy) == (422, "context_too_long", "questions.q")


def test_a_noul_described_by_its_criteria_alone_reads_the_state(app):
    questions = {"angry": {"type": "noul", "criteria": {"true": "the customer is angry", "false": "the customer is calm"}}}
    with TestClient(app) as client:
        furious = client.post("/v1/systemone", json={"state": "This is the third time. I am done with you.", "questions": questions})
        pleased = client.post("/v1/systemone", json={"state": "Thanks, that fixed it, great service.", "questions": questions})
    assert furious.status_code == 200 and pleased.status_code == 200
    assert furious.json()["answers"]["angry"]["noul"] > pleased.json()["answers"]["angry"]["noul"]


def test_models_and_playground(app, engine):
    with TestClient(app) as client:
        body = client.get("/v1/models").json()
        page = client.get("/")
    assert body == {"models": [{"name": engine.model_id, "description": body["models"][0]["description"],
                                "release_date": body["models"][0]["release_date"],
                                "id": engine.model_id, "base": engine.model_id, "revision": engine.revision,
                                "adapter": None, "calibration": {},
                                "max_input_tokens": 8192, "max_branch_tokens": engine.max_branch_tokens,
                                "max_options": engine.max_options, "rotations": engine.rotations,
                                "prior_debias": engine.prior_debias}]}
    assert page.status_code == 200 and "/v1/systemone" in page.text
    # The page lists the registry and sends the picked name; a stale page would still post with no model.
    assert "/v1/models" in page.text and 'id="model"' in page.text and "model: model.value" in page.text


def test_bearer_auth_when_configured(engine):
    with TestClient(create_app(Registry.single(engine), "s3cret", 4)) as client:
        assert client.get("/v1/models").status_code == 401
        assert client.get("/v1/models", headers={"authorization": "Bearer wrong"}).status_code == 401
        assert client.get("/v1/models", headers={"authorization": "Bearer s3cret"}).status_code == 200
        assert client.get("/health").status_code == 200


def test_python_sdk_against_live_server(live_server):
    with RulingClient(base_url=live_server) as client:
        response = client.system_one(
            state=PAYLOAD["state"],
            questions={"team": Choice(**{k: v for k, v in PAYLOAD["questions"]["team"].items() if k != "type"}),
                       "refund": Noul(instructions="The customer asks for a refund.")},
        )
        assert response.answers["team"].choice == "billing"
        assert client.models()[0].max_options > 2


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_python_sdk_authenticates_with_the_servers_key(live_server_with_key, monkeypatch):
    question = {"refund": Noul(instructions="The customer asks for a refund.")}
    with RulingClient(base_url=live_server_with_key, api_key="s3cret") as client:
        assert client.system_one(state=PAYLOAD["state"], questions=question).answers["refund"].noul > 0.5
    monkeypatch.setenv("RULING_API_KEY", "s3cret")
    with RulingClient(base_url=live_server_with_key) as client:
        assert client.models()
    monkeypatch.delenv("RULING_API_KEY")
    with RulingClient(base_url=live_server_with_key) as client, pytest.raises(httpx.HTTPStatusError) as refused:
        client.models()
    assert refused.value.response.status_code == 401


def test_typescript_sdk_against_live_server(live_server):
    script = f"""
import {{ RulingClient, choice, noul }} from {json.dumps(TS_SDK.as_uri())};
const client = new RulingClient({{ baseUrl: {json.dumps(live_server)} }});
const r = await client.systemOne({{
  state: {json.dumps(PAYLOAD["state"])},
  questions: {{ team: choice("Which team should handle this?", {json.dumps(PAYLOAD["questions"]["team"]["criteria"])}),
               refund: noul("The customer asks for a refund.") }},
}});
const models = await client.models();
console.log(JSON.stringify({{ team: r.answers.team.choice, refund: r.answers.refund.noul, models: models.length }}));
"""
    result = subprocess.run(["node", "--input-type=module", "-e", script], capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    data = json.loads(result.stdout.strip().splitlines()[-1])
    assert data["team"] == "billing" and data["refund"] > 0.5 and data["models"] == 1


def test_official_typesafe_sdk_against_live_server(live_server_with_key, engine, monkeypatch):
    """TypeSafe's own client, unmodified, pointed at ruling by environment variable alone."""
    from typesafe_sdk import Choice, Noul, Score, TypeSafeClient

    monkeypatch.setenv("TYPESAFE_BASE_URL", live_server_with_key)
    monkeypatch.setenv("TYPESAFE_API_KEY", "s3cret")
    with TypeSafeClient() as client:
        response = client.system_one(
            state=PAYLOAD["state"],
            questions={
                "team": Choice(instructions="Which team should handle this?",
                               criteria=PAYLOAD["questions"]["team"]["criteria"]),
                "urgency": Score(instructions="How urgent is this?", criteria=["Can wait", "This week", "Today"]),
                "refund": Noul(instructions="The customer asks for a refund."),
            },
        )
        models = client.models.list()

    assert response.choices["team"].choice == "billing"
    assert response.nouls["refund"].noul > 0.5
    assert set(response.scores["urgency"].probabilities) == {0, 1, 2}
    assert response.usage.input_tokens > 0 and response.usage.output_tokens == 0
    assert models.models[0].name == engine.model_id and models.models[0].release_date


def test_a_logged_decision_takes_its_outcome_and_exports_as_training_data(engine, tmp_path):
    ledger = tmp_path / "ledger.sqlite"
    server, thread, url = _serve(create_app(Registry.single(engine), None, 4, DecisionLedger(ledger)))
    try:
        with RulingClient(base_url=url) as client:
            decision = client.system_one(state=PAYLOAD["state"],
                                         questions={"refund": Noul(instructions="The customer asks for a refund.")}).id
            assert client.outcome(decision, {"refund": True}) == {"refund": True}
            assert client.system_one(state="A benchmark row.", questions={"refund": Noul(instructions="A refund.")},
                                     exclude_from_training=True).id is None
            with pytest.raises(httpx.HTTPStatusError) as wrong:
                client.outcome(decision, {"refund": "maybe"})
            assert wrong.value.response.json()["error"]["code"] == "invalid_input"
            with pytest.raises(httpx.HTTPStatusError) as missing:
                client.outcome("no-such-id", {"refund": True})
            assert missing.value.response.json()["error"]["code"] == "unknown_decision"
    finally:
        server.should_exit = True
        thread.join(timeout=10)
    exported = tmp_path / "train.jsonl"
    subprocess.run(["uv", "run", "ruling", "export-decisions", "--ledger", str(ledger), "--output", str(exported)],
                   check=True)
    [record] = load_dataset(exported)
    assert record.state == PAYLOAD["state"] and record.labels == {"refund": True}


def test_a_decision_kept_out_of_training_is_answered_but_not_logged(engine, tmp_path):
    # Endor's clients send this field to keep benchmark and evaluation traffic out of training.
    app = create_app(Registry.single(engine), None, 4, DecisionLedger(tmp_path / "ledger.sqlite"))
    with TestClient(app) as client:
        excluded = client.post("/v1/systemone", json={**PAYLOAD, "exclude_from_training": True}).json()
        logged = client.post("/v1/systemone", json=PAYLOAD).json()
    assert excluded["answers"] and "id" not in excluded
    assert logged["id"]


def test_without_a_ledger_the_wire_format_is_unchanged(app):
    # Clients built on the hosted SDK decode these bodies; nothing may appear that the hosted API does not send.
    with TestClient(app) as client:
        assert set(client.post("/v1/systemone", json=PAYLOAD).json()) == {"model", "answers", "usage"}
        assert client.post("/v1/decisions/x/outcome", json={"labels": {}}).status_code == 404


def test_endors_client_against_live_server(live_server_with_key, engine):
    """Endor's own client, unmodified, pointed at ruling by its base URL alone."""
    from endor import Choice, EndorClient, Noul, RetryPolicy, Score

    with EndorClient(api_key="s3cret", base_url=live_server_with_key, retry=RetryPolicy(max_retries=0)) as client:
        response = client.system_one(
            PAYLOAD["state"],
            {"team": Choice(instructions="Which team should handle this?",
                            criteria=PAYLOAD["questions"]["team"]["criteria"]),
             "urgency": Score(instructions="How urgent is this?", criteria=["Can wait", "This week", "Today"]),
             "refund": Noul(instructions="The customer asks for a refund.")},
            model="default",
        )
        models = client.models.list()

    assert response.model == engine.model_id
    assert response.choices["team"].choice == "billing"
    assert response.nouls["refund"].noul > 0.5
    assert set(response.scores["urgency"].probabilities) == {"0", "1", "2"}
    assert models.models[0].name == engine.model_id
