import json
import shutil
import subprocess
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from ruling.questions import Choice, Noul
from ruling.sdk import RulingClient
from ruling.server import create_app

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
    return create_app(engine)


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
    with TestClient(app) as client:
        assert client.post("/v1/systemone", json={"state": "x", "questions": {}}).status_code == 422
        bad = {"state": "x", "questions": {"q": {"type": "choice", "instructions": "q", "criteria": {"one": None}}}}
        assert client.post("/v1/systemone", json=bad).status_code == 422
        assert client.post("/v1/systemone", json={**PAYLOAD, "model": "nope"}).status_code == 404
        too_many = {"state": "x", "questions": {"q": {"type": "choice", "instructions": "q",
                    "criteria": {f"o{i}": None for i in range(256)}}}}
        assert client.post("/v1/systemone", json=too_many).status_code == 422


def test_models_and_playground(app, engine):
    with TestClient(app) as client:
        body = client.get("/v1/models").json()
        page = client.get("/")
    assert body == {"models": [{"name": engine.model_id, "description": body["models"][0]["description"],
                                "release_date": body["models"][0]["release_date"],
                                "id": engine.model_id, "revision": engine.revision, "adapter": None,
                                "max_input_tokens": 8192, "max_branch_tokens": engine.max_branch_tokens,
                                "max_options": engine.max_options, "rotations": engine.rotations,
                                "prior_debias": engine.prior_debias}]}
    assert page.status_code == 200 and "/v1/systemone" in page.text


def test_bearer_auth_when_configured(engine):
    with TestClient(create_app(engine, api_key="s3cret")) as client:
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
