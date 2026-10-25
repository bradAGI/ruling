import json
import shutil
import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from ruling.evals import Record, as_record, collect, load_dataset, report
from ruling.ledger import DecisionLedger
from ruling.questions import Choice, Noul
from ruling.sdk import RulingClient, RulingError
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
                                "long_context_tokens": engine.long_context_tokens,
                                "max_options": engine.max_options, "rotations": engine.rotations,
                                "prior_debias": engine.prior_debias, "modalities": ["text"],
                                "question_types": ["choice", "score", "noul"]}]}
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
    with RulingClient(base_url=live_server_with_key) as client, pytest.raises(RulingError) as refused:
        client.models()
    assert (refused.value.status, refused.value.code) == (401, "unauthorized")


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


def test_typescript_sdk_reports_outcomes_and_opts_out_of_the_ledger(engine, tmp_path):
    ledger = tmp_path / "ledger.sqlite"
    server, thread, url = _serve(create_app(Registry.single(engine), None, 4, DecisionLedger(ledger)))
    script = f"""
import {{ RulingClient, noul }} from {json.dumps(TS_SDK.as_uri())};
const client = new RulingClient({{ baseUrl: {json.dumps(url)} }});
const questions = {{ refund: noul("The customer asks for a refund.") }};
const logged = await client.systemOne({{ state: {json.dumps(PAYLOAD["state"])}, questions }});
const stored = await client.outcome(logged.id, {{ refund: true }});
const marked = await client.outcome(logged.id, undefined, {{ refund: {{ correct: false }} }});
const batch = await client.outcomes([{{ id: logged.id, labels: {{ refund: true }} }}]);
const excluded = await client.systemOne({{ state: "A benchmark row.", questions, excludeFromTraining: true }});
console.log(JSON.stringify({{ stored, marked, batch, excludedId: excluded.id ?? null, logged: logged.id,
                              served: logged.answers.refund.noul >= 0.5 }}));
"""
    try:
        result = subprocess.run(["node", "--input-type=module", "-e", script], capture_output=True, text=True,
                                timeout=120)
    finally:
        server.should_exit = True
        thread.join(timeout=10)
    assert result.returncode == 0, result.stderr
    out = json.loads(result.stdout.strip().splitlines()[-1])
    assert out["stored"] == {"refund": True} and out["excludedId"] is None
    assert out["batch"] == [{"id": out["logged"], "labels": {"refund": True}}]
    assert out["marked"] == {"refund": not out["served"]}
    assert [r.labels for r in DecisionLedger(ledger).records()] == [{"refund": True}]


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
            with pytest.raises(RulingError) as wrong:
                client.outcome(decision, {"refund": "maybe"})
            assert (wrong.value.code, wrong.value.param) == ("invalid_input", "labels")
            with pytest.raises(RulingError) as missing:
                client.outcome("no-such-id", {"refund": True})
            assert missing.value.code == "unknown_decision"
    finally:
        server.should_exit = True
        thread.join(timeout=10)
    exported = tmp_path / "train.jsonl"
    subprocess.run(["uv", "run", "ruling", "export-decisions", "--ledger", str(ledger), "--output", str(exported)],
                   check=True)
    [record] = load_dataset(exported)
    assert record.state == PAYLOAD["state"] and record.labels == {"refund": True}


def test_right_and_wrong_marks_and_batches_of_outcomes_become_labels(engine, tmp_path):
    ledger = tmp_path / "ledger.sqlite"
    server, thread, url = _serve(create_app(Registry.single(engine), None, 4, DecisionLedger(ledger)))
    refund = {"refund": Noul(instructions="The customer asks for a refund.")}
    try:
        with RulingClient(base_url=url) as client:
            first, second = (client.system_one(state=PAYLOAD["state"], questions=refund) for _ in range(2))
            served = first.answers["refund"].noul >= 0.5
            assert client.outcome(first.id, corrections={"refund": {"correct": False}}) == {"refund": not served}
            with pytest.raises(RulingError) as refused:
                client.outcomes([{"id": first.id, "corrections": {"refund": {"correct": True}}},
                                 {"id": second.id, "labels": {"refund": "maybe"}}])
            assert (refused.value.code, refused.value.param) == ("invalid_input", "outcomes.1.labels")
            assert client.outcomes([{"id": first.id, "corrections": {"refund": {"correct": True}}},
                                    {"id": second.id, "labels": {"refund": False}}]) == \
                [{"id": first.id, "labels": {"refund": served}}, {"id": second.id, "labels": {"refund": False}}]
    finally:
        server.should_exit = True
        thread.join(timeout=10)
    assert {r.id: r.labels for r in DecisionLedger(ledger).records()} == \
        {first.id: {"refund": served}, second.id: {"refund": False}}


def test_a_decision_kept_out_of_training_is_answered_but_not_logged(engine, tmp_path):
    # Clients send this field to keep benchmark and evaluation traffic out of training.
    app = create_app(Registry.single(engine), None, 4, DecisionLedger(tmp_path / "ledger.sqlite"))
    with TestClient(app) as client:
        excluded = client.post("/v1/systemone", json={**PAYLOAD, "exclude_from_training": True}).json()
        logged = client.post("/v1/systemone", json=PAYLOAD).json()
    assert excluded["answers"] and "id" not in excluded
    assert logged["id"]


WORDY = {"id": "wordy", "state": " ".join(f"w{i}" for i in range(40000)),
         "questions": {"refund": PAYLOAD["questions"]["refund"]}, "labels": {"refund": True}}


def test_python_sdk_answers_a_batch_with_a_row_too_long_for_the_model_kept_in_place(live_server, engine):
    rows = [{"id": "dup", **PAYLOAD, "labels": {"team": "billing", "refund": True}},
            WORDY,
            {"id": "one", "state": "Thanks, that fixed it.", "question": PAYLOAD["questions"]["refund"], "label": False}]
    with RulingClient(base_url=live_server) as client:
        answered = client.predict(rows)
        usage = client.usage()
    assert [row["id"] for row in answered] == ["dup", "wordy", "one"]
    assert answered[0]["answers"]["team"].choice == "billing" and answered[0]["answers"]["refund"].noul > 0.5
    assert answered[1]["error"]["code"] == "context_too_long" and "answers" not in answered[1]
    assert answered[2]["answers"]["decision"].noul < answered[0]["answers"]["refund"].noul
    assert sum(b["decisions"] for b in usage if b["model"] == engine.model_id) >= 2


def test_python_sdk_evaluates_labeled_rows_against_the_live_model(live_server, engine):
    rows = [{**PAYLOAD, "labels": {"team": "billing", "refund": True}},
            WORDY,
            {"state": "Thanks, that fixed it.", "question": PAYLOAD["questions"]["refund"], "label": False}]
    with RulingClient(base_url=live_server) as client:
        scored = client.evaluate(rows)
        with pytest.raises(RulingError) as unlabeled:
            client.evaluate([{k: PAYLOAD[k] for k in ("state", "questions")}])
    assert scored.pop("model") == engine.model_id
    # The same rows scored in-process, as `ruling eval` would: the route must report exactly that, wall time aside.
    local = report(collect(engine, [Record.model_validate(as_record(row)) for row in rows]), engine.calibration)
    assert scored.pop("latency_ms")["records"] == local.pop("latency_ms")["records"] == 2
    assert scored == local
    assert scored["overall"]["n"] == 3 and set(scored["by_question"]) == {"team", "refund", "decision"}
    assert scored["question_types"]["noul"]["unscored"] == 1 and scored["question_types"]["noul"]["count"] == 2
    assert (unlabeled.value.status, unlabeled.value.code, unlabeled.value.param) == (422, "invalid_input", "rows")


def test_without_a_ledger_the_wire_format_is_unchanged(app):
    # Clients built on the hosted SDK decode these bodies; nothing may appear that the hosted API does not send.
    with TestClient(app) as client:
        assert set(client.post("/v1/systemone", json=PAYLOAD).json()) == {"model", "answers", "usage"}
        assert client.post("/v1/decisions/x/outcome", json={"labels": {}}).status_code == 404


def test_drift_is_served_and_printed_from_the_ledger(engine, tmp_path, capsys):
    from ruling.cli import main

    ledger = tmp_path / "ledger.sqlite"
    app = create_app(Registry.single(engine), None, 4, DecisionLedger(ledger))
    with TestClient(app) as client:
        for state in ["I was charged twice; refund it."] * 3 + ["Love the product, no issues."] * 3:
            client.post("/v1/systemone", json={**PAYLOAD, "state": state})
        served = client.get("/v1/drift", params={"window": 3}).json()["drift"]
        assert client.get("/v1/drift", params={"window": 0}).status_code == 422
    assert {(d["model"], d["question"]) for d in served} == {(engine.model_id, q) for q in PAYLOAD["questions"]}
    main(["drift", "--ledger", str(ledger), "--window", "3"])
    assert json.loads(capsys.readouterr().out)["drift"] == served
