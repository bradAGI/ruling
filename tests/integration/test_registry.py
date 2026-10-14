"""Adapters on one base share its weights, answer by name, and stay correct under concurrent requests."""

import json
import threading

import mlx.core as mx
import numpy as np
import pytest
from fastapi.testclient import TestClient
from mlx_lm import load
from mlx_lm.tuner.utils import linear_to_lora_layers

from ruling.calibration import Calibration, Provenance
from ruling.config import Settings
from ruling.engine import ADAPTER_CONFIG, ADAPTER_WEIGHTS, Engine, lora_parameters
from ruling.questions import Choice, Noul
from ruling.registry import Registry
from ruling.server import create_app
from tests.integration.conftest import TEST_MODEL

pytestmark = pytest.mark.integration

STATE = {"message": "My card was charged twice for one order. Refund the duplicate."}
QUESTIONS = {
    "team": Choice(instructions="Which team should handle this?",
                   criteria={"billing": "Charges and refunds", "technical": "Bugs and errors", "sales": "Pricing"}),
    "refund": Noul(instructions="The customer asks for a refund."),
}
PAYLOAD = {"state": STATE, "questions": {qid: q.model_dump() for qid, q in QUESTIONS.items()}}


def write_adapter(path, num_layers: int, rank: int, seed: int) -> None:
    """A random LoRA adapter in mlx-lm's layout: fresh `lora_a`, and a `lora_b` that is not zero so it changes answers."""
    model, _ = load(TEST_MODEL)
    config = {"fine_tune_type": "lora", "num_layers": num_layers,
              "lora_parameters": {"rank": rank, "scale": 20.0, "dropout": 0.0}}
    linear_to_lora_layers(model, num_layers, config["lora_parameters"])
    mx.random.seed(seed)
    weights = {key: value if key.endswith("lora_a") else mx.random.normal(value.shape) * 0.02
               for key, value in lora_parameters(model).items()}
    mx.eval(list(weights.values()))
    path.mkdir()
    mx.save_safetensors(str(path / ADAPTER_WEIGHTS), weights)
    (path / ADAPTER_CONFIG).write_text(json.dumps(config))


@pytest.fixture(scope="module")
def adapters(tmp_path_factory):
    root = tmp_path_factory.mktemp("adapters")
    write_adapter(root / "a", num_layers=2, rank=4, seed=0)
    write_adapter(root / "b", num_layers=2, rank=4, seed=1)
    write_adapter(root / "odd", num_layers=1, rank=8, seed=2)
    return root


def logits(engine):
    return engine.score(STATE, QUESTIONS).raw["team"].logits


def test_variants_share_the_model_and_restore_each_others_weights(adapters):
    owner = Engine(TEST_MODEL, Calibration(), 4096, rotations=1, prior_debias=False)
    before = logits(owner)
    tuned = owner.variant(adapters / "a", Calibration())
    plain = owner.variant(None, Calibration())
    assert tuned.model is owner.model and plain.model is owner.model
    assert tuned.adapter is not None and tuned.adapter != owner.adapter and plain.adapter is None
    with_a = logits(tuned)
    assert not np.allclose(with_a, before, atol=1e-3), "the adapter changed nothing"
    assert np.allclose(logits(owner), before, atol=1e-4), "the owner's weights did not come back"
    assert np.allclose(logits(plain), before, atol=1e-4), "a base-only variant must answer like the base"
    second = tuned.variant(adapters / "b", Calibration())
    assert not np.allclose(logits(second), with_a, atol=1e-3)
    assert np.allclose(logits(tuned), with_a, atol=1e-4)
    with pytest.raises(Engine.IncompatibleAdapter):
        owner.variant(adapters / "odd", Calibration())
    fit_for_base = Calibration({"choice": 1.2}, Provenance(owner.model_id, owner.revision, 1, False, None))
    with pytest.raises(ValueError, match="adapter"):
        owner.variant(adapters / "a", fit_for_base)


def test_server_answers_by_name_and_stays_consistent_under_contention(adapters, tmp_path):
    (tmp_path / "models.json").write_text(json.dumps({
        "tuned": {"model": TEST_MODEL, "adapter": str(adapters / "a")},
        "tuned-b": {"model": TEST_MODEL, "adapter": str(adapters / "b")},
    }))
    settings = Settings(model=TEST_MODEL, models_path=tmp_path / "models.json", rotations=1, prior_debias=False,
                        max_input_tokens=4096)
    registry = Registry.from_settings(settings)
    assert registry.engines["tuned"].model is registry.engines[TEST_MODEL].model
    with TestClient(create_app(registry, None, settings.max_pending)) as client:
        listed = {m["name"]: m for m in client.get("/v1/models").json()["models"]}
        assert set(listed) == {TEST_MODEL, "tuned", "tuned-b"}
        assert listed["tuned"]["base"] == TEST_MODEL and listed["tuned"]["adapter"] and listed[TEST_MODEL]["adapter"] is None
        reference = {name: client.post("/v1/systemone", json={**PAYLOAD, "model": name}).json() for name in listed}
        assert reference["tuned"]["model"] == "tuned" and reference[TEST_MODEL]["model"] == TEST_MODEL
        assert reference["tuned"]["answers"] != reference["tuned-b"]["answers"] != reference[TEST_MODEL]["answers"]

        names = sorted(listed)
        results = []

        def ask(i: int) -> None:
            name = names[i % len(names)]
            results.append((name, client.post("/v1/systemone", json={**PAYLOAD, "model": name}).json()))

        threads = [threading.Thread(target=ask, args=(i,)) for i in range(4 * len(names))]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    assert len(results) == 4 * len(names)
    for name, body in results:
        assert body["model"] == reference[name]["model"]
        for qid, expected in reference[name]["answers"].items():
            got = body["answers"][qid]
            if "probabilities" in expected:
                assert np.allclose(list(got["probabilities"].values()), list(expected["probabilities"].values()), atol=1e-5)
            else:
                assert abs(got["noul"] - expected["noul"]) < 1e-5


def test_each_decision_server_entry_is_asked_at_its_own_address(live_server, tmp_path, monkeypatch):
    # Publishers' decision models each run behind their own /v1/systemone server; one ruling
    # process serves them all by name, so an entry's address must win over the global one.
    dead = "http://127.0.0.1:9"
    monkeypatch.setenv("TYPESAFE_BASE_URL", dead)
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    (tmp_path / "models.json").write_text(json.dumps({
        "up": {"model": "typesafe:", "base_url": live_server},
        "down": {"model": "typesafe:", "base_url": dead},
    }))
    registry = Registry.from_settings(Settings(model="typesafe:", models_path=tmp_path / "models.json"))
    with TestClient(create_app(registry, None, 4)) as client:
        up = client.post("/v1/systemone", json={**PAYLOAD, "model": "up"})
        assert up.status_code == 200 and up.json()["answers"]["team"]["choice"] == "billing"
        down = client.post("/v1/systemone", json={**PAYLOAD, "model": "down"})
        assert down.status_code == 503 and down.json()["error"]["code"] == "upstream_unavailable"
