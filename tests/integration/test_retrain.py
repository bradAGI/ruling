"""The loop end to end: decisions served with a ledger, outcomes reported, then retraining that
promotes a new adapter only when it beats the served one on held-out data."""

import json
from dataclasses import replace

import pytest
from fastapi.testclient import TestClient

from ruling import states
from ruling.build import write
from ruling.calibration import Calibration
from ruling.cli import main
from ruling.config import Settings
from ruling.ledger import DecisionLedger
from ruling.registry import Registry, build_engine
from ruling.retrain import CALIBRATION, CURRENT, PREVIEW, REPORT, history, should_promote
from ruling.server import create_app

from .conftest import TEST_MODEL

pytestmark = [pytest.mark.integration, pytest.mark.heavy]


def test_served_outcomes_train_a_version_that_is_promoted_only_if_better(engine, tmp_path, monkeypatch):
    ledger_path, adapters, holdout = tmp_path / "ledger.sqlite", tmp_path / "adapters", tmp_path / "holdout.jsonl"
    served = [r for r in states.generate(400, seed=21) if r.family == "state_access"][:40]
    write([r.model_dump(mode="json") for r in states.generate(400, seed=22) if r.family == "state_access"][:12],
          holdout)
    with TestClient(create_app(Registry.single(engine), None, 4, DecisionLedger(ledger_path))) as client:
        for record in served:
            body = {"state": record.state,
                    "questions": {qid: q.model_dump(exclude_none=True) for qid, q in record.questions.items()}}
            decision = client.post("/v1/systemone", json=body).json()["id"]
            assert client.post(f"/v1/decisions/{decision}/outcome", json={"labels": record.labels}).status_code == 200
        # A benchmark row sent through the server never becomes training data.
        client.post("/v1/systemone", json={**body, "exclude_from_training": True})

    monkeypatch.setenv("RULING_ROTATIONS", "1")
    command = ["retrain", "--ledger", str(ledger_path), "--adapters", str(adapters), "--holdout", str(holdout),
               "--model", TEST_MODEL, "--steps", "4", "--warmup", "1", "--eval-every", "2", "--batch-size", "4",
               "--valid-fraction", "0.2"]
    main([*command, "--min-records", str(len(served) + 1)])
    assert not adapters.exists()

    for run, version in enumerate(("v1", "v2")):
        before = (adapters / CURRENT).resolve().name if (adapters / CURRENT).exists() else None
        main([*command, "--min-records", "20"])
        report = json.loads((adapters / version / REPORT).read_text())
        assert report["records"] == len(served) and report["started_from"] == before
        assert (adapters / version / "adapters.safetensors").exists()
        assert report["promoted"] == should_promote(report["current"], report["candidate"], report["worse"])[0]
        now = (adapters / CURRENT).resolve().name if (adapters / CURRENT).exists() else None
        assert now == (version if report["promoted"] else before)
        # Both versions were judged at their own fitted temperatures, and the candidate's ship with it:
        # a server started on this version with its calibration accepts the pairing.
        calibration = Calibration.load(adapters / version / CALIBRATION)
        assert calibration.temperatures == report["candidate"]["temperatures"] and calibration.temperatures
        assert report["current"]["temperatures"]
        serving = build_engine(replace(Settings.from_env(), model=TEST_MODEL, adapter_path=adapters / version,
                                       calibration_path=adapters / version / CALIBRATION))
        assert serving.calibration.temperatures == calibration.temperatures
        # The history page reads each version's own training log, finished at the last step.
        assert history(adapters)["versions"][run]["training"][-1]["progress"] == 1.0
        assert (adapters / PREVIEW).resolve().name == version

    # Held back, a third version is only the preview; the server answers it by name beside the served one.
    before = (adapters / CURRENT).resolve().name if (adapters / CURRENT).exists() else None
    main([*command, "--min-records", "20", "--no-promote"])
    report = json.loads((adapters / "v3" / REPORT).read_text())
    assert not report["promoted"] and (adapters / PREVIEW).resolve().name == "v3"
    assert ((adapters / CURRENT).resolve().name if (adapters / CURRENT).exists() else None) == before
    main(["promote", "--adapters", str(adapters), "v1"])
    registry = Registry.from_settings(replace(Settings.from_env(), model=TEST_MODEL, adapter_path=adapters / CURRENT))
    preview = registry.resolve(f"{registry.default}:preview")[1]
    assert preview.adapter_path.resolve() == (adapters / "v3").resolve()
    with TestClient(create_app(registry, None, 4)) as client:
        asked = {"state": served[0].state, "model": f"{registry.default}:preview",
                 "questions": {qid: q.model_dump(exclude_none=True) for qid, q in served[0].questions.items()}}
        assert client.post("/v1/systemone", json=asked).status_code == 200
