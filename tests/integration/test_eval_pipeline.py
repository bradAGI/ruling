import json
from pathlib import Path

import pytest

from ruling.calibration import Calibration
from ruling.cli import main
from ruling.evals import collect, fit, load_dataset, report

pytestmark = pytest.mark.integration

DATASET = Path(__file__).resolve().parents[2] / "datasets" / "support_tickets.jsonl"


def test_report_covers_every_primitive(engine):
    records = load_dataset(DATASET)[:4]
    observations = collect(engine, records)
    assert len(observations) == 12
    result = report(observations, Calibration())["question_types"]
    assert set(result) == {"choice", "score", "noul"}
    for summary in result.values():
        assert summary["count"] == 4
        assert 0.0 <= summary["accuracy"] <= 1.0
        assert 0.0 <= summary["expected_calibration_error"] <= 1.0
    assert result["choice"]["option_reversal_flips"] <= 4
    assert "mean_family_balanced_accuracy" in result["choice"]


def test_fitted_temperatures_do_not_worsen_likelihood(engine):
    observations = collect(engine, load_dataset(DATASET))
    calibration = fit(observations, engine.provenance)
    before = report(observations, Calibration())["question_types"]
    after = report(observations, calibration)["question_types"]
    for question_type, temperature in calibration.temperatures.items():
        assert temperature > 0
        assert after[question_type]["negative_log_likelihood"] <= before[question_type]["negative_log_likelihood"] + 1e-9


def test_cli_calibrate_writes_a_file_bound_to_the_engine(tmp_path, monkeypatch, engine):
    monkeypatch.setenv("RULING_MODEL", engine.model_id)
    output = tmp_path / "calibration.json"
    main(["calibrate", str(DATASET), "--output", str(output), "--rotations", str(engine.rotations)])
    data = json.loads(output.read_text())
    assert set(data["temperatures"]) == {"choice", "score", "noul"}
    assert data["provenance"] == {"model": engine.model_id, "revision": engine.revision,
                                  "rotations": engine.rotations, "prior_debias": engine.prior_debias, "adapter": None}
    Calibration.load(output).check(engine.provenance)
    with pytest.raises(ValueError, match="rotations"):
        main(["eval", str(DATASET), "--calibration", str(output), "--rotations", str(engine.rotations + 1)])


def test_cli_ask_prints_typed_answers(capsys, monkeypatch, engine):
    monkeypatch.setenv("RULING_MODEL", engine.model_id)
    main(["ask", "The sky is clear and blue.",
          "--choice", "color=What color is the sky?|red,blue,green",
          "--score", "clouds=How cloudy is it?|clear,partly cloudy,overcast",
          "--noul", "blue=The sky is described as blue."])
    data = json.loads(capsys.readouterr().out)
    assert data["answers"]["color"]["choice"] == "blue"
    assert data["answers"]["clouds"]["score"] < 1.0
    assert data["answers"]["blue"]["noul"] > 0.5


def test_inputs_beyond_the_engine_limit_are_reported_not_fatal(engine):
    from ruling.evals import Record

    too_long = Record.model_validate({"state": "word " * 20000,
                                      "questions": {"q": {"type": "noul", "instructions": "It mentions a word."}},
                                      "labels": {"q": True}})
    records = load_dataset(DATASET)[:1] + [too_long]
    summary = report(collect(engine, records), Calibration())["question_types"]["noul"]
    assert summary["unscored"] == 1 and summary["count"] == 1
