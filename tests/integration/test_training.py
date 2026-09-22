import json
import os
from pathlib import Path

import numpy as np
import pytest

from ruling import states
from ruling.build import write
from ruling.calibration import Calibration, Provenance
from ruling.engine import Engine
from ruling.train import Trainer, TrainConfig
from ruling.worlds import Catalog, synthesize

# Trains a model in a fixture: two minutes with a GPU, hours without one.
pytestmark = [pytest.mark.integration, pytest.mark.heavy]

TRAIN_MODEL = os.environ.get("RULING_TRAIN_TEST_MODEL", "mlx-community/Qwen3-4B-Instruct-2507-4bit")


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    """A short run on access-policy states, a task with exact labels the untrained 4B model gets a third wrong.

    Models under about 1B parameters do not learn this task at any learning rate, so the test trains the
    real base model.
    """
    root = tmp_path_factory.mktemp("training")
    data = root / "access.jsonl"
    records = [r for r in states.generate(3200, seed=5) if r.family == "state_access"]
    write([r.model_dump(mode="json") for r in records], data)
    config = TrainConfig(model=TRAIN_MODEL, data=[data], output=root / "adapter", holdout=[], valid_fraction=0.15,
                         steps=120, batch_size=4, warmup=10, max_tokens=512, eval_every=60, valid_examples=160, seed=0)
    history = Trainer(config).run(log=lambda _: None)
    return config, history


def test_training_improves_held_out_accuracy_and_likelihood(trained):
    _, history = trained
    first, last = history[0], history[-1]
    assert last["loss"] < first["loss"] - 0.2
    assert last["accuracy"] > first["accuracy"] + 0.05


def test_adapter_loads_into_the_engine_with_its_own_identity(trained):
    config, _ = trained
    meta = json.loads((config.output / "training.json").read_text())
    assert meta["train_examples"] > 400 and meta["config"]["data"][0]["sha256"]
    adapted = Engine(config.model, Calibration(), 4096, rotations=1, prior_debias=False, adapter=config.output)
    assert adapted.adapter is not None and adapted.provenance.adapter == adapted.adapter
    held_out = [r for r in states.generate(400, seed=77) if r.family == "state_access"][:20]
    base = Engine(config.model, Calibration(), 4096, rotations=1, prior_debias=False)
    for record in held_out:
        tuned = adapted.score(record.state, record.questions).raw["decision"].logits
        plain = base.score(record.state, record.questions).raw["decision"].logits
        if not np.allclose(tuned, plain, atol=1e-3):
            break
    else:
        pytest.fail("the adapter did not change any answer distribution")
    fit_on_base = Calibration({"choice": 1.0}, provenance=Provenance(adapted.model_id, adapted.revision, 1, False, None))
    with pytest.raises(ValueError, match="adapter"):
        fit_on_base.check(adapted.provenance)


def test_world_synthesis_writes_labeled_records(tmp_path, engine):
    catalog = Catalog.load(Path(__file__).resolve().parents[2] / "datasets" / "worlds.json")
    records = [r for batch in synthesize(catalog, engine, count=4, seed=3, batch_size=2, min_probability=0.0, temperature=0.8,
                                         log=lambda _: None) for r in batch]
    assert len(records) >= 3
    for record in records:
        assert record.family.startswith("world_") and len(record.state) > 20
        assert set(record.labels) == set(record.questions)


def test_distillation_writes_questions_then_answers_them_with_soft_targets(tmp_path):
    from ruling.cli import main
    from ruling.evals import load_dataset
    from ruling.train import target_vector

    source = tmp_path / "states.jsonl"
    write([{"state": text, "questions": {"q": {"type": "noul", "instructions": "It is a message."}}} for text in (
        "Hi, my order #4471 arrived broken and I need a replacement before Friday.",
        "Reminder: the quarterly budget review moved to 3pm Thursday; bring the vendor quotes.",
    )], source)
    questions, answers = tmp_path / "questions.jsonl", tmp_path / "answers.jsonl"
    main(["build-corpus", "questions", "--states", str(source), "--output", str(questions), "--count", "2",
          "--writer", TRAIN_MODEL, "--batch-size", "2", "--holdout"])
    written = load_dataset(questions)
    assert written, "the writer produced no parsable questions for either state"
    assert all(r.family == "distill_states" and len(r.questions) >= 2 and not r.references for r in written)
    main(["build-corpus", "answers", "--questions", str(questions), "--output", str(answers), "--writer", TRAIN_MODEL,
          "--rotations", "1"])
    for record in load_dataset(answers):
        for qid in record.questions:
            assert abs(sum(target_vector(record, qid).values()) - 1.0) < 1e-6


def test_hosted_teacher_answers_resume_where_they_stopped(tmp_path, monkeypatch, live_server):
    from ruling.cli import main
    from ruling.evals import load_dataset

    questions, answers = tmp_path / "questions.jsonl", tmp_path / "answers.jsonl"
    write([{"state": f"Message {i}: please refund order {i}.", "questions": {
        "refund": {"type": "noul", "instructions": "The writer asks for a refund."},
        "tone": {"type": "score", "instructions": "How upset?", "criteria": ["calm", "upset"]}}} for i in range(3)], questions)
    write([load_dataset(questions)[0].model_dump(mode="json")], answers)
    monkeypatch.setenv("TYPESAFE_BASE_URL", live_server)
    main(["build-corpus", "answers", "--questions", str(questions), "--output", str(answers), "--teacher", "typesafe"])
    records = load_dataset(answers)
    assert len(records) == 3 and not records[0].references
    for record in records[1:]:
        assert set(record.references) == {"refund", "tone"}
        assert abs(sum(record.references["tone"]["target"].values()) - 1.0) < 1e-6
    main(["build-corpus", "answers", "--questions", str(questions), "--output", str(answers), "--teacher", "typesafe"])
    assert len(load_dataset(answers)) == 3
