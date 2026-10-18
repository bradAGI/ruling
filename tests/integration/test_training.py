import json
import os
import sys
from pathlib import Path

import mlx.core as mx
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


@pytest.fixture(scope="module")
def small_data(tmp_path_factory):
    data = tmp_path_factory.mktemp("resume") / "states.jsonl"
    write([r.model_dump(mode="json") for r in states.generate(60, seed=5)], data)
    return data


def _small_run(model, data, output, **overrides):
    """A few steps on the small test model: the same settings as a TrainConfig and as `ruling train` arguments."""
    settings = {"steps": 6, "batch_size": 2, "warmup": 2, "max_tokens": 256, "eval_every": 3, "valid_examples": 8,
                "valid_fraction": 0.2, "lora_layers": 2, "seed": 0, **overrides}
    args = ["train", "--data", str(data), "--output", str(output), "--holdout", "--model", model]
    for name, value in settings.items():
        args += [f"--{name.replace('_', '-')}", str(value)]
    return TrainConfig(model=model, data=[data], output=output, holdout=[], **settings), args


def test_resumed_run_matches_an_uninterrupted_one_bit_for_bit(small_data, test_model, tmp_path):
    from ruling.cli import main

    whole, _ = _small_run(test_model, small_data, tmp_path / "whole")
    expected = Trainer(whole).run(log=lambda _: None)
    _, first_half = _small_run(test_model, small_data, tmp_path / "split", steps=3)
    main(first_half)
    assert json.loads((tmp_path / "split" / "checkpoint" / "state.json").read_text())["step"] == 3
    _, second_half = _small_run(test_model, small_data, tmp_path / "split")
    with pytest.raises(ValueError, match="different learning_rate"):
        main(second_half + ["--learning-rate", "2e-5", "--resume"])
    main(second_half + ["--resume"])

    resumed = json.loads((tmp_path / "split" / "checkpoint" / "state.json").read_text())
    assert [entry["step"] for entry in resumed["history"]] == [0, 3, 6]
    timing = {key: resumed["history"][-1][key] for key in ("tokens_per_second", "elapsed_seconds")}
    assert resumed["history"][-1] == expected[-1] | timing
    uninterrupted = json.loads((tmp_path / "whole" / "checkpoint" / "state.json").read_text())
    assert resumed["tokens_seen"] == uninterrupted["tokens_seen"] and resumed["stream"] == uninterrupted["stream"]
    for name in ("adapters", "optimizer"):
        theirs = mx.load(str(tmp_path / "whole" / "checkpoint" / f"{name}.safetensors"))
        ours = mx.load(str(tmp_path / "split" / "checkpoint" / f"{name}.safetensors"))
        assert set(ours) == set(theirs)
        for key in theirs:
            assert np.array_equal(np.array(ours[key]), np.array(theirs[key])), key


def test_from_adapter_starts_where_the_source_adapter_stands(small_data, test_model, tmp_path):
    source, _ = _small_run(test_model, small_data, tmp_path / "source", steps=3)
    best = min(Trainer(source).run(log=lambda _: None), key=lambda entry: entry["loss"])
    seeded, _ = _small_run(test_model, small_data, tmp_path / "seeded", steps=3, from_adapter=tmp_path / "source")
    history = Trainer(seeded).run(log=lambda _: None)
    assert history[0]["loss"] == best["loss"] and history[0]["accuracy"] == best["accuracy"]
    assert json.loads((tmp_path / "seeded" / "training.json").read_text())["config"]["from_adapter"] == str(tmp_path / "source")
    mismatched, _ = _small_run(test_model, small_data, tmp_path / "mismatched", lora_rank=4, from_adapter=tmp_path / "source")
    with pytest.raises(ValueError, match="not the requested"):
        Trainer(mismatched)


@pytest.fixture(scope="module")
def watched_run(small_data, test_model, tmp_path_factory):
    """A short `ruling train --wandb` run with the package stood in for: the history it wrote and what reached the sink."""
    from ruling.cli import main
    from ruling.train import TRAINING
    from tests.unit.test_train import FakeWandb

    output = tmp_path_factory.mktemp("watched") / "adapter"
    _, args = _small_run(test_model, small_data, output)
    fake = FakeWandb()
    with pytest.MonkeyPatch.context() as patch:
        patch.setitem(sys.modules, "wandb", fake)
        main(args + ["--wandb", "curves"])
    return json.loads((output / TRAINING).read_text())["history"], fake


def test_every_progress_line_says_how_far_the_run_is_and_when_it_ends(watched_run):
    history, _ = watched_run
    assert history[0]["step"] == 0 and "progress" not in history[0]
    stepped = history[1:]
    assert [e["progress"] for e in stepped] == sorted(e["progress"] for e in stepped)
    assert [e["elapsed_seconds"] for e in stepped] == sorted(e["elapsed_seconds"] for e in stepped)
    assert all(e["elapsed_seconds"] > 0 and e["eta_seconds"] >= 0 for e in stepped)
    assert stepped[-1]["progress"] == 1.0 and stepped[-1]["eta_seconds"] == 0.0


def test_wandb_receives_every_history_entry_as_the_run_makes_it(watched_run):
    history, fake = watched_run
    assert fake.init_calls[0]["project"] == "curves" and fake.init_calls[0]["config"]["steps"] == 6
    assert fake.logged == [(entry["step"], entry) for entry in history]
    assert fake.finished == [0]


def test_a_run_whose_examples_all_exceed_max_tokens_refuses_to_start(small_data, test_model, tmp_path):
    """Before this was counted, such a run trained on nothing and reported NaN validation without an error."""
    config, _ = _small_run(test_model, small_data, tmp_path / "short", max_tokens=8)
    with pytest.raises(ValueError, match="longer than max_tokens 8"):
        Trainer(config)
