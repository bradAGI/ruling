import json

import numpy as np
import pytest

from ruling.evals import Record
from ruling.train import holdout_overlap, split_records, target_vector, permuted


def record(**extra):
    base = {"state": "The sky is clear.", "questions": {
        "color": {"type": "choice", "instructions": "Color?", "criteria": {"red": None, "blue": None, "green": None}},
        "rain": {"type": "noul", "instructions": "It will rain."},
        "clouds": {"type": "score", "instructions": "Cloud cover?", "criteria": ["none", "some", "full"]},
    }}
    return Record.model_validate({**base, **extra})


def test_target_prefers_soft_reference_over_hard_label():
    r = record(labels={"color": "blue", "rain": False}, references={"rain": {"target": {"yes": 0.3, "no": 0.7}}})
    assert target_vector(r, "color") == {"red": 0.0, "blue": 1.0, "green": 0.0}
    assert target_vector(r, "rain") == {"yes": 0.3, "no": 0.7}
    assert target_vector(r, "clouds") is None


def test_permutation_keeps_every_key_and_maps_targets_by_key():
    r = record(labels={"color": "blue", "rain": True, "clouds": 2})
    rng = np.random.default_rng(0)
    seen_orders = set()
    for _ in range(40):
        question, rotation = permuted(r.questions["color"], rng)
        assert rotation == 0 and sorted(question.criteria) == ["blue", "green", "red"]
        seen_orders.add(tuple(question.criteria))
        for qid in ("rain", "clouds"):
            same, rotation = permuted(r.questions[qid], rng)
            assert same == r.questions[qid] and rotation in (0, 1)
    assert len(seen_orders) == 6


def test_split_is_deterministic_disjoint_and_close_to_fraction():
    records = [record(labels={"color": "red"}).model_copy(update={"state": f"state {i}"}) for i in range(2000)]
    train_a, valid_a = split_records(records, 0.05)
    train_b, valid_b = split_records(list(reversed(records)), 0.05)
    assert {r.state for r in valid_a} == {r.state for r in valid_b}
    assert not {r.state for r in train_a} & {r.state for r in valid_a}
    assert 60 <= len(valid_a) <= 140


def test_holdout_overlap_detects_shared_states_after_normalizing_whitespace():
    training = [record(labels={"color": "red"}), record(labels={"color": "red"}).model_copy(update={"state": {"k": "v"}})]
    held = [record().model_copy(update={"state": "The  sky is\nclear."})]
    assert holdout_overlap(training, held) == 1
    assert holdout_overlap(training[1:], held) == 0


def test_trainer_refuses_data_that_overlaps_a_holdout_before_loading_a_model(tmp_path):
    from pathlib import Path

    from ruling.build import write
    from ruling.train import Trainer, TrainConfig

    authored = Path(__file__).resolve().parents[2] / "datasets" / "authored144.jsonl"
    leaked = json.loads(authored.read_text().splitlines()[5])
    data = tmp_path / "leaky.jsonl"
    write([leaked], data)
    config = TrainConfig(model="not-a-model-because-it-must-never-load", data=[data], output=tmp_path / "out", holdout=[authored])
    with pytest.raises(ValueError, match="also appear in held-out"):
        Trainer(config)
