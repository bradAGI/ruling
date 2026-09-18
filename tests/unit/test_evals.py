import json

import pytest

from ruling.evals import balanced_accuracy, label_index, load_dataset
from ruling.questions import Choice, Noul, Score


def test_label_index_per_type():
    assert label_index(Choice(instructions="q", criteria={"a": None, "b": None}), "b") == 1
    assert label_index(Score(instructions="q", criteria=["lo", "hi"]), "hi") == 1
    assert label_index(Score(instructions="q", criteria=["lo", "hi"]), 0) == 0
    assert label_index(Noul(instructions="q"), True) == 0
    assert label_index(Noul(instructions="q"), False) == 1


def test_label_index_rejects_bad_labels():
    with pytest.raises(ValueError):
        label_index(Choice(instructions="q", criteria={"a": None, "b": None}), "c")
    with pytest.raises(ValueError):
        label_index(Score(instructions="q", criteria=["lo", "hi"]), 2)
    with pytest.raises(ValueError):
        label_index(Noul(instructions="q"), "yes")


def test_balanced_accuracy_weights_classes_equally():
    gold = [0] * 9 + [1]
    assert balanced_accuracy([0] * 10, gold) == pytest.approx(0.5)
    assert balanced_accuracy(gold, gold) == pytest.approx(1.0)


def test_load_dataset_accepts_unlabeled_records_and_rejects_stray_labels(tmp_path):
    path = tmp_path / "d.jsonl"
    record = {"state": "s", "questions": {"q": {"type": "noul", "instructions": "i"}}}
    path.write_text(json.dumps(record) + "\n")
    assert load_dataset(path)[0].labels == {}
    record["labels"] = {"other": True}
    path.write_text(json.dumps(record) + "\n")
    with pytest.raises(ValueError, match="unknown questions"):
        load_dataset(path)


def test_default_holdouts_are_all_loadable_record_datasets():
    from ruling.cli import bundled_evaluations

    paths = bundled_evaluations()
    assert {p.name for p in paths} >= {"authored144.jsonl", "perturbations108.jsonl", "support_tickets.jsonl"}
    for path in paths:
        assert load_dataset(path)
