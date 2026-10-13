"""Rows from hosted decision platforms load unchanged: every label form, both file shapes, single-question rows, weights."""

import json

import pytest

from ruling.evals import Record, load_dataset

QUESTIONS = {
    "ok": {"type": "noul", "instructions": "The ticket is resolved."},
    "team": {"type": "choice", "instructions": "Team?", "criteria": {"billing": None, "technical": None, "sales": None}},
    "urgency": {"type": "score", "instructions": "Urgency?", "criteria": ["low", "mid", "high"]},
}


def record(labels, **extra):
    return Record.model_validate({"state": "s", "questions": QUESTIONS, "labels": labels, **extra})


@pytest.mark.parametrize("qid, label, canonical, target", [
    ("ok", True, True, None),
    ("ok", "true", True, None),
    ("ok", "false", False, None),
    ("ok", "False", False, None),
    ("ok", "yes", True, None),
    ("ok", 0.8, True, {"yes": 0.8, "no": pytest.approx(0.2)}),
    ("ok", 0.2, False, {"yes": 0.2, "no": pytest.approx(0.8)}),
    ("ok", 1, True, {"yes": 1.0, "no": 0.0}),
    ("ok", {"true": 0.3, "false": 0.7}, False, {"yes": 0.3, "no": 0.7}),
    ("ok", {"yes": 0.6, "no": 0.4}, True, {"yes": 0.6, "no": 0.4}),
    ("team", "sales", "sales", None),
    ("team", {"billing": 0.1, "technical": 0.6, "sales": 0.3}, "technical", {"billing": 0.1, "technical": 0.6, "sales": 0.3}),
    ("team", [0.1, 0.2, 0.7], "sales", {"billing": 0.1, "technical": 0.2, "sales": 0.7}),
    ("urgency", 2, 2, None),
    ("urgency", "2", 2, None),
    ("urgency", "mid", "mid", None),
    ("urgency", [0.2, 0.5, 0.3], 1, {"0": 0.2, "1": 0.5, "2": 0.3}),
    ("urgency", {"0": 0.7, "1": 0.2, "2": 0.1}, 0, {"0": 0.7, "1": 0.2, "2": 0.1}),
    ("urgency", {"2": 0.5, "0": 0.25, "1": 0.25}, 2, {"2": 0.5, "0": 0.25, "1": 0.25}),
])
def test_every_hosted_label_form_normalizes_to_a_canonical_label_and_target(qid, label, canonical, target):
    r = record({qid: label})
    assert r.labels[qid] == canonical and type(r.labels[qid]) is type(canonical)
    assert r.references.get(qid, {}).get("target") == target


def test_a_soft_label_ties_go_to_the_first_canonical_key():
    assert record({"ok": 0.5}).labels["ok"] is True
    assert record({"urgency": [0.4, 0.4, 0.2]}).labels["urgency"] == 0


def test_a_soft_label_keeps_a_sum_within_two_percent_and_refuses_beyond():
    record({"team": {"billing": 0.5, "technical": 0.3, "sales": 0.21}})
    with pytest.raises(ValueError, match="'team'.*sum to 1"):
        record({"team": {"billing": 0.5, "technical": 0.3, "sales": 0.25}})


@pytest.mark.parametrize("qid, label, message", [
    ("team", {"billing": 1.0}, "cover exactly"),
    ("team", {"billing": 0.5, "technical": 0.5, "sales": 0.0, "legal": 0.0}, "cover exactly"),
    ("team", [0.5, 0.5], "3 probabilities"),
    ("urgency", {"0": 1.2, "1": -0.2, "2": 0.0}, "non-negative"),
    ("ok", 1.5, r"within \[0, 1\]"),
    ("ok", [0.5, 0.5], "not a list"),
    ("ok", "maybe", "boolean"),
    ("urgency", "7", "out of range"),
    ("team", 0.5, "unknown choice option"),
])
def test_bad_soft_labels_are_refused_naming_the_question(qid, label, message):
    with pytest.raises(ValueError, match=f"'{qid}'.*{message}"):
        record({qid: label})


def test_a_hard_label_keeps_an_existing_target_reference_and_a_soft_label_replaces_it():
    kept = record({"ok": True}, references={"ok": {"target": {"yes": 0.9, "no": 0.1}}})
    assert kept.references["ok"]["target"] == {"yes": 0.9, "no": 0.1}
    replaced = record({"ok": 0.4}, references={"ok": {"target": {"yes": 0.9, "no": 0.1}, "jev": {"yes": 0.5, "no": 0.5}}})
    assert replaced.references["ok"] == {"target": {"yes": 0.4, "no": pytest.approx(0.6)}, "jev": {"yes": 0.5, "no": 0.5}}


def test_id_and_weight_ride_on_the_record():
    r = record({"ok": True}, id="row-7", weight=2.5)
    assert r.id == "row-7" and r.weight == 2.5
    assert record({"ok": True}).weight == 1.0 and record({"ok": True}).id is None
    with pytest.raises(ValueError, match="weight"):
        record({"ok": True}, weight=-0.1)


def test_json_array_and_jsonl_load_the_same_rows(tmp_path):
    rows = [{"id": "a", "state": "s", "questions": QUESTIONS, "labels": {"ok": "true", "urgency": "1"}, "weight": 0.5},
            {"state": "t", "questions": QUESTIONS}]
    as_json, as_jsonl = tmp_path / "d.json", tmp_path / "d.jsonl"
    as_json.write_text(json.dumps(rows))
    as_jsonl.write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert [r.model_dump() for r in load_dataset(as_json)] == [r.model_dump() for r in load_dataset(as_jsonl)]
    first = load_dataset(as_json)[0]
    assert first.labels == {"ok": True, "urgency": 1} and first.weight == 0.5 and first.id == "a"
    as_json.write_text(json.dumps(rows[0]))
    with pytest.raises(ValueError, match="array"):
        load_dataset(as_json)


def test_a_bad_row_is_reported_by_position_and_id(tmp_path):
    path = tmp_path / "d.jsonl"
    path.write_text(json.dumps({"state": "s", "questions": QUESTIONS, "labels": {"ok": True}}) + "\n"
                    + json.dumps({"id": "bad-one", "state": "s", "questions": QUESTIONS, "labels": {"team": [1.0]}}) + "\n")
    with pytest.raises(ValueError, match=r"(?s)row 2 \(bad-one\).*'team'"):
        load_dataset(path)


@pytest.mark.parametrize("row, label, target", [
    ({"question": QUESTIONS["ok"], "label": "true"}, True, None),
    ({"question": QUESTIONS["ok"], "target": 0.9}, True, {"yes": 0.9, "no": pytest.approx(0.1)}),
    ({"question": QUESTIONS["ok"], "label": False, "target": 0.9}, True, {"yes": 0.9, "no": pytest.approx(0.1)}),
    ({"question": QUESTIONS["ok"], "label": False, "target": True}, False, None),
    ({"question": QUESTIONS["urgency"], "expected": "2"}, 2, None),
    ({"question": QUESTIONS["team"], "expected": "sales", "label": "billing"}, "sales", None),
    ({"question": QUESTIONS["urgency"], "target": [0.1, 0.1, 0.8]}, 2, {"0": 0.1, "1": 0.1, "2": 0.8}),
])
def test_single_question_rows_become_one_record_named_decision(tmp_path, row, label, target):
    path = tmp_path / "d.jsonl"
    path.write_text(json.dumps({"id": "x", "state": "s", "weight": 3, **row}) + "\n")
    (r,) = load_dataset(path)
    assert list(r.questions) == ["decision"] and r.id == "x" and r.weight == 3.0
    assert r.labels == {"decision": label}
    assert r.references.get("decision", {}).get("target") == target


def test_an_unlabeled_single_question_row_and_an_unrecognized_row(tmp_path):
    path = tmp_path / "d.jsonl"
    path.write_text(json.dumps({"state": "s", "question": QUESTIONS["ok"]}) + "\n")
    assert load_dataset(path)[0].labels == {}
    path.write_text(json.dumps({"state": "s", "label": True}) + "\n")
    with pytest.raises(ValueError, match="row 1.*'questions' map or a single 'question'"):
        load_dataset(path)
