import numpy as np
import pytest
from pydantic import ValidationError

from ruling.calibration import Calibration
from ruling.evals import Observation, Record, answer_distribution, report
from ruling.prompt import render_question
from ruling.questions import Noul, NoulCriteria


def test_answer_distribution_covers_every_primitive():
    assert answer_distribution({"type": "noul", "noul": 0.9}, ["yes", "no"]) == {"yes": 0.9, "no": pytest.approx(0.1)}
    choice = {"type": "choice", "choice": "b", "confidence": 0.5, "probabilities": {"b": 0.7, "a": 0.3}}
    assert answer_distribution(choice, ["a", "b"]) == {"a": 0.3, "b": 0.7}
    score = {"type": "score", "score": 1.2, "legend": {"0": "lo", "1": "hi"}, "probabilities": {"0": 0.2, "1": 0.8}}
    assert answer_distribution(score, ["0", "1"]) == {"0": 0.2, "1": 0.8}
    with pytest.raises(ValueError, match="lacks options"):
        answer_distribution(choice, ["a", "b", "c"])


def test_record_allows_missing_labels_but_checks_reference_keys():
    base = {"state": "s", "questions": {"q": {"type": "choice", "instructions": "i", "criteria": {"a": None, "b": None}}}}
    Record.model_validate(base)
    Record.model_validate({**base, "references": {"q": {"jev": {"a": 0.6, "b": 0.4}}}})
    with pytest.raises(ValidationError, match="cover exactly"):
        Record.model_validate({**base, "references": {"q": {"jev": {"a": 1.0}}}})
    with pytest.raises(ValidationError, match="unknown questions"):
        Record.model_validate({**base, "labels": {"other": "a"}})


def test_report_measures_agreement_with_references():
    def obs(i, logits, label, jev):
        return Observation(record=i, question_id="q", question_type="choice", family=None, keys=["a", "b"],
                           logits=np.array(logits), label=label, references={"jev": np.array(jev)}, reversed_argmax=None)
    observations = [
        obs(0, [2.0, 0.0], 0, [0.9, 0.1]),   # both right, agree
        obs(1, [0.0, 2.0], 0, [0.8, 0.2]),   # ruling wrong, jev right, disagree
        obs(2, [2.0, 0.0], None, [0.4, 0.6]),  # unlabeled, disagree
    ]
    summary = report(observations, Calibration())["question_types"]["choice"]
    assert summary["count"] == 3 and summary["labeled_count"] == 2
    assert summary["accuracy"] == pytest.approx(0.5)
    jev = summary["references"]["jev"]
    assert jev["count"] == 3
    assert jev["argmax_agreement"] == pytest.approx(1 / 3)
    assert jev["reference_accuracy"] == pytest.approx(1.0)
    assert jev["ruling_accuracy_on_same_rows"] == pytest.approx(0.5)
    assert 0.0 <= jev["mean_total_variation"] <= 1.0


def test_noul_criteria_render_as_option_descriptions():
    question = Noul(instructions="Is it unauthorized?", criteria=NoulCriteria(true="Someone broke in", false="Sanctioned work"))
    text = render_question(question, list("ABC")).text
    assert "A. yes: Someone broke in" in text and "B. no: Sanctioned work" in text
    assert "yes:" not in render_question(Noul(instructions="q"), list("ABC")).text


def test_report_counts_unscorable_questions_without_scoring_them():
    labeled = Observation(record=0, question_id="q", question_type="choice", family=None, keys=["a", "b"],
                          logits=np.array([2.0, 0.0]), label=0, references={"jev": np.array([0.9, 0.1])}, reversed_argmax=0)
    too_long = Observation(record=1, question_id="q", question_type="choice", family=None, keys=["a", "b"],
                           logits=None, label=1, references={"jev": np.array([0.2, 0.8])}, reversed_argmax=None)
    summary = report([labeled, too_long], Calibration())["question_types"]["choice"]
    assert summary["count"] == 1 and summary["unscored"] == 1
    assert summary["accuracy"] == 1.0 and summary["option_reversal_flips"] == 0
    assert summary["references"]["jev"]["count"] == 1


def test_predictions_file_records_each_scored_question(tmp_path):
    import json

    from ruling.evals import write_predictions

    scored = Observation(record=3, question_id="q", question_type="choice", family="f", keys=["a", "b"],
                         logits=np.array([0.0, np.log(3.0)]), label=1, references={}, reversed_argmax=1)
    unscored = Observation(record=4, question_id="q", question_type="choice", family="f", keys=["a", "b"],
                           logits=None, label=0, references={}, reversed_argmax=None)
    path = tmp_path / "p.jsonl"
    write_predictions([scored, unscored], Calibration({"choice": 1.0}), path)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert rows == [{"record": 3, "question_id": "q", "type": "choice", "keys": ["a", "b"],
                     "probabilities": pytest.approx([0.25, 0.75]), "label": 1}]
