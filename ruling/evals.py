"""Labeled-dataset evaluation and temperature fitting.

A dataset is JSONL. Each record holds a state, a questions map in the API
shape, and optionally: `labels`, gold answers by question id (an option key
for Choice, a level index or level text for Score, a boolean for Noul);
`references`, published answer distributions by question id and source name,
such as Jev's own answers, used to report agreement; and `family` for grouped
balanced-accuracy reporting.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pydantic import BaseModel, Field, model_validator

from ruling.calibration import (
    Calibration,
    Provenance,
    choice_confidence,
    expected_calibration_error,
    fit_temperature,
    softmax,
)
from ruling.engine import InputTooLong
from ruling.questions import Choice, Noul, Question, Score, State, option_keys

Distribution = dict[str, float]


class Record(BaseModel):
    state: State
    questions: dict[str, Question] = Field(min_length=1)
    labels: dict[str, str | int | bool] = Field(default_factory=dict)
    references: dict[str, dict[str, Distribution]] = Field(default_factory=dict)
    family: str | None = None

    @model_validator(mode="after")
    def labels_and_references_match_questions(self) -> "Record":
        unknown = (set(self.labels) | set(self.references)) - set(self.questions)
        if unknown:
            raise ValueError(f"labels or references for unknown questions {sorted(unknown)}")
        for qid, label in self.labels.items():
            label_index(self.questions[qid], label)
        for qid, sources in self.references.items():
            keys = set(option_keys(self.questions[qid]))
            for name, distribution in sources.items():
                if set(distribution) != keys:
                    raise ValueError(f"reference {name!r} for {qid!r} must cover exactly {sorted(keys)}")
        return self


@dataclass(frozen=True)
class Observation:
    record: int
    question_id: str
    question_type: str
    family: str | None
    keys: list[str]
    logits: np.ndarray | None
    """None when the record exceeded the engine's input limit and could not be scored."""
    label: int | None
    references: dict[str, np.ndarray]
    reversed_argmax: int | None


def load_dataset(path: Path) -> list[Record]:
    return [Record.model_validate_json(line) for line in path.read_text().splitlines() if line.strip()]


def label_index(question: Choice | Score | Noul, label: str | int | bool) -> int:
    keys = option_keys(question)
    if isinstance(question, Noul):
        if not isinstance(label, bool):
            raise ValueError(f"noul label must be a boolean, got {label!r}")
        return 0 if label else 1
    if isinstance(question, Score):
        if isinstance(label, bool):
            raise ValueError("score label must be a level index or level text")
        if isinstance(label, int):
            if not 0 <= label < len(keys):
                raise ValueError(f"score level {label} out of range")
            return label
        if label in question.criteria:
            return question.criteria.index(label)
        raise ValueError(f"unknown score level {label!r}")
    if label not in keys:
        raise ValueError(f"unknown choice option {label!r}")
    return keys.index(label)


def answer_distribution(answer: dict, keys: list[str]) -> Distribution:
    """A published System One answer as a distribution over canonical keys."""
    if answer["type"] == "noul":
        return {"yes": float(answer["noul"]), "no": 1.0 - float(answer["noul"])}
    probabilities = {str(k): float(v) for k, v in answer["probabilities"].items()}
    missing = set(keys) - set(probabilities)
    if missing:
        raise ValueError(f"published answer lacks options {sorted(missing)}")
    return {key: probabilities[key] for key in keys}


def collect(engine, records: list[Record], extra_references: dict[tuple[int, str], Distribution] | None = None) -> list[Observation]:
    """Score every record. Choice questions are also scored with reversed options.

    A record longer than the engine accepts becomes unscored observations, so a
    report can say how much of a dataset a model could read at all.
    """
    observations = []
    for index, record in enumerate(records):
        try:
            scored = engine.score(record.state, record.questions)
        except InputTooLong:
            observations += [Observation(record=index, question_id=qid, question_type=q.type, family=record.family,
                                         keys=option_keys(q), logits=None,
                                         label=label_index(q, record.labels[qid]) if qid in record.labels else None,
                                         references={}, reversed_argmax=None)
                             for qid, q in record.questions.items()]
            continue
        reversed_questions = {
            qid: Choice(instructions=q.instructions, criteria=dict(reversed(list(q.criteria.items()))))
            for qid, q in record.questions.items()
            if isinstance(q, Choice)
        }
        reversed_scored = engine.score(record.state, reversed_questions) if reversed_questions else None
        for qid, question in record.questions.items():
            raw = scored.raw[qid]
            reversed_argmax = None
            if qid in reversed_questions:
                rev = reversed_scored.raw[qid]
                reversed_argmax = raw.keys.index(rev.keys[int(rev.logits.argmax())])
            references = dict(record.references.get(qid, {}))
            if extra_references and (index, qid) in extra_references:
                references["live"] = extra_references[(index, qid)]
            observations.append(
                Observation(
                    record=index,
                    question_id=qid,
                    question_type=question.type,
                    family=record.family,
                    keys=raw.keys,
                    logits=raw.logits,
                    label=label_index(question, record.labels[qid]) if qid in record.labels else None,
                    references={name: np.array([d[k] for k in raw.keys]) for name, d in references.items()},
                    reversed_argmax=reversed_argmax,
                )
            )
    return observations


def balanced_accuracy(predicted: list[int], gold: list[int]) -> float:
    recalls = []
    for label in sorted(set(gold)):
        hits = sum(1 for p, g in zip(predicted, gold) if g == label and p == label)
        recalls.append(hits / gold.count(label))
    return float(np.mean(recalls))


def total_variation(p: np.ndarray, q: np.ndarray) -> float:
    return float(0.5 * np.abs(p - q).sum())


def report(observations: list[Observation], calibration: Calibration) -> dict:
    by_type: dict[str, list[Observation]] = defaultdict(list)
    for obs in observations:
        by_type[obs.question_type].append(obs)
    result = {"question_types": {}}
    for question_type, everything in by_type.items():
        group = [o for o in everything if o.logits is not None]
        if not group:
            result["question_types"][question_type] = {"count": 0, "unscored": len(everything)}
            continue
        temperature = calibration.temperature(question_type)
        probs = [softmax(o.logits, temperature) for o in group]
        summary = {"count": len(group), "unscored": len(everything) - len(group), "temperature": temperature,
                   "mean_confidence": float(np.mean([choice_confidence(p) for p in probs]))}
        labeled = [(o, p) for o, p in zip(group, probs) if o.label is not None]
        summary["labeled_count"] = len(labeled)
        if labeled:
            predicted = [int(p.argmax()) for _, p in labeled]
            gold = [o.label for o, _ in labeled]
            correct = [a == b for a, b in zip(predicted, gold)]
            summary.update({
                "accuracy": float(np.mean(correct)),
                "balanced_accuracy": balanced_accuracy(predicted, gold),
                "expected_calibration_error": expected_calibration_error([float(p.max()) for _, p in labeled], correct),
                "brier": float(np.mean([((p - np.eye(len(p))[g]) ** 2).sum() for (_, p), g in zip(labeled, gold)])),
                "negative_log_likelihood": float(np.mean([-np.log(max(p[g], 1e-12)) for (_, p), g in zip(labeled, gold)])),
            })
            families = defaultdict(list)
            for (o, _), p, g in zip(labeled, predicted, gold):
                if o.family is not None:
                    families[o.family].append((p, g))
            if families and len(families) == len({o.family for o, _ in labeled}):
                per_family = {name: balanced_accuracy([p for p, _ in pairs], [g for _, g in pairs])
                              for name, pairs in families.items()}
                summary["family_balanced_accuracy"] = per_family
                summary["mean_family_balanced_accuracy"] = float(np.mean(list(per_family.values())))
        if question_type == "choice":
            flips = sum(1 for o, p in zip(group, probs) if o.reversed_argmax != int(p.argmax()))
            summary["option_reversal_flips"] = flips
            summary["option_reversal_flip_rate"] = flips / len(group)
        names = sorted({name for o in group for name in o.references})
        if names:
            summary["references"] = {name: _reference_summary(name, group, probs) for name in names}
        result["question_types"][question_type] = summary
    return result


def _reference_summary(name: str, group: list[Observation], probs: list[np.ndarray]) -> dict:
    pairs = [(o, p, o.references[name]) for o, p in zip(group, probs) if name in o.references]
    summary = {
        "count": len(pairs),
        "argmax_agreement": float(np.mean([int(p.argmax()) == int(r.argmax()) for _, p, r in pairs])),
        "mean_total_variation": float(np.mean([total_variation(p, r) for _, p, r in pairs])),
    }
    labeled = [(o, p, r) for o, p, r in pairs if o.label is not None]
    if labeled:
        summary["labeled_count"] = len(labeled)
        summary["reference_accuracy"] = float(np.mean([int(r.argmax()) == o.label for o, _, r in labeled]))
        summary["ruling_accuracy_on_same_rows"] = float(np.mean([int(p.argmax()) == o.label for o, p, _ in labeled]))
    return summary


def fit(observations: list[Observation], provenance: Provenance) -> Calibration:
    by_type: dict[str, list[Observation]] = defaultdict(list)
    for obs in observations:
        if obs.label is not None and obs.logits is not None:
            by_type[obs.question_type].append(obs)
    if not by_type:
        raise ValueError("cannot fit a calibration without labeled questions")
    return Calibration(
        temperatures={
            question_type: fit_temperature([o.logits for o in group], [o.label for o in group])
            for question_type, group in by_type.items()
        },
        provenance=provenance,
    )


def write_predictions(observations: list[Observation], calibration: Calibration, path: Path) -> None:
    """One line per scored question with its calibrated distribution, for ensembling and audits."""
    with path.open("w") as out:
        for obs in observations:
            if obs.logits is None:
                continue
            probs = softmax(obs.logits, calibration.temperature(obs.question_type))
            out.write(json.dumps({"record": obs.record, "question_id": obs.question_id, "type": obs.question_type,
                                  "keys": obs.keys, "probabilities": [float(p) for p in probs], "label": obs.label}) + "\n")


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n")
