"""Labeled-dataset evaluation and temperature fitting.

A dataset is JSONL, or a JSON array. Each record holds a state, a questions
map in the API shape, and optionally: `labels`, gold answers by question id
(an option key for Choice, a level index or level text for Score, a boolean
for Noul); `references`, published answer distributions by question id and
source name, such as Jev's own answers, used to report agreement; `family`
for grouped balanced-accuracy reporting; and an optional `id` and `weight`.

Rows written for hosted decision platforms are the same shape and load unchanged. Their extra label forms
are normalized on validation: the strings "true"/"false" become a boolean, a
score option id such as "2" becomes the index 2, and a soft label (a
probability of yes, a per-option map, or a per-level list) becomes the
`target` reference with the label set to its most likely option. A
single-question row holds `question` and `label` (or `target`, or
`expected`) instead of maps, and becomes one question named `decision`.
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pydantic import BaseModel, Field, model_validator

from ruling.calibration import (
    Calibration,
    Provenance,
    choice_confidence,
    confident_error_rate,
    coverage_at_error,
    expected_calibration_error,
    fit_temperature,
    softmax,
)
from ruling.engine import InputTooLong
from ruling.metrics import decision_metrics
from ruling.questions import MAX_CHOICE_OPTIONS, Choice, Noul, Question, Score, State, check_askable, option_keys

Distribution = dict[str, float]
Label = str | int | bool | float | list[float] | dict[str, float]

TARGET = "target"
"""The reference name whose distribution is the soft training and scoring target for a question."""

SINGLE_QUESTION = "decision"
"""The question id a single-question row is given, the name hosted platforms use for it."""

SOFT_SUM_TOLERANCE = 0.02
"""How far a soft label may stray from summing to one before it is refused; the tolerance hosted platforms allow."""

NOUL_KEYS = {"true": "yes", "false": "no"}
"""The true/false noul option ids hosted platforms use, mapped onto ruling's canonical keys."""


ERROR_BUDGET = 0.05
"""The error rate a deployment accepts among automated decisions; coverage is reported against it."""


class Record(BaseModel):
    id: str | int | None = None
    state: State
    questions: dict[str, Question] = Field(min_length=1)
    labels: dict[str, Label] = Field(default_factory=dict)
    references: dict[str, dict[str, Distribution]] = Field(default_factory=dict)
    family: str | None = None
    weight: float = Field(default=1.0, ge=0)
    """How much the record counts in a training update."""

    @model_validator(mode="after")
    def labels_and_references_match_questions(self) -> "Record":
        check_askable(self.questions, MAX_CHOICE_OPTIONS)
        unknown = (set(self.labels) | set(self.references)) - set(self.questions)
        if unknown:
            raise ValueError(f"labels or references for unknown questions {sorted(unknown)}")
        for qid, label in self.labels.items():
            question = self.questions[qid]
            try:
                soft = soft_target(question, label)
                if soft is not None:
                    self.references.setdefault(qid, {})[TARGET] = soft
                    label = label_for_key(question, max(option_keys(question), key=soft.__getitem__))
                self.labels[qid] = canonical_label(question, label)
                label_index(question, self.labels[qid])
            except ValueError as error:
                raise ValueError(f"label for {qid!r}: {error}") from error
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
    latency_ms: float | None = None
    """Wall time for the whole record's request, on every observation the request produced."""


def load_dataset(path: Path) -> list[Record]:
    """Records from a `.jsonl` file (one row per line) or a `.json` array, in ruling's row shape or a hosted platform's."""
    text = path.read_text()
    if path.suffix == ".json":
        rows = json.loads(text)
        if not isinstance(rows, list):
            raise ValueError(f"{path}: a .json dataset must be an array of rows")
    else:
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    records = []
    for index, row in enumerate(rows, start=1):
        try:
            records.append(Record.model_validate(as_record(row)))
        except ValueError as error:
            where = f"row {index}" + (f" ({row['id']})" if isinstance(row, dict) and row.get("id") is not None else "")
            raise ValueError(f"{path} {where}: {error}") from error
    return records


def as_record(row: dict) -> dict:
    """A row in any accepted shape as a record dict: a single-question row becomes one question, `decision`."""
    if "questions" in row:
        return row
    if "question" not in row:
        raise ValueError("a row needs a 'questions' map or a single 'question'")
    label = _single_label(row)
    record = {k: v for k, v in row.items() if k not in ("question", "label", "target", "expected")}
    record["questions"] = {SINGLE_QUESTION: row["question"]}
    if label is not None:
        record["labels"] = {SINGLE_QUESTION: label}
    return record


def _single_label(row: dict):
    """Precedence for a single-question row: `expected`, else a soft `target`, else `label`, else `target`."""
    if "expected" in row:
        return row["expected"]
    target, label = row.get("target"), row.get("label")
    if isinstance(target, (list, float, dict)) and not isinstance(target, bool):
        return target
    return label if label is not None else target


def soft_target(question: Choice | Score | Noul, label: Label) -> Distribution | None:
    """A soft label as a distribution over the question's canonical keys, or None when the label is hard.

    Noul takes a probability of yes or a map over true/false (or yes/no); Choice a map over its options or a
    list in option order; Score a map over "0".."n-1" or a list from the lowest level up.
    """
    keys = option_keys(question)
    if isinstance(label, (bool, str)):
        return None
    if isinstance(label, dict):
        rename = NOUL_KEYS if isinstance(question, Noul) else {}
        probs = {rename.get(str(k), str(k)): float(v) for k, v in label.items()}
    elif isinstance(label, list):
        if isinstance(question, Noul):
            raise ValueError("a noul soft label is a probability of yes or a map over true/false, not a list")
        if len(label) != len(keys):
            raise ValueError(f"a soft label needs {len(keys)} probabilities, got {len(label)}")
        probs = dict(zip(keys, map(float, label)))
    elif isinstance(question, Noul):
        if not 0 <= label <= 1:
            raise ValueError(f"a noul probability must be within [0, 1], got {label!r}")
        probs = {"yes": float(label), "no": 1.0 - float(label)}
    else:
        return None
    if set(probs) != set(keys):
        raise ValueError(f"a soft label must cover exactly {keys}, got {sorted(probs)}")
    if min(probs.values()) < 0:
        raise ValueError("a soft label must be non-negative")
    if abs(sum(probs.values()) - 1.0) > SOFT_SUM_TOLERANCE:
        raise ValueError(f"a soft label must sum to 1, got {sum(probs.values()):.4f}")
    return probs


def canonical_label(question: Choice | Score | Noul, label: Label) -> str | int | bool:
    """The string forms of a hard label in ruling's form: "true"/"false" to a bool, a score id "2" to 2."""
    if isinstance(question, Noul) and isinstance(label, str) and label.lower() in ("true", "false", "yes", "no"):
        return label.lower() in ("true", "yes")
    if isinstance(question, Score) and isinstance(label, str) and label.isdigit():
        return int(label)
    return label


def label_for_key(question: Choice | Score | Noul, key: str) -> str | int | bool:
    """The hard label that names one canonical key."""
    if isinstance(question, Noul):
        return key == "yes"
    if isinstance(question, Score):
        return int(key)
    return key


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
        started = time.perf_counter()
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
        latency_ms = (time.perf_counter() - started) * 1000
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
                    latency_ms=latency_ms,
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
    """Per-type summaries under `question_types`, plus the pooled decision metrics, by type and by question."""
    by_type: dict[str, list[Observation]] = defaultdict(list)
    for obs in observations:
        by_type[obs.question_type].append(obs)
    result = {"question_types": {}, **decision_blocks(observations, calibration)}
    latencies = sorted({(o.record, o.latency_ms) for o in observations if o.latency_ms is not None})
    if latencies:
        values = np.array([ms for _, ms in latencies])
        result["latency_ms"] = {"p50": float(np.percentile(values, 50)), "p95": float(np.percentile(values, 95)),
                                "records": len(values)}
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
                "coverage_at_5pct_error": coverage_at_error([float(p.max()) for _, p in labeled], correct, ERROR_BUDGET),
                "confident_error_rate": confident_error_rate([float(p.max()) for _, p in labeled], correct),
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


def decision_blocks(observations: list[Observation], calibration: Calibration) -> dict:
    """`decision_metrics` on every labeled, scored question: pooled, keyed by type, and keyed by question id.

    The target is the record's soft `target` reference where it has one, else one-hot on the label.
    """
    labeled = [o for o in observations if o.logits is not None and o.label is not None]
    probs = [softmax(o.logits, calibration.temperature(o.question_type)) for o in labeled]
    targets = [o.references[TARGET] if TARGET in o.references else np.eye(len(o.keys))[o.label] for o in labeled]

    def block(select) -> dict:
        chosen = [i for i, o in enumerate(labeled) if select(o)]
        return decision_metrics([probs[i] for i in chosen], [targets[i] for i in chosen])

    return {
        "overall": block(lambda o: True),
        "by_type": {t: block(lambda o: o.question_type == t) for t in dict.fromkeys(o.question_type for o in labeled)},
        "by_question": {q: block(lambda o: o.question_id == q) for q in dict.fromkeys(o.question_id for o in labeled)},
    }


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
