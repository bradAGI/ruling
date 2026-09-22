"""Temperature scaling, confidence, and calibration metrics.

Restricted log-probabilities (one per allowed answer) are divided by a
per-question-type temperature before the softmax. A temperature above 1
flattens an overconfident model; below 1 sharpens an underconfident one.
Temperatures are fit on labeled data by minimizing negative log-likelihood and
stored with the model, revision, and scoring settings they were fit for, so a
file cannot silently apply to a different model.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

QUESTION_TYPES = ("choice", "score", "noul")


@dataclass(frozen=True)
class Provenance:
    model: str
    revision: str | None
    rotations: int
    prior_debias: bool
    adapter: str | None = None

    def mismatch(self, other: "Provenance") -> list[str]:
        return [name for name in ("model", "revision", "rotations", "prior_debias", "adapter")
                if getattr(self, name) != getattr(other, name)]


@dataclass
class Calibration:
    temperatures: dict[str, float] = field(default_factory=dict)
    provenance: Provenance | None = None

    def temperature(self, question_type: str) -> float:
        return self.temperatures.get(question_type, 1.0)

    @classmethod
    def load(cls, path: Path) -> "Calibration":
        data = json.loads(path.read_text())
        temperatures = {k: float(v) for k, v in data["temperatures"].items()}
        for question_type, value in temperatures.items():
            if question_type not in QUESTION_TYPES or value <= 0:
                raise ValueError(f"invalid temperature for {question_type!r}: {value}")
        return cls(temperatures=temperatures, provenance=Provenance(**data["provenance"]))

    def save(self, path: Path) -> None:
        if self.provenance is None:
            raise ValueError("a calibration cannot be saved without provenance")
        data = {"temperatures": self.temperatures, "provenance": self.provenance.__dict__}
        path.write_text(json.dumps(data, indent=2) + "\n")

    def check(self, provenance: Provenance) -> None:
        """Raise unless this calibration was fit for exactly this model and scoring setup."""
        if self.provenance is None:
            return
        fields = self.provenance.mismatch(provenance)
        if fields:
            details = ", ".join(f"{f}: calibrated for {getattr(self.provenance, f)!r}, running {getattr(provenance, f)!r}" for f in fields)
            raise ValueError(f"calibration does not match the running engine ({details})")


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    scaled = np.asarray(logits, dtype=np.float64) / temperature
    scaled -= scaled.max()
    exp = np.exp(scaled)
    return exp / exp.sum()


def log_softmax(logits: np.ndarray) -> np.ndarray:
    shifted = np.asarray(logits, dtype=np.float64) - np.max(logits)
    return shifted - np.log(np.exp(shifted).sum())


def choice_confidence(probabilities: np.ndarray) -> float:
    """How far the leading option stands above a uniform distribution, from 0 to 1.

    This and `score_confidence` reimplement the formulas TypeSafe publishes in
    its MIT-licensed system-one-adapter, so a confidence value here means the
    same thing as one from the hosted API.
    """
    probabilities = _normalized(probabilities)
    if probabilities.size == 1:
        return 1.0
    uniform = 1.0 / probabilities.size
    return float((probabilities.max() - uniform) / (1.0 - uniform))


def score_confidence(probabilities: np.ndarray) -> float:
    """How tightly a score's mass sits around its modal level, from 0 to 1."""
    probabilities = _normalized(probabilities)
    if probabilities.size == 1:
        return 1.0
    levels = np.arange(probabilities.size)
    mode = int(probabilities.argmax())
    spread = float((probabilities * np.abs(levels - mode)).sum())
    uniform_spread = float(np.abs(levels - (probabilities.size - 1) / 2).mean())
    return max(0.0, 1.0 - spread / uniform_spread)


def _normalized(probabilities: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    total = probabilities.sum()
    return np.full(probabilities.size, 1.0 / probabilities.size) if total == 0 else probabilities / total


def negative_log_likelihood(logits: list[np.ndarray], labels: list[int], temperature: float) -> float:
    total = 0.0
    for row, label in zip(logits, labels):
        total -= math.log(max(softmax(row, temperature)[label], 1e-12))
    return total / len(labels)


def fit_temperature(logits: list[np.ndarray], labels: list[int]) -> float:
    """Golden-section search over log-temperature for the NLL minimum."""
    if not logits:
        raise ValueError("cannot fit a temperature without examples")
    lo, hi = math.log(0.05), math.log(20.0)
    ratio = (math.sqrt(5) - 1) / 2
    a = hi - ratio * (hi - lo)
    b = lo + ratio * (hi - lo)
    fa = negative_log_likelihood(logits, labels, math.exp(a))
    fb = negative_log_likelihood(logits, labels, math.exp(b))
    for _ in range(60):
        if fa < fb:
            hi, b, fb = b, a, fa
            a = hi - ratio * (hi - lo)
            fa = negative_log_likelihood(logits, labels, math.exp(a))
        else:
            lo, a, fa = a, b, fb
            b = lo + ratio * (hi - lo)
            fb = negative_log_likelihood(logits, labels, math.exp(b))
    return math.exp((lo + hi) / 2)


def coverage_at_error(confidence: list[float], correct: list[bool], budget: float) -> float:
    """The largest share of decisions acceptable in descending-confidence order while the
    error rate among the accepted stays within `budget`.

    This is jev-benchmarks' selective-automation metric. Equal confidences are admitted as
    a whole group, so a permutation of tied rows cannot change the answer. A model that
    is confidently wrong scores low here whatever its accuracy; a temperature cannot move
    it, because the order is what is measured.
    """
    if not 0 <= budget <= 1:
        raise ValueError("error budget must be within [0, 1]")
    confidence, correct = np.asarray(confidence, dtype=np.float64), np.asarray(correct, dtype=bool)
    if confidence.size == 0:
        return 0.0
    order = np.argsort(-confidence, kind="stable")
    ranked, errors = confidence[order], np.cumsum(~correct[order])
    group_ends = np.r_[np.flatnonzero(ranked[1:] != ranked[:-1]), ranked.size - 1]
    accepted = group_ends + 1
    within = np.flatnonzero(errors[group_ends] <= budget * accepted)
    return float(accepted[within[-1]] / ranked.size) if within.size else 0.0


def confident_error_rate(confidence: list[float], correct: list[bool], threshold: float = 0.9) -> float:
    """The share of all decisions that were wrong while reporting at least `threshold`."""
    confidence, correct = np.asarray(confidence, dtype=np.float64), np.asarray(correct, dtype=bool)
    return float(np.mean((confidence >= threshold) & ~correct)) if confidence.size else 0.0


def expected_calibration_error(top_probabilities: list[float], correct: list[bool], bins: int = 10) -> float:
    """Gap between stated top-1 probability and observed accuracy, weighted by bin size."""
    if not top_probabilities:
        return 0.0
    probs = np.asarray(top_probabilities, dtype=np.float64)
    hits = np.asarray(correct, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    error = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (probs > lo) & (probs <= hi) if lo > 0 else (probs >= lo) & (probs <= hi)
        if mask.any():
            error += mask.mean() * abs(probs[mask].mean() - hits[mask].mean())
    return float(error)
