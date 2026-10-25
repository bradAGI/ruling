"""Whether a served question's answers have moved: the most recent window of a model's answers to it against
the window before, read from the decision ledger without asking any model.

Drift says the traffic changed, not that the answers became wrong: a new product line or a seasonal rush moves
the mix of answers as surely as a broken model does. It is the cue to collect outcomes and retrain, and to
look at the questions it names.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from ruling.ledger import DecisionLedger
from ruling.questions import ChoiceAnswer, NoulAnswer, ScoreAnswer

WINDOW = 200
"""Decisions in each of the two windows compared; a question with fewer than two windows of traffic is not reported."""

SIGNIFICANCE = 0.01
"""How unlikely by chance the change in which answer wins must be before it counts."""

MIN_SHIFT = 0.1
"""The least movement in the mix of answers, as total variation, that counts however significant: large traffic
makes trivial movements significant."""


@dataclass
class _Window:
    probabilities: np.ndarray
    winners: np.ndarray
    confidence: float = 0.0
    count: int = 0

    def add(self, distribution: np.ndarray) -> None:
        self.probabilities += distribution
        self.winners[int(distribution.argmax())] += 1
        self.confidence += float(distribution.max())
        self.count += 1


def drift(ledger: DecisionLedger, window: int = WINDOW) -> list[dict]:
    """Per model and question, the two most recent windows of answers compared: `answer_shift` is the total
    variation between their mean answer distributions (0 to 1), `confidence_change` the change in the mean
    probability of the winning answer, and `changed` whether the mix of winning answers moved by at least
    `MIN_SHIFT` beyond what chance explains (a chi-square test at `SIGNIFICANCE`)."""
    seen: dict[tuple, list[_Window]] = defaultdict(list)
    for model, answers in ledger.answers_newest_first():
        for qid, answer in answers.items():
            distribution = _distribution(answer)
            if distribution is None:
                continue
            keys, values = distribution
            windows = seen[(model, qid, keys)]
            if not windows or windows[-1].count == window:
                if len(windows) == 2:
                    continue
                windows.append(_Window(np.zeros(len(keys)), np.zeros(len(keys))))
            windows[-1].add(values)
    return [_compare(model, qid, recent, earlier)
            for (model, qid, _), windows in seen.items() if len(windows) == 2
            for recent, earlier in [windows] if earlier.count == window]


def _distribution(answer) -> tuple[tuple[str, ...], np.ndarray] | None:
    """The answer's distribution over its options, keyed so a question asked with other options is another question."""
    if isinstance(answer, (ChoiceAnswer, ScoreAnswer)):
        keys = tuple(answer.probabilities)
        return keys, np.array([answer.probabilities[k] for k in keys])
    if isinstance(answer, NoulAnswer):
        return ("yes", "no"), np.array([answer.noul, 1.0 - answer.noul])
    return None


def _compare(model: str, question: str, recent: _Window, earlier: _Window) -> dict:
    shift = float(np.abs(recent.probabilities / recent.count - earlier.probabilities / earlier.count).sum() / 2)
    return {"model": model, "question": question, "decisions": [recent.count, earlier.count],
            "answer_shift": shift,
            "confidence_change": recent.confidence / recent.count - earlier.confidence / earlier.count,
            "changed": shift >= MIN_SHIFT and _chi_square_p(np.stack([recent.winners, earlier.winners])) < SIGNIFICANCE}


def _chi_square_p(table: np.ndarray) -> float:
    """The chance of a contingency table at least this uneven if both rows came from one mix, by Pearson's
    chi-square with the Wilson-Hilferty approximation to its tail."""
    table = table[:, table.sum(axis=0) > 0]
    degrees = table.shape[1] - 1
    if degrees == 0:
        return 1.0
    expected = table.sum(axis=1, keepdims=True) * table.sum(axis=0, keepdims=True) / table.sum()
    statistic = float(((table - expected) ** 2 / expected).sum())
    z = ((statistic / degrees) ** (1 / 3) - (1 - 2 / (9 * degrees))) / math.sqrt(2 / (9 * degrees))
    return 0.5 * math.erfc(z / math.sqrt(2))
