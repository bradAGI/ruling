"""Two engines behind one answer: a small one for every question, a larger one for the doubtful ones.

The first engine scores the whole request. Any question whose calibrated top
probability falls under the threshold is scored again by the second engine, and
that answer replaces the first. Measured on TypeSafe's and Every's public rows
with a threshold fitted on one set and applied to the other, the second engine
ran on about 13% of questions and the pair matched its accuracy alone.

Each engine's own temperature is applied here, so the logits handed back are
already calibrated and the cascade carries an identity calibration of its own.
"""

from __future__ import annotations

import numpy as np

from ruling.calibration import Calibration, Provenance, softmax
from ruling.engine import RawScore, Scored, answer
from ruling.questions import Choice, Noul, Score, State, SystemOneRequest, SystemOneResponse, Usage

PREFIX = "cascade:"


class CascadeEngine:
    """Same interface as the engines it wraps, so the server and the harness need no special case."""

    prior_debias = False

    def __init__(self, first, second, threshold: float):
        if not 0 <= threshold <= 1:
            raise ValueError("cascade threshold must be a probability")
        self.first, self.second, self.threshold = first, second, threshold
        self.model_id = f"{PREFIX}{first.model_id}>{second.model_id}"
        self.revision = None
        self.adapter = first.adapter
        self.rotations = first.rotations
        self.max_input_tokens = min(first.max_input_tokens, second.max_input_tokens)
        self.max_branch_tokens = min(first.max_branch_tokens, second.max_branch_tokens)
        self.max_options = min(first.max_options, second.max_options)
        self.calibration = Calibration()

    @property
    def provenance(self) -> Provenance:
        return Provenance(self.model_id, self.revision, self.rotations, self.prior_debias, self.adapter)

    def score(self, state: State, questions: dict[str, Choice | Score | Noul]) -> Scored:
        first = self.first.score(state, questions)
        raw = {qid: self._tempered(self.first, q, first.raw[qid]) for qid, q in questions.items()}
        doubtful = {qid: q for qid, q in questions.items() if softmax(raw[qid].logits).max() < self.threshold}
        tokens = first.input_tokens
        if doubtful:
            second = self.second.score(state, doubtful)
            tokens += second.input_tokens
            for qid, q in doubtful.items():
                raw[qid] = self._tempered(self.second, q, second.raw[qid])
        return Scored(raw=raw, input_tokens=tokens)

    @staticmethod
    def _tempered(engine, question: Choice | Score | Noul, raw: RawScore) -> RawScore:
        """The engine's logits with its own fitted temperature already divided out."""
        temperature = engine.calibration.temperature(question.type)
        return RawScore(keys=raw.keys, logits=np.asarray(raw.logits, dtype=np.float64) / temperature)

    def evaluate(self, request: SystemOneRequest) -> SystemOneResponse:
        scored = self.score(request.state, request.questions)
        answers = {qid: answer(request.questions[qid], scored.raw[qid], self.calibration) for qid in request.questions}
        return SystemOneResponse(model=self.model_id, answers=answers, usage=Usage(input_tokens=scored.input_tokens))

    def answer(self, question, raw: RawScore):
        return answer(question, raw, self.calibration)
