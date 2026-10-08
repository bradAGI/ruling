"""One engine answers; another takes over whenever the first is unavailable.

Built for a hosted primary such as `typesafe:jev-latest` with a local model
behind it: a 503, a 429 or a timeout from the host is answered by the fallback
instead of failing the request. Which engine answered is in the response's
`model`, so a caller can tell a fallback answer from a primary one.

Only unavailability triggers the fallback. A refusal (413, 400) or a bug is
raised as it would be from the primary alone.
"""

from __future__ import annotations

import logging

from ruling.calibration import Provenance
from ruling.engine import RawScore, Scored
from ruling.hosted import HostUnavailable
from ruling.questions import Choice, Noul, Score, State, SystemOneRequest, SystemOneResponse, Usage

PREFIX = "failover:"
log = logging.getLogger("ruling")


class FailoverEngine:
    """Same interface as the engines it wraps; limits are the primary's, since the fallback is best effort."""

    def __init__(self, primary, fallback):
        self.primary, self.fallback = primary, fallback
        self.model_id = f"{PREFIX}{primary.model_id}>{fallback.model_id}"
        self.revision = primary.revision
        self.adapter = primary.adapter
        self.rotations = primary.rotations
        self.prior_debias = primary.prior_debias
        self.max_input_tokens = primary.max_input_tokens
        self.max_branch_tokens = primary.max_branch_tokens
        self.max_options = primary.max_options
        self.calibration = primary.calibration

    @property
    def provenance(self) -> Provenance:
        return Provenance(self.model_id, self.revision, self.rotations, self.prior_debias, self.adapter)

    def _answering(self, state: State, questions: dict[str, Choice | Score | Noul]):
        """The engine that answered, and its scores."""
        try:
            return self.primary, self.primary.score(state, questions)
        except HostUnavailable as exc:
            log.warning("failover from %s to %s: %s", self.primary.model_id, self.fallback.model_id, exc)
            return self.fallback, self.fallback.score(state, questions)

    def score(self, state: State, questions: dict[str, Choice | Score | Noul]) -> Scored:
        return self._answering(state, questions)[1]

    def evaluate(self, request: SystemOneRequest) -> SystemOneResponse:
        engine, scored = self._answering(request.state, request.questions)
        answers = {qid: engine.answer(request.questions[qid], scored.raw[qid]) for qid in request.questions}
        return SystemOneResponse(model=engine.model_id, answers=answers, usage=Usage(input_tokens=scored.input_tokens))

    def answer(self, question, raw: RawScore):
        return self.primary.answer(question, raw)
