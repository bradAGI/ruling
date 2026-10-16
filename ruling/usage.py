"""What a server answered, counted per hour and per model.

The counts live in memory for the server's lifetime: a restart starts them
over, and nothing is written to disk. One decision is one `/v1/systemone`
request or one row of a `/v1/predict` or `/v1/evaluations` batch that was
answered; refused requests and rows count nothing.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime

HOUR = 3600


@dataclass
class _Counts:
    decisions: int = 0
    input_tokens: int = 0


class UsageMeter:
    def __init__(self, now: Callable[[], float] = time.time):
        self._now = now
        self._buckets: dict[tuple[int, str], _Counts] = {}
        self._lock = threading.Lock()

    def record(self, model: str, decisions: int, input_tokens: int) -> None:
        hour = int(self._now() // HOUR) * HOUR
        with self._lock:
            counts = self._buckets.setdefault((hour, model), _Counts())
            counts.decisions += decisions
            counts.input_tokens += input_tokens

    def buckets(self, starting_on: datetime | None = None, ending_before: datetime | None = None) -> list[dict]:
        """Every bucket whose hour starts within the window, by hour then model; a naive bound is read as UTC."""
        start, end = _epoch(starting_on), _epoch(ending_before)
        with self._lock:
            items = sorted(self._buckets.items())
        return [{"hour": datetime.fromtimestamp(hour, UTC).isoformat(), "model": model,
                 "decisions": counts.decisions, "input_tokens": counts.input_tokens}
                for (hour, model), counts in items
                if (start is None or hour >= start) and (end is None or hour < end)]


class Metered:
    """An engine whose scoring passes add up their input tokens, so a batch that answers
    through `ruling.evals` can report what it cost without the evaluation code counting."""

    def __init__(self, engine):
        self.engine = engine
        self.input_tokens = 0

    def __getattr__(self, name: str):
        return getattr(self.engine, name)

    def score(self, state, questions):
        scored = self.engine.score(state, questions)
        self.input_tokens += scored.input_tokens
        return scored

    def evaluate(self, request):
        response = self.engine.evaluate(request)
        self.input_tokens += response.usage.input_tokens
        return response


def _epoch(moment: datetime | None) -> float | None:
    if moment is None:
        return None
    return (moment if moment.tzinfo is not None else moment.replace(tzinfo=UTC)).timestamp()
