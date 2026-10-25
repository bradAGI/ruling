"""Every decision a server answers, and the outcomes reported for it later, kept on disk.

An outcome is what turned out to be right for a question, reported by whoever
learns it: the ticket's final team, whether the refund was granted. Decisions
with outcomes are training records, so a model can be retrained on its own
traffic with `ruling export-decisions` and `ruling train`.

SQLite keeps the ledger on disk and indexed, so lookups and memory stay flat
however much traffic it holds. A retention bounds the disk: a decision nobody
reports an outcome for within it is dropped, state and all, the next time a
decision is recorded; a decision with an outcome is training data and stays.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from collections.abc import Callable, Iterator
from pathlib import Path

from pydantic import BaseModel, Field, model_validator

from ruling.evals import Label, Record, label_index
from ruling.questions import (QUESTION, Choice, ChoiceAnswer, Noul, NoulAnswer, Score, ScoreAnswer, Strict,
                              SystemOneRequest, SystemOneResponse, option_keys)

SCHEMA = """
CREATE TABLE IF NOT EXISTS decisions (
    id TEXT PRIMARY KEY,
    created REAL NOT NULL,
    model TEXT NOT NULL,
    request TEXT NOT NULL,
    response TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS decisions_created ON decisions(created);
CREATE TABLE IF NOT EXISTS outcomes (
    decision TEXT NOT NULL REFERENCES decisions(id),
    question TEXT NOT NULL,
    label TEXT NOT NULL,
    reported REAL NOT NULL,
    PRIMARY KEY (decision, question)
);
CREATE TABLE IF NOT EXISTS idempotency (
    key TEXT PRIMARY KEY,
    decision TEXT NOT NULL REFERENCES decisions(id)
);
"""


IDEMPOTENCY_HEADER = "Idempotency-Key"
"""A client's id for one decision, the same across its retries, so a retry is answered with the logged decision."""

MAX_OUTCOMES = 1000
"""Outcomes one batch may carry; a batch is stored whole or not at all."""


class Correction(Strict):
    """Whether the served answer to a question was right, for whoever can say that more easily than name the
    label: right labels it with what was served; wrong labels it with `answer`, which a question of two options
    may leave out since the other option is the only one left."""

    correct: bool
    answer: str | int | bool | None = None

    @model_validator(mode="after")
    def _answer_only_when_wrong(self) -> Correction:
        if self.correct and self.answer is not None:
            raise ValueError("a correct answer needs no answer; send one only with correct: false")
        return self


class Outcome(Strict):
    """What turned out to be right, per question: `labels` in the forms a training record takes, `corrections` as
    right or wrong marks on what was served, or both for different questions."""

    labels: dict[str, Label] = Field(default_factory=dict)
    corrections: dict[str, Correction] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _says_something_once(self) -> Outcome:
        if not self.labels and not self.corrections:
            raise ValueError("an outcome needs labels or corrections")
        both = set(self.labels) & set(self.corrections)
        if both:
            raise ValueError(f"{', '.join(sorted(both))}: given both a label and a correction")
        return self


class DecisionOutcome(Outcome):
    """An outcome with the id of the decision it is for, as a batch carries it."""

    id: str


class OutcomeBatch(Strict):
    outcomes: list[DecisionOutcome] = Field(min_length=1, max_length=MAX_OUTCOMES)


class OutcomeRefused(ValueError):
    """An outcome that does not fit its decision, naming the field at fault as the API reports it."""

    def __init__(self, message: str, param: str):
        super().__init__(message)
        self.param = param


class UnknownDecision(KeyError):
    def __str__(self) -> str:
        return f"no decision {self.args[0]!r} in the ledger"


class DecisionLedger:
    def __init__(self, path: Path, retain_seconds: float | None = None, now: Callable[[], float] = time.time):
        # Server requests arrive on a thread pool; one connection behind a lock serves them all.
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.executescript(SCHEMA)
        self._lock = threading.Lock()
        self._retain = retain_seconds
        self._now = now

    def record(self, request: SystemOneRequest, response: SystemOneResponse, key: str | None = None) -> str:
        """Store one answered request and return the id outcomes will refer to. A request carrying the `key` of a
        decision already held is a client's retry of it, and gets that decision's id instead of a second entry.

        With a retention, unresolved decisions older than it go first, so the ledger never
        outgrows the traffic of one retention period plus what has been resolved."""
        now = self._now()
        with self._lock, self._db:
            if self._retain is not None:
                self._db.execute("DELETE FROM decisions WHERE created < ? AND id NOT IN (SELECT decision FROM outcomes)",
                                 (now - self._retain,))
                self._db.execute("DELETE FROM idempotency WHERE decision NOT IN (SELECT id FROM decisions)")
            if key is not None:
                held = self._db.execute("SELECT decision FROM idempotency WHERE key = ?", (key,)).fetchone()
                if held is not None:
                    return held[0]
            decision = uuid.uuid4().hex
            self._db.execute("INSERT INTO decisions VALUES (?, ?, ?, ?, ?)",
                             (decision, now, response.model, request.model_dump_json(exclude_none=True),
                              response.model_dump_json(exclude_none=True)))
            if key is not None:
                self._db.execute("INSERT INTO idempotency VALUES (?, ?)", (key, decision))
        return decision

    def replay(self, key: str) -> SystemOneResponse | None:
        """The logged response, with its id, of the decision made under `key`; None when there is none."""
        with self._lock:
            row = self._db.execute("SELECT d.id, d.response FROM idempotency i JOIN decisions d ON d.id = i.decision "
                                   "WHERE i.key = ?", (key,)).fetchone()
        if row is None:
            return None
        return SystemOneResponse.model_validate_json(row[1]).model_copy(update={"id": row[0]})

    def ids(self) -> list[str]:
        """Every decision still held, oldest first."""
        with self._lock:
            return [row[0] for row in self._db.execute("SELECT id FROM decisions ORDER BY created, rowid")]

    def outcome(self, decision: str, labels: dict | None = None, corrections: dict | None = None) -> Record:
        """Attach outcomes to a decision; a later outcome for the same question replaces the earlier one.

        The labels are checked as a training record would check them, so nothing
        stored here can fail later in `ruling train`. Returns the record as it now stands.
        """
        with self._lock, self._db:
            return self._apply(decision, Outcome(labels=labels or {}, corrections=corrections or {}))

    def outcomes(self, outcomes: list[tuple[str, Outcome]]) -> list[Record]:
        """Attach every outcome in one transaction: an unknown decision or a label that does not fit refuses
        the whole batch, so a client retrying it never leaves half of it stored."""
        records = []
        with self._lock, self._db:
            for index, (decision, outcome) in enumerate(outcomes):
                try:
                    records.append(self._apply(decision, outcome))
                except OutcomeRefused as exc:
                    raise OutcomeRefused(str(exc), f"outcomes.{index}.{exc.param}") from exc
        return records

    def _apply(self, decision: str, outcome: Outcome) -> Record:
        row = self._db.execute("SELECT request, response FROM decisions WHERE id = ?", (decision,)).fetchone()
        if row is None:
            raise UnknownDecision(decision)
        request, response = json.loads(row[0]), SystemOneResponse.model_validate_json(row[1])
        try:
            labels = outcome.labels | {qid: _corrected(request["questions"], response, qid, correction)
                                       for qid, correction in outcome.corrections.items()}
        except ValueError as exc:
            raise OutcomeRefused(str(exc), "corrections") from exc
        try:
            record = self._record(decision, request, self._labels(decision) | labels)
        except ValueError as exc:
            raise OutcomeRefused(str(exc), "labels") from exc
        self._db.executemany("INSERT OR REPLACE INTO outcomes VALUES (?, ?, ?, ?)",
                             [(decision, qid, json.dumps(label), self._now()) for qid, label in labels.items()])
        return record

    def answers_newest_first(self, page: int = 500) -> Iterator[tuple[str, dict]]:
        """Every logged decision's model and answers, newest first, read a page at a time so neither memory nor
        the lock is held for the whole ledger."""
        before = None
        while True:
            with self._lock:
                rows = self._db.execute(
                    "SELECT rowid, model, response FROM decisions WHERE ? IS NULL OR rowid < ? ORDER BY rowid DESC LIMIT ?",
                    (before, before, page)).fetchall()
            if not rows:
                return
            for _, model, response in rows:
                yield model, SystemOneResponse.model_validate_json(response).answers
            before = rows[-1][0]

    def records(self) -> Iterator[Record]:
        """Every decision with at least one outcome, oldest first, as a training record."""
        with self._lock:
            rows = self._db.execute(
                "SELECT id, request FROM decisions WHERE id IN (SELECT decision FROM outcomes) ORDER BY created"
            ).fetchall()
            labeled = [(decision, json.loads(request), self._labels(decision)) for decision, request in rows]
        for decision, request, labels in labeled:
            yield self._record(decision, request, labels)

    def _labels(self, decision: str) -> dict:
        return {qid: json.loads(label) for qid, label in
                self._db.execute("SELECT question, label FROM outcomes WHERE decision = ?", (decision,))}

    @staticmethod
    def _record(decision: str, request: dict, labels: dict) -> Record:
        return Record(id=decision, state=request["state"], questions=request["questions"], labels=labels)


def _served(question: Choice | Score | Noul, answer: ChoiceAnswer | ScoreAnswer | NoulAnswer) -> int:
    """The index of the served answer: the choice, the likeliest score level, or yes when the noul leaned yes."""
    if isinstance(answer, ChoiceAnswer):
        return option_keys(question).index(answer.choice)
    if isinstance(answer, ScoreAnswer):
        return int(max(answer.probabilities, key=answer.probabilities.get))
    return 0 if answer.noul >= 0.5 else 1


def _label(question: Choice | Score | Noul, index: int) -> str | int | bool:
    """The label naming option `index`, in the form `label_index` reads back."""
    if isinstance(question, Noul):
        return index == 0
    return index if isinstance(question, Score) else option_keys(question)[index]


def _corrected(questions: dict, response: SystemOneResponse, qid: str, correction: Correction) -> str | int | bool:
    """The label a right or wrong mark on the served answer to `qid` implies."""
    if qid not in questions:
        raise ValueError(f"{qid}: no such question in the decision")
    question = QUESTION.validate_python(questions[qid])
    answer = response.answers[qid]
    if not isinstance(answer, (ChoiceAnswer, ScoreAnswer, NoulAnswer)):
        raise ValueError(f"{qid}: a {answer.type} answer is not trained on, so it takes no correction")
    served = _served(question, answer)
    if correction.correct:
        return _label(question, served)
    if correction.answer is None:
        if len(option_keys(question)) != 2:
            raise ValueError(f"{qid}: wrong among {len(option_keys(question))} options; send the right answer")
        return _label(question, 1 - served)
    if label_index(question, correction.answer) == served:
        raise ValueError(f"{qid}: marked wrong, but {correction.answer!r} is what was served")
    return correction.answer
