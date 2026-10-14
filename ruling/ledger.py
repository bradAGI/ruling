"""Every decision a server answers, and the outcomes reported for it later, kept on disk.

An outcome is what turned out to be right for a question, reported by whoever
learns it: the ticket's final team, whether the refund was granted. Decisions
with outcomes are training records, so a model can be retrained on its own
traffic with `ruling export-decisions` and `ruling train`.

SQLite keeps the ledger on disk and indexed, so lookups and memory stay flat
however much traffic it holds.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from collections.abc import Iterator
from pathlib import Path

from pydantic import BaseModel, Field

from ruling.evals import Label, Record
from ruling.questions import SystemOneRequest, SystemOneResponse

SCHEMA = """
CREATE TABLE IF NOT EXISTS decisions (
    id TEXT PRIMARY KEY,
    created REAL NOT NULL,
    model TEXT NOT NULL,
    request TEXT NOT NULL,
    response TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS outcomes (
    decision TEXT NOT NULL REFERENCES decisions(id),
    question TEXT NOT NULL,
    label TEXT NOT NULL,
    reported REAL NOT NULL,
    PRIMARY KEY (decision, question)
);
"""


class Outcome(BaseModel):
    """What turned out to be right, per question, in the label forms a training record takes."""

    labels: dict[str, Label] = Field(min_length=1)


class UnknownDecision(KeyError):
    def __str__(self) -> str:
        return f"no decision {self.args[0]!r} in the ledger"


class DecisionLedger:
    def __init__(self, path: Path):
        # Server requests arrive on a thread pool; one connection behind a lock serves them all.
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.executescript(SCHEMA)
        self._lock = threading.Lock()

    def record(self, request: SystemOneRequest, response: SystemOneResponse) -> str:
        """Store one answered request and return the id outcomes will refer to."""
        decision = uuid.uuid4().hex
        with self._lock, self._db:
            self._db.execute("INSERT INTO decisions VALUES (?, ?, ?, ?, ?)",
                             (decision, time.time(), response.model, request.model_dump_json(exclude_none=True),
                              response.model_dump_json(exclude_none=True)))
        return decision

    def outcome(self, decision: str, labels: dict) -> Record:
        """Attach outcomes to a decision; a later outcome for the same question replaces the earlier one.

        The labels are checked as a training record would check them, so nothing
        stored here can fail later in `ruling train`. Returns the record as it now stands.
        """
        with self._lock, self._db:
            row = self._db.execute("SELECT request FROM decisions WHERE id = ?", (decision,)).fetchone()
            if row is None:
                raise UnknownDecision(decision)
            record = self._record(decision, json.loads(row[0]), self._labels(decision) | labels)
            self._db.executemany("INSERT OR REPLACE INTO outcomes VALUES (?, ?, ?, ?)",
                                 [(decision, qid, json.dumps(label), time.time()) for qid, label in labels.items()])
        return record

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
