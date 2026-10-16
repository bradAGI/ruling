"""HTTP surface: decisions one at a time or in batches over a registry of models, scored evaluations,
usage counts, model listing, health, and a playground page."""

from __future__ import annotations

import gzip
import logging
import math
import threading
import time
import zlib
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field, ValidationError
from starlette.datastructures import Headers
from starlette.types import ASGIApp, Receive, Scope, Send

from ruling.config import Settings
from ruling.engine import InputTooLong
from ruling.evals import Record, as_record, collect, predict, report
from ruling.hosted import HostUnavailable
from ruling.ledger import DecisionLedger, Outcome, UnknownDecision
from ruling.questions import (
    MAX_PREDICT_ROWS,
    InvalidOptions,
    ModelList,
    SystemOneRequest,
    SystemOneResponse,
    UnsupportedModality,
    check_askable,
    check_readable,
)
from ruling.registry import Registry, UnknownModel
from ruling.retrain import CURRENT, history
from ruling.usage import Metered, UsageMeter

log = logging.getLogger("ruling")
PLAYGROUND = Path(__file__).with_name("playground.html")
HISTORY = Path(__file__).with_name("history.html")

# Weight of the latest request in the running estimate of how long one takes, which
# sizes the Retry-After a turned-away client is given.
LATENCY_SMOOTHING = 0.2


class ApiError(Exception):
    """An error in the hosted API's shape: a stable `code` for programs, a `message` for people."""

    def __init__(self, status: int, code: str, message: str, param: str | None = None, retry_after: int | None = None):
        super().__init__(message)
        self.status, self.code, self.message, self.param, self.retry_after = status, code, message, param, retry_after

    def response(self) -> JSONResponse:
        error = {"code": self.code, "message": self.message}
        if self.param is not None:
            error["param"] = self.param
        headers = {"Retry-After": str(self.retry_after)} if self.retry_after is not None else None
        return JSONResponse({"error": error}, status_code=self.status, headers=headers)


class Admission:
    """How many requests may be in flight at once, and how long a refused one should wait."""

    def __init__(self, limit: int):
        if limit < 1:
            raise ValueError("at least one request must be admitted")
        self.limit = limit
        self.pending = 0
        self.seconds_per_request: float | None = None
        self._lock = threading.Lock()

    def enter(self) -> bool:
        with self._lock:
            if self.pending >= self.limit:
                return False
            self.pending += 1
            return True

    def leave(self, elapsed: float) -> None:
        with self._lock:
            self.pending -= 1
            if self.seconds_per_request is None:
                self.seconds_per_request = elapsed
            else:
                self.seconds_per_request += LATENCY_SMOOTHING * (elapsed - self.seconds_per_request)

    def retry_after(self) -> int:
        """Whole seconds until the queue has likely drained, never less than one."""
        with self._lock:
            estimate = self.pending * (self.seconds_per_request or 0.0)
        return max(1, math.ceil(estimate))

    @contextmanager
    def slot(self) -> Iterator[Callable[[], float]]:
        """Hold one place in the queue for the block, or refuse with a wait; yields the elapsed time so far."""
        if not self.enter():
            raise ApiError(429, "queue_full", f"{self.limit} requests are already in flight",
                           retry_after=self.retry_after())
        started = time.perf_counter()

        def elapsed() -> float:
            return time.perf_counter() - started

        try:
            yield elapsed
        finally:
            self.leave(elapsed())


class GzipRequests:
    """Inflates a request body sent with `Content-Encoding: gzip`, so a client can shrink a large batch."""

    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or Headers(scope=scope).get("content-encoding") != "gzip":
            await self.app(scope, receive, send)
            return
        packed = b""
        while True:
            message = await receive()
            packed += message.get("body", b"")
            if not message.get("more_body"):
                break
        try:
            body = gzip.decompress(packed)
        except (gzip.BadGzipFile, EOFError, zlib.error) as exc:
            await ApiError(422, "invalid_input", f"the body is not valid gzip: {exc}").response()(scope, receive, send)
            return
        headers = [(k, v) for k, v in scope["headers"] if k not in (b"content-encoding", b"content-length")]
        headers.append((b"content-length", str(len(body)).encode()))

        async def receive_inflated() -> dict:
            return {"type": "http.request", "body": body, "more_body": False}

        await self.app({**scope, "headers": headers}, receive_inflated, send)


class BatchRequest(BaseModel):
    """Rows in any shape `ruling.evals` reads for `ruling predict`; a row's shape is checked row by row."""

    model: str | None = None
    rows: list[Any] = Field(min_length=1, max_length=MAX_PREDICT_ROWS)


def parse_row(row: Any) -> Record:
    if not isinstance(row, dict):
        raise ValueError("a row must be an object")
    return Record.model_validate(as_record(row))


def row_error(exc: ValueError) -> dict[str, str]:
    """The API's code for what refused a row. A schema failure is reported by its first problem; one a
    record's own validator raised, such as a one-option choice, keeps the code that error carries."""
    if isinstance(exc, ValidationError):
        first = exc.errors()[0]
        cause = first.get("ctx", {}).get("error")
        if isinstance(cause, ValueError):
            return row_error(cause)
        where = ".".join(str(part) for part in first["loc"])
        return {"code": "invalid_input", "message": f"{where}: {first['msg']}" if where else first["msg"]}
    if isinstance(exc, InputTooLong):
        return {"code": "context_too_long", "message": str(exc)}
    if isinstance(exc, InvalidOptions):
        return {"code": "invalid_options", "message": str(exc)}
    if isinstance(exc, UnsupportedModality):
        return {"code": "unsupported_modality", "message": str(exc)}
    return {"code": "invalid_input", "message": str(exc)}


def answer_rows(engine: Metered, rows: list[Any]) -> list[dict]:
    """One entry per row in order: its answers, or the error that refused that row alone."""
    entries: list[dict | None] = [None] * len(rows)
    records, places = [], []
    for index, row in enumerate(rows):
        try:
            records.append(parse_row(row))
            places.append(index)
        except ValueError as exc:
            named = isinstance(row, dict) and row.get("id") is not None
            entries[index] = {**({"id": row["id"]} if named else {}), "error": row_error(exc)}
    for index, line in zip(places, predict(engine, records)):
        entries[index] = line
    return entries


def labeled_records(engine, rows: list[Any]) -> list[Record]:
    """Every row as a record the engine can score, or the 422 naming the first row it cannot."""
    records = []
    for index, row in enumerate(rows):
        try:
            record = parse_row(row)
            check_askable(record.questions, engine.max_options)
            check_readable(record.state, engine.reads_images)
        except ValueError as exc:
            raise ApiError(422, param=f"rows.{index}", **row_error(exc)) from exc
        records.append(record)
    if not any(record.labels for record in records):
        raise ApiError(422, "invalid_input", "no row carries labels; nothing to score against", "rows")
    return records


def create_app(registry: Registry, api_key: str | None, max_pending: int,
               ledger: DecisionLedger | None = None, usage: UsageMeter | None = None) -> FastAPI:
    app = FastAPI(title="ruling", version="0.2.0")
    app.add_middleware(GzipRequests)
    bearer = HTTPBearer(auto_error=False)
    admission = Admission(max_pending)
    usage = usage if usage is not None else UsageMeter()

    def resolve(model: str | None):
        try:
            return registry.resolve(model)
        except UnknownModel as exc:
            raise ApiError(404, "unknown_model", str(exc), "model") from exc

    @contextmanager
    def engine_errors() -> Iterator[None]:
        """What an engine raises while answering, in the API's shape."""
        try:
            yield
        except InputTooLong as exc:
            param = f"questions.{exc.question_id}" if exc.question_id is not None else None
            raise ApiError(422, "context_too_long", str(exc), param) from exc
        except InvalidOptions as exc:
            raise ApiError(422, "invalid_options", str(exc), f"questions.{exc.question_id}") from exc
        except ValueError as exc:
            raise ApiError(422, "invalid_input", str(exc)) from exc
        except HostUnavailable as exc:
            raise ApiError(503, "upstream_unavailable", str(exc)) from exc

    def authorize(credentials: HTTPAuthorizationCredentials | None = Depends(bearer)) -> None:
        if api_key is None:
            return
        if credentials is None or credentials.credentials != api_key:
            raise ApiError(401, "unauthorized", "invalid or missing bearer token")

    @app.exception_handler(ApiError)
    async def api_error(request: Request, exc: ApiError) -> JSONResponse:
        return exc.response()

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request: Request, exc: RequestValidationError) -> JSONResponse:
        first, *rest = exc.errors()
        param = ".".join(str(part) for part in first["loc"][1:]) or None
        message = first["msg"] + (f" (and {len(rest)} more)" if rest else "")
        return ApiError(422, "invalid_input", message, param).response()

    @app.post("/v1/systemone", response_model=SystemOneResponse, response_model_exclude_none=True,
              dependencies=[Depends(authorize)])
    async def system_one(request: SystemOneRequest) -> SystemOneResponse:
        name, engine = resolve(request.model)
        try:
            check_askable(request.questions, engine.max_options)
        except InvalidOptions as exc:
            raise ApiError(422, "invalid_options", str(exc), f"questions.{exc.question_id}") from exc
        try:
            check_readable(request.state, engine.reads_images)
        except UnsupportedModality as exc:
            raise ApiError(422, "unsupported_modality", f"{name}: {exc}", "state") from exc
        with admission.slot() as elapsed, engine_errors():
            response = await run_in_threadpool(engine.evaluate, request)
        usage.record(name, 1, response.usage.input_tokens)
        log.info("systemone model=%s questions=%d input_tokens=%d ms=%.0f",
                 name, len(request.questions), response.usage.input_tokens, elapsed() * 1000)
        # An entry answers under its own name; the primary keeps the engine's report, which a
        # failover changes to say which side answered.
        if name != registry.default:
            response = response.model_copy(update={"model": name})
        if ledger is not None and not request.exclude_from_training:
            response = response.model_copy(update={"id": await run_in_threadpool(ledger.record, request, response)})
        return response

    @app.post("/v1/predict", dependencies=[Depends(authorize)])
    async def predict_rows(request: BatchRequest) -> dict:
        """Every row answered in order; a row the model refuses carries its error and the batch goes on."""
        name, engine = resolve(request.model)
        metered = Metered(engine)
        with admission.slot() as elapsed, engine_errors():
            entries = await run_in_threadpool(answer_rows, metered, request.rows)
        answered = sum("answers" in entry for entry in entries)
        usage.record(name, answered, metered.input_tokens)
        log.info("predict model=%s rows=%d answered=%d input_tokens=%d ms=%.0f",
                 name, len(entries), answered, metered.input_tokens, elapsed() * 1000)
        return {"model": name, "answers": entries, "usage": {"input_tokens": metered.input_tokens, "output_tokens": 0}}

    @app.post("/v1/evaluations", dependencies=[Depends(authorize)])
    async def evaluations(request: BatchRequest) -> dict:
        """The rows scored against their labels: exactly the report `ruling eval` prints."""
        name, engine = resolve(request.model)
        metered = Metered(engine)
        with admission.slot() as elapsed, engine_errors():
            records = await run_in_threadpool(labeled_records, engine, request.rows)
            observations = await run_in_threadpool(collect, metered, records)
        scored = len({o.record for o in observations if o.logits is not None})
        usage.record(name, scored, metered.input_tokens)
        log.info("evaluations model=%s rows=%d scored=%d input_tokens=%d ms=%.0f",
                 name, len(records), scored, metered.input_tokens, elapsed() * 1000)
        return {"model": name, **report(observations, engine.calibration)}

    @app.get("/v1/usage", dependencies=[Depends(authorize)])
    def usage_report(starting_on: datetime | None = None, ending_before: datetime | None = None) -> dict:
        """Hourly counts per model since the server started, kept in memory only."""
        return {"usage": usage.buckets(starting_on, ending_before)}

    if ledger is not None:
        @app.post("/v1/decisions/{decision}/outcome", dependencies=[Depends(authorize)])
        async def outcome(decision: str, body: Outcome) -> dict:
            try:
                record = await run_in_threadpool(ledger.outcome, decision, body.labels)
            except UnknownDecision as exc:
                raise ApiError(404, "unknown_decision", str(exc)) from exc
            except ValueError as exc:
                raise ApiError(422, "invalid_input", str(exc), "labels") from exc
            return {"id": decision, "labels": record.labels}

    @app.get("/v1/models", response_model=ModelList, dependencies=[Depends(authorize)])
    def models() -> ModelList:
        return ModelList(models=registry.info())

    @app.get("/v1/retrain", dependencies=[Depends(authorize)])
    def retraining() -> dict:
        """Each served model whose adapter is a `ruling retrain` directory's `current` link, with its versions."""
        return {"models": [{"name": name, **history(engine.adapter_path.parent)}
                           for name, engine in registry.engines.items()
                           if getattr(engine, "adapter_path", None) is not None and engine.adapter_path.name == CURRENT]}

    @app.get("/health")
    def health() -> dict[str, str | None]:
        engine = registry.engines[registry.default]
        return {"status": "ok", "model": registry.default, "revision": engine.revision}

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def playground(request: Request) -> str:
        return PLAYGROUND.read_text()

    @app.get("/history", response_class=HTMLResponse, include_in_schema=False)
    def history_page() -> str:
        return HISTORY.read_text()

    return app


def app_from_env() -> FastAPI:
    settings = Settings.from_env()
    retain = settings.ledger_retain_days * 86_400 if settings.ledger_retain_days is not None else None
    ledger = DecisionLedger(settings.ledger_path, retain) if settings.ledger_path else None
    return create_app(Registry.from_settings(settings), settings.api_key, settings.max_pending, ledger)
