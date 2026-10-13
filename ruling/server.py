"""HTTP surface: one decision endpoint over a registry of models, model listing, health, and a playground page."""

from __future__ import annotations

import logging
import math
import threading
import time
from pathlib import Path

from fastapi import Depends, FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from ruling.config import Settings
from ruling.engine import InputTooLong
from ruling.hosted import HostUnavailable
from ruling.questions import InvalidOptions, ModelList, SystemOneRequest, SystemOneResponse, check_askable
from ruling.registry import Registry, UnknownModel

log = logging.getLogger("ruling")
PLAYGROUND = Path(__file__).with_name("playground.html")

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


def create_app(registry: Registry, api_key: str | None, max_pending: int) -> FastAPI:
    app = FastAPI(title="ruling", version="0.2.0")
    bearer = HTTPBearer(auto_error=False)
    admission = Admission(max_pending)

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

    @app.post("/v1/systemone", response_model=SystemOneResponse, dependencies=[Depends(authorize)])
    async def system_one(request: SystemOneRequest) -> SystemOneResponse:
        try:
            name, engine = registry.resolve(request.model)
        except UnknownModel as exc:
            raise ApiError(404, "unknown_model", str(exc), "model") from exc
        try:
            check_askable(request.questions, engine.max_options)
        except InvalidOptions as exc:
            raise ApiError(422, "invalid_options", str(exc), f"questions.{exc.question_id}") from exc
        if not admission.enter():
            raise ApiError(429, "queue_full", f"{admission.limit} requests are already in flight",
                           retry_after=admission.retry_after())
        started = time.perf_counter()
        try:
            response = await run_in_threadpool(engine.evaluate, request)
        except InputTooLong as exc:
            param = f"questions.{exc.question_id}" if exc.question_id is not None else None
            raise ApiError(422, "context_too_long", str(exc), param) from exc
        except InvalidOptions as exc:
            raise ApiError(422, "invalid_options", str(exc), f"questions.{exc.question_id}") from exc
        except ValueError as exc:
            raise ApiError(422, "invalid_input", str(exc)) from exc
        except HostUnavailable as exc:
            raise ApiError(503, "upstream_unavailable", str(exc)) from exc
        finally:
            elapsed = time.perf_counter() - started
            admission.leave(elapsed)
        log.info("systemone model=%s questions=%d input_tokens=%d ms=%.0f",
                 name, len(request.questions), response.usage.input_tokens, elapsed * 1000)
        # An entry answers under its own name; the primary keeps the engine's report, which a
        # failover changes to say which side answered.
        return response if name == registry.default else response.model_copy(update={"model": name})

    @app.get("/v1/models", response_model=ModelList, dependencies=[Depends(authorize)])
    def models() -> ModelList:
        return ModelList(models=registry.info())

    @app.get("/health")
    def health() -> dict[str, str | None]:
        engine = registry.engines[registry.default]
        return {"status": "ok", "model": registry.default, "revision": engine.revision}

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def playground(request: Request) -> str:
        return PLAYGROUND.read_text()

    return app


def app_from_env() -> FastAPI:
    settings = Settings.from_env()
    return create_app(Registry.from_settings(settings), settings.api_key, settings.max_pending)
