"""HTTP surface: one decision endpoint plus model listing, health, and a playground page."""

from __future__ import annotations

import logging
import time
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from ruling.calibration import Calibration
from ruling.config import Settings
from ruling.decision import DecisionEngine, is_checkpoint
from ruling.engine import Engine, InputTooLong, model_released
from ruling.hosted import HOSTED_PREFIX, OPENROUTER_PREFIX, Budget, HostedEngine, OpenRouterEngine
from ruling.questions import DEFAULT_MODEL_ALIASES, ModelInfo, ModelList, SystemOneRequest, SystemOneResponse

log = logging.getLogger("ruling")
PLAYGROUND = Path(__file__).with_name("playground.html")


def build_engine(settings: Settings):
    """Any OpenAI-compatible host, OpenRouter, a trained decision checkpoint, or a local decoder."""
    calibration = Calibration.load(settings.calibration_path) if settings.calibration_path else Calibration()
    if settings.model.startswith(HOSTED_PREFIX):
        if not settings.openai_base_url:
            raise ValueError("set RULING_OPENAI_BASE_URL to the endpoint serving the model, ending in /v1")
        return HostedEngine(settings.model, calibration, settings.rotations, settings.openai_base_url,
                            settings.openai_api_key, top_logprobs=settings.openai_top_logprobs,
                            extra_body=settings.openai_extra_body)
    if settings.model.startswith(OPENROUTER_PREFIX):
        if not settings.openrouter_providers:
            raise ValueError("set RULING_OPENROUTER_PROVIDERS to the hosts allowed to serve the model")
        return OpenRouterEngine(settings.model, calibration, settings.rotations, list(settings.openrouter_providers),
                                Budget(settings.openrouter_budget_usd))
    if is_checkpoint(settings.model):
        return DecisionEngine(Path(settings.model), calibration, settings.max_input_tokens)
    return Engine(settings.model, calibration, settings.max_input_tokens, settings.rotations, settings.prior_debias,
                  settings.adapter_path, settings.max_branch_tokens)


def create_app(engine: Engine | DecisionEngine, api_key: str | None = None) -> FastAPI:
    app = FastAPI(title="ruling", version="0.2.0")
    bearer = HTTPBearer(auto_error=False)

    def authorize(credentials: HTTPAuthorizationCredentials | None = Depends(bearer)) -> None:
        if api_key is None:
            return
        if credentials is None or credentials.credentials != api_key:
            raise HTTPException(401, "invalid or missing bearer token")

    @app.post("/v1/systemone", response_model=SystemOneResponse, dependencies=[Depends(authorize)])
    def system_one(request: SystemOneRequest) -> SystemOneResponse:
        if request.model not in (None, engine.model_id, *DEFAULT_MODEL_ALIASES):
            raise HTTPException(404, f"unknown model {request.model!r}; loaded model is {engine.model_id!r}")
        started = time.perf_counter()
        try:
            response = engine.evaluate(request)
        except InputTooLong as exc:
            raise HTTPException(413, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        log.info("systemone questions=%d input_tokens=%d ms=%.0f",
                 len(request.questions), response.usage.input_tokens, (time.perf_counter() - started) * 1000)
        return response

    @app.get("/v1/models", response_model=ModelList, dependencies=[Depends(authorize)])
    def models() -> ModelList:
        return ModelList(models=[ModelInfo(
            name=engine.model_id,
            description=f"{engine.model_id} answered by ruling over {engine.rotations} answer orderings",
            release_date=model_released(engine.model_id),
            id=engine.model_id, revision=engine.revision, adapter=engine.adapter,
            max_input_tokens=engine.max_input_tokens, max_branch_tokens=engine.max_branch_tokens,
            max_options=engine.max_options, rotations=engine.rotations, prior_debias=engine.prior_debias,
        )])

    @app.get("/health")
    def health() -> dict[str, str | None]:
        return {"status": "ok", "model": engine.model_id, "revision": engine.revision}

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def playground(request: Request) -> str:
        return PLAYGROUND.read_text()

    return app


def app_from_env() -> FastAPI:
    settings = Settings.from_env()
    return create_app(build_engine(settings), settings.api_key)
