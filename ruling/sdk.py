"""Python client mirroring the shape of the hosted System One SDK."""

from __future__ import annotations

import os

import httpx

from ruling.questions import Choice, ModelInfo, ModelList, Noul, Score, State, SystemOneRequest, SystemOneResponse

DEFAULT_BASE_URL = "http://127.0.0.1:8010"


class RulingClient:
    def __init__(self, base_url: str | None = None, timeout: float = 60.0):
        self.base_url = base_url or os.environ.get("RULING_BASE_URL", DEFAULT_BASE_URL)
        self._http = httpx.Client(base_url=self.base_url, timeout=timeout)

    def system_one(
        self, state: State, questions: dict[str, Choice | Score | Noul], model: str | None = None
    ) -> SystemOneResponse:
        request = SystemOneRequest(model=model, state=state, questions=questions)
        response = self._http.post("/v1/systemone", json=request.model_dump())
        response.raise_for_status()
        return SystemOneResponse.model_validate(response.json())

    def models(self) -> list[ModelInfo]:
        response = self._http.get("/v1/models")
        response.raise_for_status()
        return ModelList.model_validate(response.json()).models

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> "RulingClient":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
