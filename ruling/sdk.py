"""Python client mirroring the shape of the hosted System One SDK."""

from __future__ import annotations

import os

import httpx

from ruling.questions import Choice, ModelInfo, ModelList, Noul, Score, State, SystemOneRequest, SystemOneResponse

DEFAULT_BASE_URL = "http://127.0.0.1:8010"


class RulingClient:
    def __init__(self, base_url: str | None = None, api_key: str | None = None, timeout: float = 60.0):
        self.base_url = base_url or os.environ.get("RULING_BASE_URL", DEFAULT_BASE_URL)
        # The server's own RULING_API_KEY, so one environment configures both ends.
        api_key = api_key or os.environ.get("RULING_API_KEY")
        headers = {"authorization": f"Bearer {api_key}"} if api_key else {}
        self._http = httpx.Client(base_url=self.base_url, headers=headers, timeout=timeout)

    def system_one(
        self, state: State, questions: dict[str, Choice | Score | Noul], model: str | None = None,
        exclude_from_training: bool = False,
    ) -> SystemOneResponse:
        request = SystemOneRequest(model=model, state=state, questions=questions,
                                   exclude_from_training=exclude_from_training)
        # The flag goes only when set, so every other body is exactly System One's.
        body = request.model_dump(exclude=None if exclude_from_training else {"exclude_from_training"})
        response = self._http.post("/v1/systemone", json=body)
        response.raise_for_status()
        return SystemOneResponse.model_validate(response.json())

    def outcome(self, decision_id: str, labels: dict) -> dict:
        """Report what turned out to be right for a logged decision; returns the labels as stored."""
        response = self._http.post(f"/v1/decisions/{decision_id}/outcome", json={"labels": labels})
        response.raise_for_status()
        return response.json()["labels"]

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
