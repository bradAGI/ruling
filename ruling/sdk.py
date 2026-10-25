"""Python client mirroring the shape of the hosted System One SDK, plus ruling's batch, evaluation and usage routes."""

from __future__ import annotations

import base64
import gzip
import json
import os
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from asyncio import sleep as asleep
from collections.abc import Callable
from time import sleep

import httpx
from pydantic import TypeAdapter

from ruling.ask import AskResponse
from ruling.ledger import IDEMPOTENCY_HEADER, Outcome, OutcomeBatch
from ruling.questions import (
    IMAGE_SUFFIXES,
    Answer,
    Box,
    Choice,
    ModelInfo,
    ModelList,
    Noul,
    Score,
    State,
    SystemOneRequest,
    SystemOneResponse,
)
from ruling.questions import Image as ImageBlock

DEFAULT_BASE_URL = "http://127.0.0.1:8010"

GZIP_THRESHOLD = 64 * 1024
"""Bodies of at least this many bytes are sent gzipped. JSON rows compress several times over, and
below this size the round trip is dominated by the model, not the bytes."""

RETRIED_STATUSES = (429, 503)
"""A full queue and an absent upstream are the server's two transient refusals."""

ANSWER = TypeAdapter(Answer)


@dataclass(frozen=True)
class RetryPolicy:
    """How a request that was turned away or never arrived is tried again: at most `max_retries`
    more times, waiting the server's `Retry-After` when it gives one, else doubling from
    `backoff_initial` seconds up to `backoff_max`."""

    max_retries: int = 2
    backoff_initial: float = 0.5
    backoff_max: float = 5.0

    def delay(self, attempt: int, retry_after: str | None) -> float:
        if retry_after is not None:
            return float(retry_after)
        return min(self.backoff_initial * 2**attempt, self.backoff_max)


class RulingError(Exception):
    """A refusal from the server: its HTTP `status` and the `code`, `message` and `param` of its error body."""

    def __init__(self, status: int, code: str, message: str, param: str | None = None):
        super().__init__(f"{status} {code}: {message}" + (f" (param {param})" if param else ""))
        self.status, self.code, self.message, self.param = status, code, message, param

    @classmethod
    def from_response(cls, response: httpx.Response) -> "RulingError":
        try:
            error = response.json()["error"]
        except (ValueError, KeyError, TypeError):
            # Something between the client and ruling answered, such as a proxy's HTML error page.
            return cls(response.status_code, "http_error", response.text)
        return cls(response.status_code, error["code"], error["message"], error.get("param"))


class Image:
    """Image blocks for a state, in the wire shape the server reads."""

    @staticmethod
    def from_bytes(data: bytes, media_type: str) -> dict:
        block = ImageBlock(type="image", media_type=media_type, data=base64.b64encode(data).decode())
        return block.model_dump()

    @staticmethod
    def from_path(path: str | os.PathLike) -> dict:
        path = Path(path)
        try:
            media_type = IMAGE_SUFFIXES[path.suffix.lower()]
        except KeyError:
            raise ValueError(f"{path.name}: an image must be one of {', '.join(IMAGE_SUFFIXES)}, "
                             f"got {path.suffix or 'no suffix'}") from None
        return Image.from_bytes(path.read_bytes(), media_type)


@dataclass(frozen=True)
class _Call:
    """One request to the server and how its JSON body becomes what the caller gets back."""

    method: str
    path: str
    body: dict | None = None
    params: dict | None = None
    headers: dict | None = None
    parse: Callable[[dict], object] = lambda body: body


class _Endpoints:
    """Every route the server offers, each described once as a `_Call`; a client runs the calls on its own
    transport, so the blocking and the async client answer alike."""

    def __init__(self, base_url: str | None, api_key: str | None, retry: RetryPolicy | None, gzip: bool):
        self.base_url = base_url or os.environ.get("RULING_BASE_URL", DEFAULT_BASE_URL)
        # The server's own RULING_API_KEY, so one environment configures both ends.
        api_key = api_key or os.environ.get("RULING_API_KEY")
        self._auth = {"authorization": f"Bearer {api_key}"} if api_key else {}
        self.retry = retry if retry is not None else RetryPolicy()
        self.gzip = gzip

    def _run(self, call: _Call):
        raise NotImplementedError

    def system_one(
        self, state: State, questions: dict[str, Choice | Score | Noul | Box], model: str | None = None,
        exclude_from_training: bool = False, assets: dict[str, dict] | None = None,
    ) -> SystemOneResponse:
        """A decision. `assets` names images the text mentions, `{"<photo>": Image.from_path(...)}`, which the
        server puts in the state as `<image N>`; a `Box` question asks where something is in one of them."""
        request = SystemOneRequest(model=model, state=state, questions=questions, assets=assets,
                                   exclude_from_training=exclude_from_training)
        # Unset fields stay off the wire, so a body without the flag or assets is exactly System One's.
        body = request.model_dump(mode="json", exclude_none=True,
                                  exclude=None if exclude_from_training else {"exclude_from_training"})
        # One key for every attempt, so a retry of a decision the server already logged is not logged twice.
        return self._run(_Call("POST", "/v1/systemone", body, headers={IDEMPOTENCY_HEADER: uuid.uuid4().hex},
                               parse=SystemOneResponse.model_validate))

    def ask(self, text: str, model: str | None = None, images: list[dict] | None = None) -> AskResponse:
        """A question in plain words: the model's readings of it, each answer with what was said, and the work."""
        body = {"model": model, "input": text, **({"images": images} if images else {})}
        return self._run(_Call("POST", "/v1/ask", body, parse=AskResponse.model_validate))

    def predict(self, rows: list[dict], model: str | None = None) -> list[dict]:
        """Every row answered in order: `{"id"?, "answers": {id: Answer}}`, or `{"id"?, "error": {...}}` for
        a row the server refused on its own."""
        return self._run(_Call("POST", "/v1/predict", {"model": model, "rows": rows}, parse=lambda body: [
            {**entry, "answers": {qid: ANSWER.validate_python(a) for qid, a in entry["answers"].items()}}
            if "answers" in entry else entry
            for entry in body["answers"]]))

    def evaluate(self, rows: list[dict], model: str | None = None) -> dict:
        """The labeled rows scored: the report `ruling eval` prints, with the answering model's name."""
        return self._run(_Call("POST", "/v1/evaluations", {"model": model, "rows": rows}))

    def usage(self, starting_on: datetime | None = None, ending_before: datetime | None = None) -> list[dict]:
        """Hourly buckets per model since the server started, within the window when one is given."""
        params = {name: moment.isoformat() for name, moment in
                  (("starting_on", starting_on), ("ending_before", ending_before)) if moment is not None}
        return self._run(_Call("GET", "/v1/usage", params=params, parse=lambda body: body["usage"]))

    def outcome(self, decision_id: str, labels: dict | None = None, corrections: dict | None = None) -> dict:
        """Report what turned out to be right for a logged decision: `labels` name it, `corrections` mark the
        served answers, `{"team": {"correct": False, "answer": "billing"}}`. Returns the labels as stored."""
        body = Outcome(labels=labels or {}, corrections=corrections or {}).model_dump(mode="json", exclude_defaults=True)
        return self._run(_Call("POST", f"/v1/decisions/{decision_id}/outcome", body, parse=lambda b: b["labels"]))

    def outcomes(self, outcomes: list[dict]) -> list[dict]:
        """Outcomes for many decisions, `[{"id", "labels"?, "corrections"?}]`, stored all or none; returns each
        decision's labels as stored."""
        body = OutcomeBatch(outcomes=outcomes).model_dump(mode="json", exclude_defaults=True)
        return self._run(_Call("POST", "/v1/decisions/outcomes", body, parse=lambda b: b["outcomes"]))

    def models(self) -> list[ModelInfo]:
        return self._run(_Call("GET", "/v1/models", parse=lambda body: ModelList.model_validate(body).models))

    def _prepared(self, call: _Call) -> tuple[bytes | None, dict[str, str]]:
        """The call's body as sent, gzipped above the threshold, and its headers."""
        if call.body is None:
            return None, call.headers or {}
        content = json.dumps(call.body).encode()
        headers = {"content-type": "application/json", **(call.headers or {})}
        if self.gzip and len(content) >= GZIP_THRESHOLD:
            content = gzip.compress(content)
            headers["content-encoding"] = "gzip"
        return content, headers

    def _backoff(self, attempt: int, response: httpx.Response | None) -> float | None:
        """How long to wait before trying again after `attempt`, or None to stop: a transient refusal or a
        request that never arrived (`response` None) is tried again until the policy runs out."""
        if attempt == self.retry.max_retries:
            return None
        if response is None:
            return self.retry.delay(attempt, None)
        if response.status_code in RETRIED_STATUSES:
            return self.retry.delay(attempt, response.headers.get("retry-after"))
        return None

    @staticmethod
    def _answer(call: _Call, response: httpx.Response):
        if response.is_error:
            raise RulingError.from_response(response)
        return call.parse(response.json())


class RulingClient(_Endpoints):
    """The server's routes as blocking calls."""

    def __init__(self, base_url: str | None = None, api_key: str | None = None, timeout: float = 60.0,
                 retry: RetryPolicy | None = None, gzip: bool = True,
                 transport: httpx.BaseTransport | None = None):
        super().__init__(base_url, api_key, retry, gzip)
        self._http = httpx.Client(base_url=self.base_url, headers=self._auth, timeout=timeout, transport=transport)

    def _run(self, call: _Call):
        content, headers = self._prepared(call)
        attempt = 0
        while True:
            try:
                response = self._http.request(call.method, call.path, content=content, headers=headers,
                                              params=call.params)
            except httpx.TransportError:
                wait = self._backoff(attempt, None)
                if wait is None:
                    raise
            else:
                wait = self._backoff(attempt, response)
                if wait is None:
                    return self._answer(call, response)
            sleep(wait)
            attempt += 1

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> "RulingClient":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class AsyncRulingClient(_Endpoints):
    """The server's routes as coroutines: every method of `RulingClient`, awaited, so many decisions can be in
    flight from one event loop."""

    def __init__(self, base_url: str | None = None, api_key: str | None = None, timeout: float = 60.0,
                 retry: RetryPolicy | None = None, gzip: bool = True,
                 transport: httpx.AsyncBaseTransport | None = None):
        super().__init__(base_url, api_key, retry, gzip)
        self._http = httpx.AsyncClient(base_url=self.base_url, headers=self._auth, timeout=timeout,
                                       transport=transport)

    async def _run(self, call: _Call):
        content, headers = self._prepared(call)
        attempt = 0
        while True:
            try:
                response = await self._http.request(call.method, call.path, content=content, headers=headers,
                                                    params=call.params)
            except httpx.TransportError:
                wait = self._backoff(attempt, None)
                if wait is None:
                    raise
            else:
                wait = self._backoff(attempt, response)
                if wait is None:
                    return self._answer(call, response)
            await asleep(wait)
            attempt += 1

    async def aclose(self) -> None:
        await self._http.aclose()

    async def __aenter__(self) -> "AsyncRulingClient":
        return self

    async def __aexit__(self, *exc) -> None:
        await self.aclose()
