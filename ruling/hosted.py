"""Models served behind an OpenAI-compatible chat endpoint.

Questions are rendered exactly as the local decoder engine renders them, and
the answer is read from the first token's logprobs, restricted to the option
letters, so a hosted model and a local one are directly comparable. Each
question is asked once per option ordering, all calls for a record in parallel.

`HostedEngine` speaks to any server that implements `/chat/completions` with
`top_logprobs`: vLLM, SGLang, llama.cpp, LM Studio, mlx_lm.server, or a
commercial API. `OpenRouterEngine` adds what only OpenRouter offers, provider
pinning and per-response cost, so a run can stop at a budget.
"""

from __future__ import annotations

import math
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import httpx
import numpy as np

from ruling.calibration import Calibration, Provenance
from ruling.engine import RawScore, Scored, answer
from ruling.prompt import SYSTEM_PROMPT, render_question, render_state
from ruling.evals import answer_distribution
from ruling.questions import MAX_CHOICE_OPTIONS, Choice, Noul, Score, State, SystemOneRequest, SystemOneResponse, Usage, option_keys

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_PREFIX = "openrouter:"
HOSTED_PREFIX = "openai:"
LETTERS = [chr(ord("A") + i) for i in range(20)]
TOP_LOGPROBS = 20
"""How many top logprobs a host will return, and so how many options a question may offer.

Hosts differ: OpenAI and OpenRouter allow 20, `mlx_lm.server` allows 11, vLLM is
configurable. A host that allows fewer than a question has options cannot answer
it, so the limit is reported as `max_options` rather than discovered at run time.
"""


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class Budget:
    limit_usd: float
    spent_usd: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def charge(self, usd: float) -> None:
        with self._lock:
            self.spent_usd += usd

    def check(self) -> None:
        if self.spent_usd > self.limit_usd:
            raise BudgetExceeded(f"spent ${self.spent_usd:.4f} of a ${self.limit_usd:.2f} budget")


def letter_log_probs(top_logprobs: list[dict], letters: list[str]) -> list[float]:
    """Normalized log-probabilities over `letters` from one token's top logprobs.

    Spacing variants such as " A" count as "A". A letter absent from the top
    list gets half the probability of the least likely finite token returned, an
    upper bound on its true mass, before renormalizing. All arithmetic stays in
    log space because hosts report logprobs as low as -9999.
    """
    finite = [e for e in top_logprobs if math.isfinite(e["logprob"])]
    grouped: dict[str, list[float]] = {letter: [] for letter in letters}
    for entry in finite:
        token = entry["token"].strip()
        if token in grouped:
            grouped[token].append(entry["logprob"])
    if not any(grouped.values()):
        raise ValueError("none of the answer letters appear in the top logprobs")
    floor = min(e["logprob"] for e in finite) - math.log(2)
    values = [_logsumexp(grouped[letter]) if grouped[letter] else floor for letter in letters]
    total = _logsumexp(values)
    return [v - total for v in values]


def _logsumexp(values: list[float]) -> float:
    top = max(values)
    return top + math.log(sum(math.exp(v - top) for v in values))


def first_token_top_logprobs(data: dict) -> list[dict] | None:
    """The first generated token's top logprobs, or None when the host returned none."""
    logprobs = data["choices"][0].get("logprobs")
    content = (logprobs or {}).get("content") or []
    return content[0]["top_logprobs"] if content else None


MISSING_LOGPROB_ATTEMPTS = 3
"""Hosts that advertise logprobs occasionally answer without them under load; the request is repeated this many times."""


class HostedEngine:
    """Scores typed questions with a model behind an OpenAI-compatible endpoint.

    Same interface as the local engines, so the server, the evaluation harness
    and the calibration fitter cannot tell the difference.
    """

    prior_debias = False
    prefix = HOSTED_PREFIX

    def __init__(self, model: str, calibration: Calibration, rotations: int, base_url: str,
                 api_key: str | None = None, concurrency: int = 16, top_logprobs: int = TOP_LOGPROBS,
                 extra_body: dict | None = None):
        self.remote = model.removeprefix(self.prefix)
        self.model_id = self.prefix + self.remote
        self.base_url = base_url.rstrip("/")
        self.revision = None
        self.adapter = None
        self.rotations = rotations
        self.top_logprobs = min(top_logprobs, len(LETTERS))
        # Hosts disable reasoning differently: vLLM and mlx_lm.server take
        # chat_template_kwargs, OpenRouter takes `reasoning`. Nothing is sent by
        # default, because OpenAI rejects body fields it does not know.
        self.extra_body = extra_body or {}
        self.max_options = self.top_logprobs
        # A hosted model's context is the host's business; ruling does not re-impose one.
        self.max_input_tokens = 1_000_000
        self.max_branch_tokens = self.max_input_tokens
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = httpx.Client(headers=headers, timeout=120)
        self._pool = ThreadPoolExecutor(max_workers=concurrency)
        calibration.check(self.provenance)
        self.calibration = calibration

    @property
    def provenance(self) -> Provenance:
        return Provenance(self.model_id, self.revision, self.rotations, self.prior_debias, self.adapter)

    def body(self, messages: list[dict], max_tokens: int) -> dict:
        return {"model": self.remote, "messages": messages, "max_tokens": max_tokens,
                "temperature": 0} | self.extra_body

    def charge(self, data: dict) -> None:
        """Hook for hosts that report a per-response price."""

    def complete(self, messages: list[dict], max_tokens: int, logprobs: bool) -> dict:
        body = self.body(messages, max_tokens)
        if logprobs:
            body |= {"logprobs": True, "top_logprobs": self.top_logprobs}
        response = self._client.post(f"{self.base_url}/chat/completions", json=body)
        if response.status_code != 200:
            raise RuntimeError(f"{self.base_url} returned {response.status_code} for {self.remote}: {response.text[:500]}")
        data = response.json()
        self.charge(data)
        return data

    def _ask(self, state: State, question: Choice | Score | Noul, rotation: int) -> tuple[list[str], list[float], int]:
        rendered = render_question(question, LETTERS, rotation)
        messages = [{"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": render_state(state) + rendered.text}]
        for _ in range(MISSING_LOGPROB_ATTEMPTS):
            data = self.complete(messages, max_tokens=1, logprobs=True)
            top = first_token_top_logprobs(data)
            if top is not None:
                return rendered.keys, letter_log_probs(top, rendered.codes), int(data["usage"]["prompt_tokens"])
        raise RuntimeError(f"{self.remote} returned no logprobs in {MISSING_LOGPROB_ATTEMPTS} attempts (last host {data.get('provider')})")

    def score(self, state: State, questions: dict[str, Choice | Score | Noul]) -> Scored:
        for qid, question in questions.items():
            if len(option_keys(question)) > self.max_options:
                raise ValueError(f"question {qid!r} has more than {self.max_options} options")
        jobs = [(qid, rotation) for qid, question in questions.items()
                for rotation in range(2 if isinstance(question, Score) else len(option_keys(question)))
                if rotation < self.rotations]
        results = list(self._pool.map(lambda job: (job[0], self._ask(state, questions[job[0]], job[1])), jobs))
        per_question: dict[str, dict[str, list[float]]] = {qid: {} for qid in questions}
        tokens = 0
        for qid, (keys, log_probs, prompt_tokens) in results:
            tokens += prompt_tokens
            for key, value in zip(keys, log_probs):
                per_question[qid].setdefault(key, []).append(value)
        raw = {qid: RawScore(keys=option_keys(q), logits=np.array([np.mean(per_question[qid][k]) for k in option_keys(q)]))
               for qid, q in questions.items()}
        return Scored(raw=raw, input_tokens=tokens)

    def evaluate(self, request: SystemOneRequest) -> SystemOneResponse:
        scored = self.score(request.state, request.questions)
        answers = {qid: answer(request.questions[qid], scored.raw[qid], self.calibration) for qid in request.questions}
        return SystemOneResponse(model=self.model_id, answers=answers, usage=Usage(input_tokens=scored.input_tokens))

    def answer(self, question, raw: RawScore):
        return answer(question, raw, self.calibration)

    def write(self, prompts: list[list[dict]], max_tokens: int) -> list[str]:
        """Plain text completions for several prompts in parallel, used to write questions."""
        return list(self._pool.map(lambda messages: self.complete(messages, max_tokens, logprobs=False)["choices"][0]["message"]["content"] or "",
                                   prompts))


class OpenRouterEngine(HostedEngine):
    """OpenRouter, which pins the serving provider and prices every response."""

    prefix = OPENROUTER_PREFIX

    def __init__(self, model: str, calibration: Calibration, rotations: int, providers: list[str],
                 budget: Budget, concurrency: int = 16, api_key: str | None = None):
        self.providers = providers
        self.budget = budget
        super().__init__(model, calibration, rotations, OPENROUTER_URL.removesuffix("/chat/completions"),
                         api_key or os.environ["OPENROUTER_API_KEY"], concurrency)

    def body(self, messages: list[dict], max_tokens: int) -> dict:
        self.budget.check()
        return super().body(messages, max_tokens) | {
            "reasoning": {"enabled": False}, "usage": {"include": True},
            "provider": {"only": self.providers, "require_parameters": True},
        }

    def charge(self, data: dict) -> None:
        self.budget.charge(float(data["usage"]["cost"]))


TYPESAFE_PREFIX = "typesafe:"


class TypeSafeEngine:
    """TypeSafe's hosted API, or any server that speaks its protocol, as an engine.

    The host returns probabilities, so the logits handed back are their logs and
    the host's own calibration is what you get. Useful for scoring a dataset
    against the hosted model with the same harness, and as the second stage of
    a cascade, where a local model answers what it is sure of and only the
    doubtful questions are paid for.
    """

    prior_debias = False
    max_options = MAX_CHOICE_OPTIONS
    rotations = 1
    """The host orders options as it sees fit; ruling does not re-ask it."""

    def __init__(self, model: str, calibration: Calibration, base_url: str, api_key: str | None,
                 max_input_tokens: int, max_branch_tokens: int, client: httpx.Client | None = None):
        self.remote = model.removeprefix(TYPESAFE_PREFIX)
        self.model_id = TYPESAFE_PREFIX + self.remote
        self.revision = None
        self.adapter = None
        self.max_input_tokens = max_input_tokens
        self.max_branch_tokens = max_branch_tokens
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = client or httpx.Client(base_url=base_url.rstrip("/"), headers=headers, timeout=60)
        calibration.check(self.provenance)
        self.calibration = calibration

    @property
    def provenance(self) -> Provenance:
        return Provenance(self.model_id, self.revision, self.rotations, self.prior_debias, self.adapter)

    def score(self, state: State, questions: dict[str, Choice | Score | Noul]) -> Scored:
        body = {"state": state, "questions": {qid: q.model_dump(exclude_none=True) for qid, q in questions.items()}}
        if self.remote:
            body["model"] = self.remote
        response = self._client.post("/v1/systemone", json=body)
        if response.status_code != 200:
            raise RuntimeError(f"{self.model_id} returned {response.status_code}: {response.text[:500]}")
        data = response.json()
        raw = {}
        for qid, question in questions.items():
            keys = option_keys(question)
            distribution = answer_distribution(data["answers"][qid], keys)
            raw[qid] = RawScore(keys=keys, logits=np.log(np.clip([distribution[k] for k in keys], 1e-12, 1.0)))
        return Scored(raw=raw, input_tokens=int((data.get("usage") or {}).get("input_tokens") or 0))

    def evaluate(self, request: SystemOneRequest) -> SystemOneResponse:
        scored = self.score(request.state, request.questions)
        answers = {qid: answer(request.questions[qid], scored.raw[qid], self.calibration) for qid in request.questions}
        return SystemOneResponse(model=self.model_id, answers=answers, usage=Usage(input_tokens=scored.input_tokens))

    def answer(self, question, raw: RawScore):
        return answer(question, raw, self.calibration)
