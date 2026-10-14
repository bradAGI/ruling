"""Which models one server answers for: the environment's primary, plus named entries from a file.

Entries that fine-tune the same base share its loaded weights: the LoRA layers
are swapped under the engine lock before each request, so a server can hold a
plain base, a tuned adapter per decision, and a large escalation model at the
cost of one copy of each base.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, replace
from pathlib import Path

from ruling.against import API_KEY_ENV, BASE_URL_ENV, DEFAULT_BASE_URL
from ruling.calibration import Calibration
from ruling.cascade import CascadeEngine
from ruling.config import Settings
from ruling.decision import DecisionEngine, is_checkpoint
from ruling.engine import Engine, model_released
from ruling.failover import FailoverEngine
from ruling.hosted import (
    HOSTED_PREFIX,
    OPENROUTER_PREFIX,
    TYPESAFE_PREFIX,
    VLLM_PREFIX,
    Budget,
    HostedEngine,
    OpenRouterEngine,
    TypeSafeEngine,
    VllmEngine,
)
from ruling.questions import DEFAULT_MODEL_ALIASES, ModelInfo

ENTRY_FIELDS = ("model", "adapter", "calibration")


class UnknownModel(KeyError):
    def __init__(self, name: str, known: list[str]):
        super().__init__(name)
        self.name = name
        self.known = known

    def __str__(self) -> str:
        return f"no model {self.name!r}; this server answers for {', '.join(repr(k) for k in self.known)}"


@dataclass(frozen=True)
class Entry:
    model: str
    adapter: Path | None = None
    calibration: Path | None = None

    @classmethod
    def parse(cls, name: str, raw: object, relative_to: Path) -> "Entry":
        if not isinstance(raw, dict) or "model" not in raw:
            raise ValueError(f"model {name!r} must be an object with at least a 'model' field")
        unknown = set(raw) - set(ENTRY_FIELDS)
        if unknown:
            raise ValueError(f"model {name!r} has unknown fields {sorted(unknown)}; allowed: {list(ENTRY_FIELDS)}")
        paths = {key: relative_to / raw[key] for key in ("adapter", "calibration") if raw.get(key)}
        return cls(model=str(raw["model"]), **paths)


def load_entries(path: Path) -> dict[str, Entry]:
    """Named entries from a JSON object; relative paths resolve against the file's directory."""
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"{path} must hold a JSON object of model name to entry")
    entries = {}
    for name, raw in data.items():
        if not name or name in DEFAULT_MODEL_ALIASES:
            raise ValueError(f"{name!r} cannot name a model; it is empty or a default alias")
        entries[name] = Entry.parse(name, raw, path.parent)
    return entries


def build_engine(settings: Settings):
    """Any OpenAI-compatible host, OpenRouter, a trained decision checkpoint, or a local decoder.

    With `cascade_to` set, that model answers whichever questions the first one is
    unsure about, judged against `cascade_threshold`. With `failover_to` set, that
    model answers whenever the first one is unavailable.
    """
    if settings.failover_to:
        primary = build_engine(replace(settings, failover_to=None))
        fallback = build_engine(replace(settings, model=settings.failover_to, failover_to=None, cascade_to=None,
                                        adapter_path=None, calibration_path=None))
        return FailoverEngine(primary, fallback)
    if settings.cascade_to:
        if settings.cascade_threshold is None:
            raise ValueError("set RULING_CASCADE_THRESHOLD, the top probability below which a question is escalated")
        first = build_engine(replace(settings, cascade_to=None))
        second = build_engine(replace(settings, model=settings.cascade_to, cascade_to=None, adapter_path=None,
                                      calibration_path=None))
        return CascadeEngine(first, second, settings.cascade_threshold)
    calibration = Calibration.load(settings.calibration_path) if settings.calibration_path else Calibration()
    if settings.model.startswith(TYPESAFE_PREFIX):
        return TypeSafeEngine(settings.model, calibration, os.environ.get(BASE_URL_ENV, DEFAULT_BASE_URL),
                              os.environ.get(API_KEY_ENV), settings.max_input_tokens, settings.max_branch_tokens)
    if settings.model.startswith(VLLM_PREFIX):
        if not settings.vllm_base_url:
            raise ValueError("set RULING_VLLM_BASE_URL to the vLLM server, without a path")
        return VllmEngine(settings.model, calibration, settings.vllm_base_url, settings.vllm_api_key,
                          settings.max_input_tokens, settings.max_branch_tokens)
    if settings.model.startswith(HOSTED_PREFIX):
        if not settings.openai_base_url:
            raise ValueError("set RULING_OPENAI_BASE_URL to the endpoint serving the model, ending in /v1")
        return HostedEngine(settings.model, calibration, settings.rotations, settings.openai_base_url,
                            settings.openai_api_key, concurrency=settings.openai_concurrency,
                            top_logprobs=settings.openai_top_logprobs,
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


class Registry:
    """The engines a server answers for, by the name a request sends as `model`."""

    def __init__(self, default: str, engines: dict[str, object]):
        if default not in engines:
            raise ValueError(f"default model {default!r} is not among {sorted(engines)}")
        self.default = default
        self.engines = engines

    @classmethod
    def single(cls, engine) -> "Registry":
        return cls(engine.model_id, {engine.model_id: engine})

    @classmethod
    def from_settings(cls, settings: Settings) -> "Registry":
        primary = build_engine(settings)
        engines: dict[str, object] = {primary.model_id: primary}
        if settings.models_path:
            for name, entry in load_entries(settings.models_path).items():
                if name in engines:
                    raise ValueError(f"model {name!r} is both the primary model and an entry in {settings.models_path}")
                engines[name] = _build_entry(entry, settings, engines)
        return cls(primary.model_id, engines)

    def resolve(self, name: str | None) -> tuple[str, object]:
        """The entry name and engine a request's `model` field selects; None and the aliases mean the default."""
        if name is None or name in DEFAULT_MODEL_ALIASES:
            name = self.default
        try:
            return name, self.engines[name]
        except KeyError:
            raise UnknownModel(name, sorted(self.engines)) from None

    def info(self) -> list[ModelInfo]:
        return [ModelInfo(
            name=name, id=name, base=engine.model_id,
            description=f"{engine.model_id} answered by ruling over {engine.rotations} answer orderings",
            release_date=model_released(engine.model_id),
            revision=engine.revision, adapter=engine.adapter, calibration=dict(engine.calibration.temperatures),
            max_input_tokens=engine.max_input_tokens, max_branch_tokens=engine.max_branch_tokens,
            max_options=engine.max_options, rotations=engine.rotations, prior_debias=engine.prior_debias,
        ) for name, engine in self.engines.items()]


def _build_entry(entry: Entry, settings: Settings, loaded: dict[str, object]):
    """An engine for one entry, sharing an already loaded copy of its base when the LoRA shapes allow."""
    calibration = Calibration.load(entry.calibration) if entry.calibration else Calibration()
    for engine in loaded.values():
        if isinstance(engine, Engine) and engine.model_id == entry.model:
            try:
                return engine.variant(entry.adapter, calibration)
            except Engine.IncompatibleAdapter:
                break
    return build_engine(replace(settings, model=entry.model, adapter_path=entry.adapter,
                                calibration_path=entry.calibration, cascade_to=None, failover_to=None))
