"""Runtime settings, read from the environment."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import mlx.core as mx

from ruling.hosted import TOP_LOGPROBS

DEFAULT_MODEL = "mlx-community/Qwen3.5-4B-4bit"


def configure_memory(settings: "Settings") -> None:
    """Keep MLX well inside physical memory.

    MLX's default ceiling is 1.5 times the GPU's recommended working set, 60 GB
    on a 48 GB machine, so it swaps rather than stopping. Batches of varying
    length also leave differently sized freed buffers in its cache, which by
    default may grow to that ceiling: 33 GB after twenty training steps of a
    150M-parameter encoder, which got the process killed once another model
    was loaded alongside it. The memory limit makes MLX evaluate in smaller
    pieces and raise if a job cannot fit; the cache limit returns freed buffers.
    """
    physical = mx.device_info()["memory_size"]
    mx.set_memory_limit(int(physical * settings.memory_fraction))
    mx.set_cache_limit(int(settings.cache_limit_gb * 2**30))


def _flag(value: str) -> bool:
    if value.lower() in ("1", "true", "yes", "on"):
        return True
    if value.lower() in ("0", "false", "no", "off"):
        return False
    raise ValueError(f"expected a boolean, got {value!r}")


@dataclass(frozen=True)
class Settings:
    model: str = DEFAULT_MODEL
    calibration_path: Path | None = None
    adapter_path: Path | None = None
    max_input_tokens: int = 65_536
    max_branch_tokens: int = 32_768
    rotations: int = 3
    prior_debias: bool = False
    api_key: str | None = None
    cache_limit_gb: float = 4.0
    memory_fraction: float = 0.5
    cascade_to: str | None = None
    cascade_threshold: float | None = None
    openai_base_url: str = ""
    openai_api_key: str | None = None
    openai_top_logprobs: int = TOP_LOGPROBS
    openai_extra_body: dict = field(default_factory=dict)
    openrouter_providers: tuple[str, ...] = ()
    openrouter_budget_usd: float = 1.0
    host: str = "127.0.0.1"
    port: int = 8010

    @classmethod
    def from_env(cls) -> "Settings":
        env = os.environ
        calibration = env.get("RULING_CALIBRATION")
        adapter = env.get("RULING_ADAPTER")
        return cls(
            model=env.get("RULING_MODEL", DEFAULT_MODEL),
            calibration_path=Path(calibration) if calibration else None,
            adapter_path=Path(adapter) if adapter else None,
            max_input_tokens=int(env.get("RULING_MAX_INPUT_TOKENS", cls.max_input_tokens)),
            max_branch_tokens=int(env.get("RULING_MAX_BRANCH_TOKENS", cls.max_branch_tokens)),
            rotations=int(env.get("RULING_ROTATIONS", cls.rotations)),
            prior_debias=_flag(env.get("RULING_PRIOR_DEBIAS", str(cls.prior_debias))),
            api_key=env.get("RULING_API_KEY") or None,
            cache_limit_gb=float(env.get("RULING_CACHE_LIMIT_GB", cls.cache_limit_gb)),
            memory_fraction=float(env.get("RULING_MEMORY_FRACTION", cls.memory_fraction)),
            cascade_to=env.get("RULING_CASCADE_TO") or None,
            cascade_threshold=float(env["RULING_CASCADE_THRESHOLD"]) if env.get("RULING_CASCADE_THRESHOLD") else None,
            openai_base_url=env.get("RULING_OPENAI_BASE_URL", cls.openai_base_url),
            openai_api_key=env.get("RULING_OPENAI_API_KEY") or None,
            openai_top_logprobs=int(env.get("RULING_OPENAI_TOP_LOGPROBS", cls.openai_top_logprobs)),
            openai_extra_body=json.loads(env.get("RULING_OPENAI_EXTRA_BODY", "{}")),
            openrouter_providers=tuple(p for p in env.get("RULING_OPENROUTER_PROVIDERS", "").split(",") if p),
            openrouter_budget_usd=float(env.get("RULING_OPENROUTER_BUDGET_USD", cls.openrouter_budget_usd)),
            host=env.get("RULING_HOST", cls.host),
            port=int(env.get("RULING_PORT", cls.port)),
        )
