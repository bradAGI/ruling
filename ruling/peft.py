"""Convert a ruling adapter to and from PEFT's LoRA layout.

mlx-lm stores `lora_a` as `(in, r)` and `lora_b` as `(r, out)` and applies
`scale * (x @ lora_a) @ lora_b`. PEFT stores `lora_A.weight` as `(r, in)` and
`lora_B.weight` as `(out, r)` and applies `(lora_alpha / r) * lora_B(lora_A(x))`.
The conversion is therefore a transpose of each tensor and `lora_alpha = scale * r`.

Only plain per-layer linears convert. mlx-lm can also put LoRA on switch
(MoE) linears and embeddings, and multimodal wrappers nest the stack under
`language_model`; none of those has a one-to-one PEFT key, so they are refused
rather than exported under a name PEFT would silently ignore.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import mlx.core as mx
from huggingface_hub import hf_hub_download

MLX_KEY = re.compile(r"^model\.layers\.(\d+)\.(.+)\.lora_([ab])$")
PEFT_KEY = re.compile(r"^base_model\.model\.model\.layers\.(\d+)\.(.+)\.lora_([AB])\.weight$")
MLX_WEIGHTS = "adapters.safetensors"
PEFT_WEIGHTS = "adapter_model.safetensors"
CONFIG = "adapter_config.json"
TRAINING = "training.json"


def export_adapter(adapter: Path, output: Path, base_model: str) -> None:
    """Write `adapter`, a directory from `ruling train`, as a PEFT adapter for `base_model`."""
    config = json.loads((adapter / CONFIG).read_text())
    if config.get("fine_tune_type", "lora") != "lora":
        raise ValueError(f"only LoRA adapters export to PEFT, not {config['fine_tune_type']}")
    lora = config["lora_parameters"]
    rank = lora["rank"]
    modules = _modules(mx.load(str(adapter / MLX_WEIGHTS)), MLX_KEY, "a", "b", rank, transposed=False)
    layers, keys = _product(modules)
    weights = {}
    for (layer, key), (a, b) in modules.items():
        prefix = f"base_model.model.model.layers.{layer}.{key}"
        weights[f"{prefix}.lora_A.weight"] = a.T.astype(mx.float32)
        weights[f"{prefix}.lora_B.weight"] = b.T.astype(mx.float32)
    alpha = lora["scale"] * rank
    peft_config = {
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "base_model_name_or_path": base_model,
        "revision": None,
        "inference_mode": False,
        "r": rank,
        "lora_alpha": int(alpha) if alpha == int(alpha) else alpha,
        "lora_dropout": lora["dropout"],
        "target_modules": sorted({key.rsplit(".", 1)[-1] for key in keys}),
        "layers_to_transform": layers,
        "bias": "none",
    }
    _write(output, PEFT_WEIGHTS, weights, peft_config, adapter / TRAINING)


def import_adapter(peft: Path, output: Path) -> None:
    """Write `peft`, a PEFT LoRA adapter directory, in the layout `ruling train` produces and the engine serves."""
    config = json.loads((peft / CONFIG).read_text())
    _check_peft_config(config)
    rank = config["r"]
    modules = _modules(mx.load(str(peft / PEFT_WEIGHTS)), PEFT_KEY, "A", "B", rank, transposed=True)
    layers, keys = _product(modules)
    total = base_layer_count(config["base_model_name_or_path"], config.get("revision"))
    if layers != list(range(total - len(layers), total)):
        raise ValueError(f"LoRA layers {layers} are not the last {len(layers)} of {total}; "
                         "mlx-lm can only adapt a suffix of the stack")
    weights = {}
    for (layer, key), (a, b) in modules.items():
        weights[f"model.layers.{layer}.{key}.lora_a"] = a.T
        weights[f"model.layers.{layer}.{key}.lora_b"] = b.T
    mlx_config = {
        "fine_tune_type": "lora",
        "num_layers": len(layers),
        "lora_parameters": {"rank": rank, "scale": config["lora_alpha"] / rank, "dropout": config.get("lora_dropout", 0.0),
                            "keys": keys},
    }
    _write(output, MLX_WEIGHTS, weights, mlx_config, peft / TRAINING)


def base_layer_count(base_model: str, revision: str | None) -> int:
    """How many decoder layers the base model has, from its Hugging Face `config.json`."""
    config = json.loads(Path(hf_hub_download(base_model, "config.json", revision=revision)).read_text())
    if "num_hidden_layers" not in config:
        raise ValueError(f"{base_model}'s config.json has no top-level num_hidden_layers, so the LoRA layer "
                         "range cannot be checked against the stack")
    return config["num_hidden_layers"]


def _modules(weights: dict[str, mx.array], pattern: re.Pattern, first: str, second: str, rank: int,
             transposed: bool) -> dict[tuple[int, str], tuple[mx.array, mx.array]]:
    """Pair each module's two LoRA factors by (layer, module path), refusing any key outside the pattern."""
    halves: dict[tuple[int, str], dict[str, mx.array]] = {}
    for name, tensor in weights.items():
        match = pattern.match(name)
        if match is None:
            raise ValueError(f"{name!r} is not a LoRA factor on a plain per-layer linear; only "
                             "model.layers.N.<module> adapters convert")
        layer, key, factor = int(match.group(1)), match.group(2), match.group(3)
        halves.setdefault((layer, key), {})[factor] = tensor
    modules = {}
    for (layer, key), pair in sorted(halves.items()):
        if set(pair) != {first, second}:
            raise ValueError(f"layer {layer} {key} has only one of its two LoRA factors")
        a, b = pair[first], pair[second]
        rank_axis = (a.shape[0], b.shape[1]) if transposed else (a.shape[1], b.shape[0])
        if rank_axis != (rank, rank):
            raise ValueError(f"layer {layer} {key} has rank {rank_axis}, but the config says {rank}")
        modules[(layer, key)] = (a, b)
    return modules


def _product(modules: dict[tuple[int, str], object]) -> tuple[list[int], list[str]]:
    """The layer indices and module paths, which must form a full product for either layout to express them."""
    layers = sorted({layer for layer, _ in modules})
    keys = sorted({key for _, key in modules})
    missing = [(layer, key) for layer in layers for key in keys if (layer, key) not in modules]
    if missing:
        raise ValueError(f"every adapted layer must adapt the same modules; missing {missing}")
    return layers, keys


def _check_peft_config(config: dict) -> None:
    if config.get("peft_type") != "LORA":
        raise ValueError(f"only LORA adapters import, not {config.get('peft_type')}")
    for field, plain in (("use_dora", False), ("use_rslora", False), ("bias", "none"), ("rank_pattern", {}),
                         ("alpha_pattern", {}), ("fan_in_fan_out", False)):
        if config.get(field, plain) not in (plain, None):
            raise ValueError(f"PEFT adapters with {field}={config[field]!r} have no mlx-lm equivalent")


def _write(output: Path, weights_name: str, weights: dict[str, mx.array], config: dict, training: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(output / weights_name), weights)
    (output / CONFIG).write_text(json.dumps(config, indent=2) + "\n")
    # Provenance travels with the weights in both directions.
    if training.exists():
        shutil.copyfile(training, output / TRAINING)
