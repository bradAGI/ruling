import json

import mlx.core as mx
import numpy as np
import pytest

from ruling import peft
from ruling.peft import export_adapter, import_adapter

LORA = {"rank": 8, "scale": 20.0, "dropout": 0.0, "keys": ["self_attn.q_proj", "self_attn.v_proj"]}


def synthetic_adapter(directory, layers, keys=LORA["keys"], rank=LORA["rank"]):
    """An mlx-lm adapter over `layers` with random fp32 factors on each of `keys`, as `ruling train` writes it."""
    rng = np.random.default_rng(len(layers))
    weights = {}
    for layer in layers:
        for key in keys:
            weights[f"model.layers.{layer}.{key}.lora_a"] = mx.array(rng.standard_normal((16, rank)), dtype=mx.float32)
            weights[f"model.layers.{layer}.{key}.lora_b"] = mx.array(rng.standard_normal((rank, 24)), dtype=mx.float32)
    directory.mkdir()
    mx.save_safetensors(str(directory / "adapters.safetensors"), weights)
    config = {"fine_tune_type": "lora", "num_layers": len(layers), "lora_parameters": {**LORA, "rank": rank, "keys": list(keys)}}
    (directory / "adapter_config.json").write_text(json.dumps(config))
    (directory / "training.json").write_text(json.dumps({"history": [{"step": 0, "loss": 1.5}]}))
    return weights, config


def test_export_then_import_round_trips_every_tensor_and_the_config(tmp_path, monkeypatch):
    monkeypatch.setattr(peft, "base_layer_count", lambda base, revision: 4)
    weights, config = synthetic_adapter(tmp_path / "mlx", layers=[2, 3])
    export_adapter(tmp_path / "mlx", tmp_path / "peft", base_model="org/base")
    exported = mx.load(str(tmp_path / "peft" / "adapter_model.safetensors"))
    peft_config = json.loads((tmp_path / "peft" / "adapter_config.json").read_text())
    assert peft_config["r"] == 8 and peft_config["lora_alpha"] == 160 and peft_config["base_model_name_or_path"] == "org/base"
    assert peft_config["target_modules"] == ["q_proj", "v_proj"] and peft_config["layers_to_transform"] == [2, 3]
    assert set(exported) == {f"base_model.model.model.layers.{layer}.self_attn.{m}.lora_{f}.weight"
                             for layer in (2, 3) for m in ("q_proj", "v_proj") for f in ("A", "B")}
    for name, tensor in exported.items():
        assert tensor.dtype == mx.float32 and tensor.shape[0 if name.endswith("lora_A.weight") else 1] == 8

    import_adapter(tmp_path / "peft", tmp_path / "back")
    imported = mx.load(str(tmp_path / "back" / "adapters.safetensors"))
    assert set(imported) == set(weights)
    for name in weights:
        assert np.array_equal(np.array(imported[name]), np.array(weights[name])), name
    assert json.loads((tmp_path / "back" / "adapter_config.json").read_text()) == config
    assert json.loads((tmp_path / "back" / "training.json").read_text()) == {"history": [{"step": 0, "loss": 1.5}]}


def test_import_refuses_layers_that_are_not_the_end_of_the_stack(tmp_path, monkeypatch):
    monkeypatch.setattr(peft, "base_layer_count", lambda base, revision: 4)
    synthetic_adapter(tmp_path / "mlx", layers=[1, 2])
    export_adapter(tmp_path / "mlx", tmp_path / "peft", base_model="org/base")
    with pytest.raises(ValueError, match="not the last 2 of 4"):
        import_adapter(tmp_path / "peft", tmp_path / "back")


def test_export_refuses_keys_outside_plain_layer_linears(tmp_path):
    weights, _ = synthetic_adapter(tmp_path / "mlx", layers=[0])
    weights["language_model.model.layers.0.self_attn.q_proj.lora_a"] = weights["model.layers.0.self_attn.q_proj.lora_a"]
    mx.save_safetensors(str(tmp_path / "mlx" / "adapters.safetensors"), weights)
    with pytest.raises(ValueError, match="language_model"):
        export_adapter(tmp_path / "mlx", tmp_path / "peft", base_model="org/base")


def test_export_refuses_layers_that_adapt_different_modules(tmp_path):
    weights, _ = synthetic_adapter(tmp_path / "mlx", layers=[0, 1])
    del weights["model.layers.1.self_attn.v_proj.lora_a"], weights["model.layers.1.self_attn.v_proj.lora_b"]
    mx.save_safetensors(str(tmp_path / "mlx" / "adapters.safetensors"), weights)
    with pytest.raises(ValueError, match="same modules"):
        export_adapter(tmp_path / "mlx", tmp_path / "peft", base_model="org/base")


def test_import_refuses_peft_variants_mlx_lm_cannot_serve(tmp_path, monkeypatch):
    monkeypatch.setattr(peft, "base_layer_count", lambda base, revision: 4)
    synthetic_adapter(tmp_path / "mlx", layers=[2, 3])
    export_adapter(tmp_path / "mlx", tmp_path / "peft", base_model="org/base")
    config_path = tmp_path / "peft" / "adapter_config.json"
    config = json.loads(config_path.read_text())
    config_path.write_text(json.dumps({**config, "use_rslora": True}))
    with pytest.raises(ValueError, match="use_rslora"):
        import_adapter(tmp_path / "peft", tmp_path / "back")
