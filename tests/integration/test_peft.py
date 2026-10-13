import json
import os

import numpy as np
import pytest

from ruling import states
from ruling.build import write
from ruling.calibration import Calibration
from ruling.cli import main
from ruling.engine import Engine
from ruling.train import Trainer, TrainConfig

pytestmark = [pytest.mark.integration, pytest.mark.heavy]

# A pure-attention base: the default test model, Qwen3.5, nests its stack under `language_model` and mixes in
# linear-attention blocks whose projections have no PEFT counterpart, so its adapters are refused by design.
PEFT_TEST_MODEL = os.environ.get("RULING_PEFT_TEST_MODEL", "mlx-community/Qwen3-0.6B-4bit")
PEFT_BASE = os.environ.get("RULING_PEFT_TEST_BASE", "Qwen/Qwen3-0.6B")


@pytest.fixture(scope="module")
def adapter(tmp_path_factory):
    """Two steps at a high learning rate, saved as they stand rather than at the best validation loss.

    Two steps rarely beat the untrained validation loss, and the untrained adapter has lora_b at zero, which
    would change nothing.
    """
    root = tmp_path_factory.mktemp("peft")
    data = root / "states.jsonl"
    write([r.model_dump(mode="json") for r in states.generate(60, seed=5)], data)
    config = TrainConfig(model=PEFT_TEST_MODEL, data=[data], output=root / "adapter", holdout=[], valid_fraction=0.2,
                         steps=2, batch_size=2, warmup=1, learning_rate=1e-3, lora_layers=2, max_tokens=256, eval_every=2,
                         valid_examples=8, seed=0)
    trainer = Trainer(config)
    trainer.save(root / "final", trainer.run(log=lambda _: None))
    return config, root / "final"


def test_export_loads_in_peft_on_the_fp16_base_and_changes_its_output(adapter, tmp_path):
    _, source = adapter
    import torch
    from peft import LoraConfig, get_peft_model
    from peft.utils import set_peft_model_state_dict
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM, AutoTokenizer

    main(["export-adapter", str(source), "--output", str(tmp_path / "peft"), "--base-model", PEFT_BASE])
    assert json.loads((tmp_path / "peft" / "training.json").read_text())["history"]

    base = AutoModelForCausalLM.from_pretrained(PEFT_BASE, dtype=torch.float32)
    base.eval()
    inputs = AutoTokenizer.from_pretrained(PEFT_BASE)("Since this morning every request returns 401.", return_tensors="pt")
    with torch.no_grad():
        before = base(**inputs).logits.clone()
    model = get_peft_model(base, LoraConfig.from_pretrained(tmp_path / "peft"))
    result = set_peft_model_state_dict(model, load_file(tmp_path / "peft" / "adapter_model.safetensors"))
    assert result.unexpected_keys == []
    assert [k for k in result.missing_keys if "lora_" in k] == []
    model.eval()
    with torch.no_grad():
        after = model(**inputs).logits
    assert not torch.allclose(before, after, atol=1e-4)


def test_import_of_the_export_serves_the_same_distributions(adapter, tmp_path):
    config, source = adapter
    main(["export-adapter", str(source), "--output", str(tmp_path / "peft"), "--base-model", PEFT_BASE])
    main(["import-adapter", str(tmp_path / "peft"), "--output", str(tmp_path / "back")])
    imported_config = json.loads((tmp_path / "back" / "adapter_config.json").read_text())
    assert imported_config["num_layers"] == config.lora_layers
    assert imported_config["lora_parameters"]["scale"] == config.lora_scale
    original = Engine(config.model, Calibration(), 4096, rotations=1, prior_debias=False, adapter=source)
    imported = Engine(config.model, Calibration(), 4096, rotations=1, prior_debias=False, adapter=tmp_path / "back")
    for record in states.generate(6, seed=77):
        theirs = original.score(record.state, record.questions).raw
        ours = imported.score(record.state, record.questions).raw
        for qid in theirs:
            assert np.allclose(ours[qid].logits, theirs[qid].logits, atol=1e-5)
