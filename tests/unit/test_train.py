import dataclasses
import json

import numpy as np
import pytest

from ruling.evals import Record
from ruling.train import holdout_overlap, split_records, target_vector, permuted


def record(**extra):
    base = {"state": "The sky is clear.", "questions": {
        "color": {"type": "choice", "instructions": "Color?", "criteria": {"red": None, "blue": None, "green": None}},
        "rain": {"type": "noul", "instructions": "It will rain."},
        "clouds": {"type": "score", "instructions": "Cloud cover?", "criteria": ["none", "some", "full"]},
    }}
    return Record.model_validate({**base, **extra})


def test_target_prefers_soft_reference_over_hard_label():
    r = record(labels={"color": "blue", "rain": False}, references={"rain": {"target": {"yes": 0.3, "no": 0.7}}})
    assert target_vector(r, "color") == {"red": 0.0, "blue": 1.0, "green": 0.0}
    assert target_vector(r, "rain") == {"yes": 0.3, "no": 0.7}
    assert target_vector(r, "clouds") is None


def test_permutation_keeps_every_key_and_maps_targets_by_key():
    r = record(labels={"color": "blue", "rain": True, "clouds": 2})
    rng = np.random.default_rng(0)
    seen_orders = set()
    for _ in range(40):
        question, rotation = permuted(r.questions["color"], rng)
        assert rotation == 0 and sorted(question.criteria) == ["blue", "green", "red"]
        seen_orders.add(tuple(question.criteria))
        for qid in ("rain", "clouds"):
            same, rotation = permuted(r.questions[qid], rng)
            assert same == r.questions[qid] and rotation in (0, 1)
    assert len(seen_orders) == 6


def test_split_is_deterministic_disjoint_and_close_to_fraction():
    records = [record(labels={"color": "red"}).model_copy(update={"state": f"state {i}"}) for i in range(2000)]
    train_a, valid_a = split_records(records, 0.05)
    train_b, valid_b = split_records(list(reversed(records)), 0.05)
    assert {r.state for r in valid_a} == {r.state for r in valid_b}
    assert not {r.state for r in train_a} & {r.state for r in valid_a}
    assert 60 <= len(valid_a) <= 140


def test_holdout_overlap_detects_shared_states_after_normalizing_whitespace():
    training = [record(labels={"color": "red"}), record(labels={"color": "red"}).model_copy(update={"state": {"k": "v"}})]
    held = [record().model_copy(update={"state": "The  sky is\nclear."})]
    assert holdout_overlap(training, held) == 1
    assert holdout_overlap(training[1:], held) == 0


def test_trainer_refuses_data_that_overlaps_a_holdout_before_loading_a_model(tmp_path):
    from pathlib import Path

    from ruling.build import write
    from ruling.train import Trainer, TrainConfig

    authored = Path(__file__).resolve().parents[2] / "datasets" / "authored144.jsonl"
    leaked = json.loads(authored.read_text().splitlines()[5])
    data = tmp_path / "leaky.jsonl"
    write([leaked], data)
    config = TrainConfig(model="not-a-model-because-it-must-never-load", data=[data], output=tmp_path / "out", holdout=[authored])
    with pytest.raises(ValueError, match="also appear in held-out"):
        Trainer(config)


def test_examples_carry_the_record_weight_and_skip_weightless_rows():
    from types import SimpleNamespace

    from ruling.train import Example, Trainer

    records = [record(labels={"color": "blue"}, weight=2.0), record(labels={"rain": True}, weight=0.0),
               record(labels={"clouds": 1})]
    stub = SimpleNamespace(engine=SimpleNamespace(codes=SimpleNamespace(max_options=255)))
    assert Trainer._examples(stub, records) == [Example(0, "color", 2.0), Example(2, "clouds", 1.0)]


def test_batches_carry_one_weight_per_row():
    from types import SimpleNamespace

    import mlx.core as mx

    from ruling.train import Trainer

    stub = SimpleNamespace(engine=SimpleNamespace(_pad_id=0, codes=SimpleNamespace(token_ids=[11, 12, 13])))
    tokens, last, ids, mask, targets, weights, states = Trainer._batch(
        stub, [([5, 6], [1.0, 0.0], 0.5, None), ([7], [0.0, 0.0, 1.0], 2.0, {"photo": "seen"})])
    assert states == [None, {"photo": "seen"}]
    assert weights.tolist() == [0.5, 2.0] and weights.dtype == mx.float32
    assert targets.tolist() == [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]] and mask.tolist() == [[True, True, False], [True, True, True]]


def test_config_mismatch_ignores_steps_and_names_every_other_change(tmp_path):
    from ruling.train import TrainConfig, config_json, config_mismatch

    base = TrainConfig(model="m", data=[tmp_path / "a.jsonl"], output=tmp_path / "out", holdout=[])
    assert config_mismatch(config_json(base), config_json(base)) == []
    longer = dataclasses.replace(base, steps=base.steps + 500)
    assert config_mismatch(config_json(base), config_json(longer)) == []
    changed = dataclasses.replace(longer, learning_rate=3e-5, lora_rank=16, from_adapter=tmp_path / "seed")
    assert config_mismatch(config_json(base), config_json(changed)) == ["from_adapter", "learning_rate", "lora_rank"]
    assert json.loads(json.dumps(config_json(changed)))["from_adapter"] == str(tmp_path / "seed")


def test_checkpoint_round_trips_weights_optimizer_state_and_bookkeeping(tmp_path):
    import mlx.core as mx
    from mlx.utils import tree_flatten

    from ruling.train import load_checkpoint, save_checkpoint

    rng = np.random.default_rng(3)
    rng.permutation(50)
    weights = {"model.layers.5.self_attn.q_proj.lora_a": mx.array(rng.standard_normal((4, 2)), dtype=mx.float32)}
    optimizer = {"step": mx.array(7), "learning_rate": mx.array(2e-6),
                 "model": {"layers": [{}] * 5 + [{"self_attn": {"q_proj": {"lora_a": {"m": mx.ones((4, 2)), "v": mx.zeros((4, 2))}}}}]}}
    state = {"step": 7, "tokens_seen": 1234, "elapsed": 9.5, "history": [{"step": 0, "loss": 1.25}],
             "stream": {"epoch_rng": rng.bit_generator.state, "chunk_rng": rng.bit_generator.state, "start": 128, "cursor": 3},
             "config": {"steps": 10}}
    save_checkpoint(tmp_path / "checkpoint", weights, optimizer, state)
    back_weights, back_optimizer, back_state = load_checkpoint(tmp_path / "checkpoint")
    assert back_state == state
    assert np.array_equal(np.array(back_weights["model.layers.5.self_attn.q_proj.lora_a"]), np.array(weights["model.layers.5.self_attn.q_proj.lora_a"]))
    assert len(back_optimizer["model"]["layers"]) == 6 and int(back_optimizer["step"]) == 7
    for key, value in tree_flatten(optimizer):
        assert np.array_equal(np.array(dict(tree_flatten(back_optimizer))[key]), np.array(value)), key
    restored = np.random.default_rng(0)
    restored.bit_generator.state = back_state["stream"]["chunk_rng"]
    assert np.array_equal(restored.permutation(50), rng.permutation(50))


def test_batch_stream_resumes_to_the_exact_next_batch():
    from ruling.train import BatchStream

    def stream(rng):
        return BatchStream(300, 2, rng, lambda i: ([int(i)] * (1 + rng.integers(5)), [1.0]), lambda group: [t for t, _ in group])

    whole = stream(np.random.default_rng(1))
    first = [next(whole) for _ in range(200)]
    resumed = stream(np.random.default_rng(1))
    for _ in range(137):
        next(resumed)
    state = json.loads(json.dumps(resumed.state()))
    again = stream(np.random.default_rng(99))
    again.restore(state)
    assert [next(again) for _ in range(63)] == first[137:]
    assert again.state() == whole.state()
