import dataclasses
import json
import sys

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


def test_progress_runs_from_the_first_step_to_one_and_the_eta_to_zero():
    from ruling.train import progress

    steps, seconds_per_step = 10, 1.5
    entries = [progress(step, steps, step * seconds_per_step) for step in range(1, steps + 1)]
    assert [e["progress"] for e in entries] == sorted(e["progress"] for e in entries)
    assert entries[-1]["progress"] == 1.0 and entries[-1]["eta_seconds"] == 0.0
    for step, entry in enumerate(entries, start=1):
        assert entry["elapsed_seconds"] == step * seconds_per_step
        assert entry["eta_seconds"] == pytest.approx((steps - step) * seconds_per_step)


def test_progress_after_a_resume_rates_the_whole_run_not_the_part_since_restart():
    from ruling.train import progress

    # Twelve seconds for six steps: the checkpoint's elapsed plus this process's, so two seconds a step.
    assert progress(6, 10, 12.0)["eta_seconds"] == pytest.approx(8.0)


class FakeWandb:
    """A stand-in for the wandb module: records what a run was opened with, logged, and finished with."""

    def __init__(self):
        self.init_calls, self.logged, self.finished = [], [], []

    def init(self, project, config):
        self.init_calls.append({"project": project, "config": config})
        return self

    def log(self, data, step):
        self.logged.append((step, data))

    def finish(self, exit_code):
        self.finished.append(exit_code)


def _config(tmp_path):
    from ruling.train import TrainConfig

    return TrainConfig(model="m", data=[tmp_path / "a.jsonl"], output=tmp_path / "out", holdout=[], steps=4)


def test_wandb_sink_opens_a_run_on_the_config_logs_every_entry_by_step_and_finishes(tmp_path, monkeypatch):
    from ruling.train import WandbSink, config_json

    fake = FakeWandb()
    monkeypatch.setitem(sys.modules, "wandb", fake)
    entries = [{"step": 0, "loss": 1.4}, {"step": 2, "loss": 1.1, "train_loss": 1.3, "progress": 0.5}]
    with WandbSink("tickets", _config(tmp_path)) as sink:
        assert fake.init_calls == [{"project": "tickets", "config": config_json(_config(tmp_path))}]
        for entry in entries:
            sink(entry)
        assert fake.finished == []
    assert fake.logged == [(0, entries[0]), (2, entries[1])]
    assert fake.finished == [0]


def test_wandb_sink_finishes_the_run_as_failed_when_training_raises(tmp_path, monkeypatch):
    from ruling.train import WandbSink

    fake = FakeWandb()
    monkeypatch.setitem(sys.modules, "wandb", fake)
    with pytest.raises(RuntimeError, match="out of memory"):
        with WandbSink(None, _config(tmp_path)) as sink:
            sink({"step": 0, "loss": 1.4})
            raise RuntimeError("out of memory")
    assert fake.init_calls[0]["project"] is None
    assert fake.finished == [1]


def test_wandb_sink_names_the_extra_to_install_when_the_package_is_missing(tmp_path, monkeypatch):
    from ruling.train import WandbSink

    monkeypatch.setitem(sys.modules, "wandb", None)
    with pytest.raises(ImportError, match=r"pip install 'ruling\[wandb\]'"):
        WandbSink("tickets", _config(tmp_path))


def test_examples_longer_than_max_tokens_are_dropped_and_counted():
    from types import SimpleNamespace

    from ruling.engine import _Prefix
    from ruling.train import Example, Trainer

    records = [record(state="short", labels={"rain": True}), record(state="a much longer state", labels={"rain": True})]
    examples = [Example(0, "rain", 1.0), Example(1, "rain", 1.0)]
    engine = SimpleNamespace(codes=SimpleNamespace(codes="ABC"), _prefix=lambda state: _Prefix(list(state), ""),
                             _encode=lambda text: [0])
    stub = SimpleNamespace(engine=engine, config=SimpleNamespace(max_tokens=6), _tokens=None)
    stub._tokens = lambda *args: Trainer._tokens(stub, *args)
    # The longest example, state and question as the model reads them, is the limit that would keep them all.
    assert Trainer._fitting(stub, records, examples) == ([Example(0, "rain", 1.0)], 1, 20)


def test_options_with_few_or_no_labels_are_named_before_training():
    from ruling.train import Example, thin_options

    records = [record(labels={"color": "red"}) for _ in range(12)] + [record(labels={"color": "blue"})] * 3
    records.append(record(references={"color": {"target": {"red": 0.5, "blue": 0.5, "green": 0.0}}}))
    examples = [Example(i, "color", 1.0) for i in range(len(records))]
    assert thin_options(records, examples, few=10) == ["color: no labels for green; fewer than 10 for blue (4)"]
    assert thin_options(records[:12], [Example(i, "color", 1.0) for i in range(12)], few=1) == \
        ["color: no labels for blue, green"]
