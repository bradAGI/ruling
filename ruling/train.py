"""LoRA fine-tuning of the answer-letter readout on typed-question records.

Keep the learning rate near 1e-5. At 1e-4 the letter readout collapses toward
uniform answers: loss falls while accuracy drops below the untrained model.

Train a pure-attention base. Hybrid linear-attention models such as Qwen3.5
fall back to a step-by-step recurrence when training in mlx-lm, which is one
to two orders of magnitude slower on this hardware.

Every labeled question becomes one example rendered exactly as the engine
renders it at inference, with the answer order randomized each time it is
seen: Choice options are shuffled, Noul and Score take either direction. The
loss is cross-entropy between the model's distribution over the allowed answer
letters and the target distribution, which is a proper scoring rule, so its
minimum is the true probability. Targets are soft where annotator fractions or
designed ambiguity exist (a `target` reference) and one-hot otherwise.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten
from mlx_lm.tuner.trainer import grad_checkpoint
from mlx_lm.tuner.utils import linear_to_lora_layers

from ruling.calibration import Calibration, expected_calibration_error
from ruling.engine import Engine, option_logits
from ruling.evals import Record, label_index, load_dataset
from ruling.prompt import render_question
from ruling.questions import Choice, Noul, Score, option_keys, render_text

TARGET = "target"


@dataclass(frozen=True)
class TrainConfig:
    model: str
    data: list[Path]
    output: Path
    holdout: list[Path]
    valid_fraction: float = 0.02
    steps: int = 2000
    batch_size: int = 8
    learning_rate: float = 1e-5
    max_grad_norm: float = 1.0
    warmup: int = 100
    lora_layers: int = 16
    lora_rank: int = 8
    lora_scale: float = 20.0
    lora_dropout: float = 0.0
    max_tokens: int = 1024
    eval_every: int = 250
    valid_examples: int = 1500
    grad_checkpoint: bool = True
    seed: int = 0


@dataclass(frozen=True)
class Example:
    record: int
    question_id: str


def target_vector(record: Record, question_id: str) -> dict[str, float] | None:
    """The training target over canonical keys: a soft `target` reference, else the hard label."""
    soft = record.references.get(question_id, {}).get(TARGET)
    if soft is not None:
        return soft
    if question_id not in record.labels:
        return None
    question = record.questions[question_id]
    keys = option_keys(question)
    gold = label_index(question, record.labels[question_id])
    return {key: float(i == gold) for i, key in enumerate(keys)}


def permuted(question: Choice | Score | Noul, rng: np.random.Generator) -> tuple[Choice | Score | Noul, int]:
    """A random presentation of the question: shuffled Choice options, or a random direction."""
    if isinstance(question, Choice):
        order = list(rng.permutation(list(question.criteria)))
        return Choice(instructions=question.instructions, criteria={k: question.criteria[k] for k in order}), 0
    return question, int(rng.integers(2))


def split_records(records: list[Record], valid_fraction: float) -> tuple[list[Record], list[Record]]:
    """Deterministic split by a hash of the state, so reruns and reorderings agree."""
    train, valid = [], []
    for record in records:
        digest = hashlib.sha256(_state_key(record.state).encode()).digest()
        (valid if int.from_bytes(digest[:8], "big") / 2**64 < valid_fraction else train).append(record)
    return train, valid


def holdout_overlap(training: list[Record], held_out: list[Record]) -> int:
    """How many training states also appear, up to whitespace, in held-out evaluation data."""
    held = {_state_key(r.state) for r in held_out}
    return sum(1 for r in training if _state_key(r.state) in held)


def _state_key(state) -> str:
    return " ".join(render_text(state).split())


class Trainer:
    def __init__(self, config: TrainConfig):
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        records = [r for path in config.data for r in load_dataset(path)]
        held_out = [r for path in config.holdout for r in load_dataset(path)]
        overlap = holdout_overlap(records, held_out)
        if overlap:
            raise ValueError(f"{overlap} training states also appear in held-out evaluation data")
        self.train_records, self.valid_records = split_records(records, config.valid_fraction)

        mx.random.seed(config.seed)
        self.engine = Engine(config.model, Calibration(), config.max_tokens, rotations=1, prior_debias=False)
        model = self.engine.model
        model.freeze()
        lora = {"rank": config.lora_rank, "scale": config.lora_scale, "dropout": config.lora_dropout}
        linear_to_lora_layers(model, config.lora_layers, lora)
        if config.grad_checkpoint:
            grad_checkpoint(model.layers[0])
        self.lora = lora

        self.train_examples = self._examples(self.train_records)
        valid = self._examples(self.valid_records)
        self.valid_examples = [valid[i] for i in self.rng.permutation(len(valid))[: config.valid_examples]]
        if not self.train_examples or not self.valid_examples:
            raise ValueError("training and validation both need labeled questions")

    def _examples(self, records: list[Record]) -> list[Example]:
        examples = []
        for index, record in enumerate(records):
            for qid, question in record.questions.items():
                cardinality = len(option_keys(question))
                if cardinality <= self.engine.codes.max_options and target_vector(record, qid) is not None:
                    examples.append(Example(index, qid))
        return examples

    def _render(self, records: list[Record], example: Example, augment: bool) -> tuple[list[int], list[float]] | None:
        record = records[example.record]
        question = record.questions[example.question_id]
        presented, rotation = permuted(question, self.rng) if augment else (question, 0)
        rendered = render_question(presented, self.engine.codes.codes, rotation)
        prefix, tail = self.engine.split_prompt(record.state)
        tokens = self.engine._encode(prefix) + self.engine._encode(rendered.text + tail)
        if len(tokens) > self.config.max_tokens:
            return None
        target = target_vector(record, example.question_id)
        return tokens, [target[key] for key in rendered.keys]

    def _batch(self, rendered: list[tuple[list[int], list[float]]]):
        width = math.ceil(max(len(t) for t, _ in rendered) / 32) * 32
        options = max(len(target) for _, target in rendered)
        pad = self.engine._pad_id
        tokens = mx.array([t + [pad] * (width - len(t)) for t, _ in rendered])
        last = mx.array([len(t) - 1 for t, _ in rendered])
        ids = mx.array([self.engine.codes.token_ids[:options] for _ in rendered])
        mask = mx.array([[i < len(target) for i in range(options)] for _, target in rendered])
        targets = mx.array([target + [0.0] * (options - len(target)) for _, target in rendered], dtype=mx.float32)
        return tokens, last, ids, mask, targets

    @staticmethod
    def _log_probs(model, tokens, last, ids, mask):
        logits = mx.where(mask, option_logits(model, tokens, None, last, ids), -mx.inf)
        return logits - mx.logsumexp(logits, axis=-1, keepdims=True)

    @classmethod
    def loss(cls, model, tokens, last, ids, mask, targets):
        log_probs = cls._log_probs(model, tokens, last, ids, mask)
        return -mx.where(mask, targets * log_probs, 0.0).sum(axis=-1).mean()

    def _batches(self):
        """Endless shuffled batches, grouped by length to limit padding."""
        size = self.config.batch_size
        while True:
            order = self.rng.permutation(len(self.train_examples))
            for start in range(0, len(order), size * 64):
                chunk = [self._render(self.train_records, self.train_examples[i], augment=True) for i in order[start:start + size * 64]]
                chunk = sorted((c for c in chunk if c is not None), key=lambda c: len(c[0]))
                groups = [chunk[i:i + size] for i in range(0, len(chunk) - size + 1, size)]
                for g in self.rng.permutation(len(groups)):
                    yield self._batch(groups[g])

    def validate(self) -> dict:
        model = self.engine.model
        model.eval()
        losses, top, correct = [], [], []
        size = self.config.batch_size
        rendered = [r for r in (self._render(self.valid_records, e, augment=False) for e in self.valid_examples) if r is not None]
        rendered.sort(key=lambda c: len(c[0]))
        for start in range(0, len(rendered), size):
            group = rendered[start:start + size]
            tokens, last, ids, mask, targets = self._batch(group)
            log_probs = self._log_probs(model, tokens, last, ids, mask)
            ce = -mx.where(mask, targets * log_probs, 0.0).sum(axis=-1)
            mx.eval(ce, log_probs)
            probs = np.exp(np.where(np.array(mask), np.array(log_probs), -np.inf))
            for row, (_, target), loss in zip(probs, group, np.array(ce)):
                row = row[: len(target)]
                losses.append(float(loss))
                top.append(float(row.max()))
                correct.append(int(row.argmax()) == int(np.argmax(target)))
        model.train()
        return {"loss": float(np.mean(losses)), "accuracy": float(np.mean(correct)),
                "expected_calibration_error": expected_calibration_error(top, correct), "count": len(losses)}

    def save(self, directory: Path, history: list[dict]) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        weights = dict(tree_flatten(self.engine.model.trainable_parameters()))
        mx.save_safetensors(str(directory / "adapters.safetensors"), weights)
        (directory / "adapter_config.json").write_text(json.dumps(
            {"fine_tune_type": "lora", "num_layers": self.config.lora_layers, "lora_parameters": self.lora}, indent=2) + "\n")
        data = [{"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in self.config.data]
        meta = {"config": {**asdict(self.config), "data": data, "output": str(self.config.output),
                           "holdout": [str(p) for p in self.config.holdout]},
                "base_revision": self.engine.revision, "train_records": len(self.train_records),
                "train_examples": len(self.train_examples), "valid_examples": len(self.valid_examples), "history": history}
        (directory / "training.json").write_text(json.dumps(meta, indent=2) + "\n")

    def run(self, log=print) -> list[dict]:
        config = self.config
        model = self.engine.model
        model.train()
        schedule = optim.join_schedules(
            [optim.linear_schedule(1e-7, config.learning_rate, config.warmup),
             optim.cosine_decay(config.learning_rate, max(config.steps - config.warmup, 1), config.learning_rate * 0.05)],
            [config.warmup],
        )
        optimizer = optim.AdamW(learning_rate=schedule, weight_decay=0.0)
        step_fn = nn.value_and_grad(model, self.loss)
        trainable = sum(v.size for _, v in tree_flatten(model.trainable_parameters()))
        log(f"trainable parameters {trainable:,}; {len(self.train_examples):,} train and {len(self.valid_examples):,} validation examples")

        history = [{"step": 0, **self.validate()}]
        log(json.dumps(history[-1]))
        best = history[-1]["loss"]
        self.save(config.output, history)
        batches = self._batches()
        tokens_seen, started, window = 0, time.perf_counter(), []
        for step in range(1, config.steps + 1):
            tokens, last, ids, mask, targets = next(batches)
            loss, grads = step_fn(model, tokens, last, ids, mask, targets)
            grads, _ = optim.clip_grad_norm(grads, config.max_grad_norm)
            optimizer.update(model, grads)
            mx.eval(model.trainable_parameters(), optimizer.state, loss)
            window.append(float(loss))
            tokens_seen += tokens.size
            if step % config.eval_every == 0 or step == config.steps:
                elapsed = time.perf_counter() - started
                entry = {"step": step, "train_loss": float(np.mean(window)), "tokens_per_second": tokens_seen / elapsed,
                         **self.validate()}
                window = []
                history.append(entry)
                log(json.dumps(entry))
                if entry["loss"] < best:
                    best = entry["loss"]
                    self.save(config.output, history)
                else:
                    meta = json.loads((config.output / "training.json").read_text())
                    meta["history"] = history
                    (config.output / "training.json").write_text(json.dumps(meta, indent=2) + "\n")
        return history
