"""Full fine-tuning of a decision encoder on typed-question records.

Each training example is a whole record: its state and every question that
has a target, packed and scored in one pass exactly as the engine serves it.
Answers do not depend on option order or on which other questions are asked,
so there is nothing to augment. The loss is cross-entropy against each
question's target distribution, a proper scoring rule, averaged over the
questions in a batch. The new scoring head starts at zero, so the untrained
model answers uniformly, and learns at a higher rate than the pretrained
encoder.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten

from ruling.calibration import expected_calibration_error
from ruling.decision import Checkpoint, InputTooLong, checkpoint_layers, Packed, batch_inputs, load_model, pack
from ruling.evals import Record, load_dataset
from ruling.train import holdout_overlap, split_records, target_vector


@dataclass(frozen=True)
class DecisionTrainConfig:
    base: str
    revision: str
    data: list[Path]
    holdout: list[Path]
    output: Path
    valid_fraction: float = 0.02
    steps: int = 3000
    batch_tokens: int = 8192
    learning_rate: float = 3e-5
    head_learning_rate: float = 1e-3
    weight_decay: float = 0.01
    warmup: int = 150
    max_tokens: int = 2048
    eval_every: int = 250
    valid_records: int = 800
    grad_checkpoint: bool = True
    seed: int = 0


@dataclass(frozen=True)
class Example:
    packed: Packed
    targets: list[list[float]]


class DecisionTrainer:
    def __init__(self, config: DecisionTrainConfig):
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        records = [r for path in config.data for r in load_dataset(path)]
        held_out = [r for path in config.holdout for r in load_dataset(path)]
        overlap = holdout_overlap(records, held_out)
        if overlap:
            raise ValueError(f"{overlap} training states also appear in held-out evaluation data")
        train_records, valid_records = split_records(records, config.valid_fraction)
        mx.random.seed(config.seed)
        self.model, self.tokenizer, self.special = load_model(config.base, config.revision, None)
        if config.grad_checkpoint:
            checkpoint_layers(self.model)
        self.train = self._examples(train_records)
        valid = self._examples(valid_records)
        self.valid = [valid[i] for i in self.rng.permutation(len(valid))[: config.valid_records]]
        if not self.train or not self.valid:
            raise ValueError("training and validation both need records with labeled questions")

    def _examples(self, records: list[Record]) -> list[Example]:
        encode = lambda text: self.tokenizer.encode(text, add_special_tokens=False)
        examples = []
        for record in records:
            questions = {qid: q for qid, q in record.questions.items() if target_vector(record, qid) is not None}
            if not questions:
                continue
            try:
                packed = pack(record.state, questions, encode, self.special, self.config.max_tokens)
            except InputTooLong:
                continue
            targets = [[target_vector(record, qid)[key] for key, _ in packed.slots[qid]] for qid in questions]
            examples.append(Example(packed, targets))
        return examples

    def _batch(self, examples: list[Example]):
        tokens, positions, mask = batch_inputs([e.packed for e in examples], self.special.pad)
        rows, cols, slot_index, targets = [], [], [], []
        width = max(len(t) for e in examples for t in e.targets)
        for b, example in enumerate(examples):
            for slots, target in zip(example.packed.slots.values(), example.targets):
                start = len(rows)
                for _, col in slots:
                    rows.append(b)
                    cols.append(col)
                slot_index.append([start + i for i in range(len(slots))] + [0] * (width - len(slots)))
                targets.append(target + [0.0] * (width - len(target)))
        valid = [[i < len(e_t) for i in range(width)] for e in examples for e_t in e.targets]
        return (tokens, positions, mask, mx.array(rows), mx.array(cols), mx.array(slot_index),
                mx.array(valid), mx.array(targets, dtype=mx.float32))

    @staticmethod
    def _log_probs(model, tokens, positions, mask, rows, cols, slot_index, valid):
        scores = model(tokens, positions, mask, rows, cols)
        logits = mx.where(valid, scores[slot_index], -mx.inf)
        return logits - mx.logsumexp(logits, axis=-1, keepdims=True)

    @classmethod
    def loss(cls, model, tokens, positions, mask, rows, cols, slot_index, valid, targets):
        log_probs = cls._log_probs(model, tokens, positions, mask, rows, cols, slot_index, valid)
        return -mx.where(valid, targets * log_probs, 0.0).sum(axis=-1).mean()

    def _groups(self, examples: list[Example], shuffle: bool) -> list[list[Example]]:
        order = self.rng.permutation(len(examples)) if shuffle else np.arange(len(examples))
        groups = []
        for start in range(0, len(order), 512):
            chunk = sorted((examples[i] for i in order[start:start + 512]), key=lambda e: len(e.packed.tokens))
            current: list[Example] = []
            for example in chunk:
                width = max([len(e.packed.tokens) for e in current] + [len(example.packed.tokens)])
                if current and width * (len(current) + 1) > self.config.batch_tokens:
                    groups.append(current)
                    current = []
                current.append(example)
            if current:
                groups.append(current)
        if shuffle:
            groups = [groups[i] for i in self.rng.permutation(len(groups))]
        return groups

    def validate(self) -> dict:
        self.model.eval()
        losses, top, correct = [], [], []
        for group in self._groups(self.valid, shuffle=False):
            batch = self._batch(group)
            log_probs = self._log_probs(self.model, *batch[:7])
            valid, targets = np.array(batch[6]), np.array(batch[7])
            probs = np.exp(np.where(valid, np.array(log_probs), -np.inf))
            for row, v, target in zip(probs, valid, targets):
                row, target = row[v], target[v]
                losses.append(float(-(target * np.log(np.maximum(row, 1e-12))).sum()))
                top.append(float(row.max()))
                correct.append(int(row.argmax()) == int(target.argmax()))
        self.model.train()
        return {"loss": float(np.mean(losses)), "accuracy": float(np.mean(correct)),
                "expected_calibration_error": expected_calibration_error(top, correct), "questions": len(losses)}

    def save(self, history: list[dict]) -> None:
        data = [{"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in self.config.data]
        extra = {"training": {**asdict(self.config), "data": data, "output": str(self.config.output),
                              "holdout": [str(p) for p in self.config.holdout]},
                 "train_records": len(self.train), "valid_records": len(self.valid), "history": history}
        Checkpoint(self.config.base, self.config.revision).save(self.config.output, self.model, extra)

    def run(self, log=print) -> list[dict]:
        config = self.config

        def schedule(peak):
            return optim.join_schedules([optim.linear_schedule(peak * 0.01, peak, config.warmup),
                                         optim.cosine_decay(peak, max(config.steps - config.warmup, 1), peak * 0.05)],
                                        [config.warmup])

        optimizer = optim.MultiOptimizer(
            [optim.AdamW(learning_rate=schedule(config.head_learning_rate), weight_decay=0.0),
             optim.AdamW(learning_rate=schedule(config.learning_rate), weight_decay=config.weight_decay)],
            [lambda path, _: path.startswith("score.")],
        )
        self.model.train()
        step_fn = nn.value_and_grad(self.model, self.loss)
        log(f"{sum(v.size for _, v in tree_flatten(self.model.trainable_parameters())):,} parameters; "
            f"{len(self.train):,} train and {len(self.valid):,} validation records")
        history = [{"step": 0, **self.validate()}]
        log(json.dumps(history[-1]))
        best = history[-1]["loss"]
        step, tokens_seen, started, window = 0, 0, time.perf_counter(), []
        while step < config.steps:
            for group in self._groups(self.train, shuffle=True):
                step += 1
                batch = self._batch(group)
                loss, grads = step_fn(self.model, *batch)
                grads, _ = optim.clip_grad_norm(grads, 1.0)
                optimizer.update(self.model, grads)
                mx.eval(self.model.parameters(), optimizer.state, loss)
                window.append(float(loss))
                tokens_seen += batch[0].size
                if step % config.eval_every == 0 or step == config.steps:
                    entry = {"step": step, "train_loss": float(np.mean(window)),
                             "tokens_per_second": tokens_seen / (time.perf_counter() - started), **self.validate()}
                    window = []
                    history.append(entry)
                    log(json.dumps(entry))
                    if entry["loss"] < best:
                        best = entry["loss"]
                        self.save(history)
                if step == config.steps:
                    break
        if best == history[0]["loss"]:
            self.save(history)
        return history
