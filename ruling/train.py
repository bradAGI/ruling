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

Three optional terms move the objective toward the calibrated-decision recipes
TypeSafe and Cloudflare describe, all off by default:

- label smoothing mixes a little of the uniform distribution into the target,
  so the model is never trained to put all its mass on one option;
- a Brier term, the squared distance between the predicted and target
  distributions, which punishes confident errors less steeply than log loss
  and so pulls the hard, wrong answers back toward the middle;
- a confidence-ranking term, which within each batch penalizes any wrong
  answer reported more confidently than a right one. This targets the
  ordering of confidences, which is what coverage at a fixed error budget
  measures and what a temperature cannot change.
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
from mlx.utils import tree_flatten, tree_unflatten
from mlx_lm.tuner.trainer import grad_checkpoint
from mlx_lm.tuner.utils import linear_to_lora_layers

from ruling.calibration import Calibration, coverage_at_error, expected_calibration_error
from ruling.engine import Engine, option_logits
from ruling.evals import TARGET, Record, label_index, load_dataset
from ruling.prompt import render_question
from ruling.questions import Choice, Noul, Score, option_keys, render_text, state_images
from ruling.vision import VISION_PREFIX, VisionEngine


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
    label_smoothing: float = 0.0
    brier_weight: float = 0.0
    ranking_weight: float = 0.0
    ranking_margin: float = 0.05
    from_adapter: Path | None = None
    max_image_tokens: int = 1024
    """Tokens one image may become on a `vision:` model; keep it equal to the server's RULING_MAX_IMAGE_TOKENS."""


CHECKPOINT = "checkpoint"
TRAINING = "training.json"
# Rendering happens in chunks of this many batches, which are then sorted by length and cut into batches.
CHUNK_BATCHES = 64


@dataclass(frozen=True)
class Example:
    record: int
    question_id: str
    weight: float
    """The record's weight: its share of a batch's update relative to the other rows."""


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


def objective(log_probs: mx.array, mask: mx.array, targets: mx.array, label_smoothing: float = 0.0,
              brier_weight: float = 0.0, ranking_weight: float = 0.0, ranking_margin: float = 0.05,
              weights: mx.array | None = None) -> mx.array:
    """Cross-entropy against the (optionally smoothed) target, plus the optional Brier and ranking terms.

    Rows are averaged by `weights` (one per row, summing to more than zero; uniform when None), so the update
    is the weighted mean of the batch. With every term off this is exactly the original soft-target
    cross-entropy. The ranking term compares rows pairwise and takes no weights.
    """
    if weights is None:
        weights = mx.ones(log_probs.shape[0])

    def weighted_mean(per_row: mx.array) -> mx.array:
        return (weights * per_row).sum() / weights.sum()

    options = mask.sum(axis=-1, keepdims=True).astype(mx.float32)
    smoothed = (1.0 - label_smoothing) * targets + mx.where(mask, label_smoothing / options, 0.0)
    total = weighted_mean(-mx.where(mask, smoothed * log_probs, 0.0).sum(axis=-1))
    if brier_weight == 0.0 and ranking_weight == 0.0:
        return total
    probs = mx.where(mask, mx.exp(log_probs), 0.0)
    if brier_weight:
        total = total + brier_weight * weighted_mean(((probs - targets) ** 2).sum(axis=-1))
    if ranking_weight:
        confidence = probs.max(axis=-1)
        right = mx.stop_gradient((probs.argmax(axis=-1) == targets.argmax(axis=-1)).astype(mx.float32))
        # pairs[i, j] is 1 where answer i was right and answer j was wrong.
        pairs = right[:, None] * (1.0 - right)[None, :]
        violations = mx.maximum(confidence[None, :] - confidence[:, None] + ranking_margin, 0.0) * pairs
        total = total + ranking_weight * violations.sum() / mx.maximum(pairs.sum(), 1.0)
    return total


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


def config_json(config: TrainConfig) -> dict:
    """The config as JSON-ready values, paths as strings."""
    return {**asdict(config), "data": [str(p) for p in config.data], "holdout": [str(p) for p in config.holdout],
            "output": str(config.output), "from_adapter": None if config.from_adapter is None else str(config.from_adapter)}


def config_mismatch(saved: dict, requested: dict) -> list[str]:
    """The fields a resumed run would change, other than `steps`, which may be raised to continue a finished run."""
    return sorted(k for k in set(saved) | set(requested) if k != "steps" and saved.get(k) != requested.get(k))


def progress(step: int, steps: int, elapsed: float) -> dict:
    """How far a run is and when it should end, rating the steps left at the run's pace so far.

    `elapsed` covers every step since the run began, across resumes, so the pace is the whole run's."""
    return {"elapsed_seconds": elapsed, "progress": step / steps, "eta_seconds": (steps - step) * elapsed / step}


class WandbSink:
    """Mirrors every history entry to a Weights & Biases run, opened on entering and finished on leaving,
    marked failed when training raised. The package is imported here only, so it stays optional."""

    def __init__(self, project: str | None, config: TrainConfig):
        try:
            import wandb
        except ImportError as error:
            raise ImportError("Weights & Biases logging needs the wandb package: pip install 'ruling[wandb]'") from error
        self.wandb, self.project, self.config = wandb, project, config
        self.run = None

    def __enter__(self) -> "WandbSink":
        self.run = self.wandb.init(project=self.project, config=config_json(self.config))
        return self

    def __call__(self, entry: dict) -> None:
        self.run.log(entry, step=entry["step"])

    def __exit__(self, kind, value, traceback) -> None:
        self.run.finish(exit_code=0 if kind is None else 1)


def save_checkpoint(directory: Path, weights: dict[str, mx.array], optimizer_state: dict, state: dict) -> None:
    """Everything a run needs to continue from its next step: LoRA weights, optimizer state and the trainer's bookkeeping."""
    directory.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(directory / "adapters.safetensors"), weights)
    mx.save_safetensors(str(directory / "optimizer.safetensors"), dict(tree_flatten(optimizer_state)))
    (directory / "state.json").write_text(json.dumps(state, indent=2) + "\n")


def load_checkpoint(directory: Path) -> tuple[dict[str, mx.array], dict, dict]:
    if not (directory / "state.json").exists():
        raise FileNotFoundError(f"no checkpoint to resume in {directory}")
    weights = dict(mx.load(str(directory / "adapters.safetensors")))
    optimizer_state = tree_unflatten(list(mx.load(str(directory / "optimizer.safetensors")).items()))
    return weights, optimizer_state, json.loads((directory / "state.json").read_text())


class BatchStream:
    """Endless shuffled batches, grouped by length to limit padding, resumable to the exact batch.

    An epoch is one permutation of the examples, rendered a chunk at a time; each chunk is sorted by length, cut into
    batches, and the batches shuffled. Augmentation draws from the same generator, so the generator's state at the
    start of the epoch and of the current chunk, with the chunk's offset and how many of its batches went out,
    determine every later batch exactly.
    """

    def __init__(self, count: int, size: int, rng: np.random.Generator, render, batch):
        self.count, self.size, self.rng, self.render, self.batch = count, size, rng, render, batch
        self.order = None
        self.epoch_rng = self.chunk_rng = None
        self.start = self.cursor = 0
        self.batches = []

    def __next__(self):
        while self.cursor >= len(self.batches):
            width = self.size * CHUNK_BATCHES
            if self.order is None or self.start + width >= len(self.order):
                self.epoch_rng = self.rng.bit_generator.state
                self.order = self.rng.permutation(self.count)
                self.start = 0
            else:
                self.start += width
            self.chunk_rng = self.rng.bit_generator.state
            self._render_chunk()
        batch = self.batches[self.cursor]
        self.cursor += 1
        return batch

    def _render_chunk(self) -> None:
        chunk = [self.render(i) for i in self.order[self.start:self.start + self.size * CHUNK_BATCHES]]
        chunk = sorted((c for c in chunk if c is not None), key=lambda c: len(c[0]))
        groups = [chunk[i:i + self.size] for i in range(0, len(chunk) - self.size + 1, self.size)]
        self.batches = [self.batch(groups[g]) for g in self.rng.permutation(len(groups))]
        self.cursor = 0

    def state(self) -> dict:
        return {"epoch_rng": self.epoch_rng, "chunk_rng": self.chunk_rng, "start": self.start, "cursor": self.cursor}

    def restore(self, state: dict) -> None:
        self.epoch_rng, self.chunk_rng, self.start = state["epoch_rng"], state["chunk_rng"], state["start"]
        self.rng.bit_generator.state = self.epoch_rng
        self.order = self.rng.permutation(self.count)
        self.rng.bit_generator.state = self.chunk_rng
        self._render_chunk()
        self.cursor = state["cursor"]


class Trainer:
    def __init__(self, config: TrainConfig, resume: bool = False):
        self.config = config
        self.checkpoint = None
        if resume:
            self.checkpoint = load_checkpoint(config.output / CHECKPOINT)
            _, _, state = self.checkpoint
            mismatch = config_mismatch(state["config"], config_json(config))
            if mismatch:
                raise ValueError(f"cannot resume {config.output}: the checkpoint was trained with different {', '.join(mismatch)}")
        self.rng = np.random.default_rng(config.seed)
        records = [r for path in config.data for r in load_dataset(path)]
        held_out = [r for path in config.holdout for r in load_dataset(path)]
        overlap = holdout_overlap(records, held_out)
        if overlap:
            raise ValueError(f"{overlap} training states also appear in held-out evaluation data")
        if not config.model.startswith(VISION_PREFIX) and any(state_images(r.state) for r in records):
            raise ValueError(f"the records hold images and {config.model} reads text only; "
                             f"train a {VISION_PREFIX} model on them")
        self.train_records, self.valid_records = split_records(records, config.valid_fraction)

        self.lora = {"rank": config.lora_rank, "scale": config.lora_scale, "dropout": config.lora_dropout}
        if config.from_adapter is not None:
            self._check_adapter(config.from_adapter)

        mx.random.seed(config.seed)
        if config.model.startswith(VISION_PREFIX):
            self.engine = VisionEngine(config.model, Calibration(), config.max_tokens, 1, False, config.max_image_tokens)
        else:
            self.engine = Engine(config.model, Calibration(), config.max_tokens, rotations=1, prior_debias=False)
        model = self.engine.model
        model.freeze()
        linear_to_lora_layers(model, config.lora_layers, self.lora)
        if config.grad_checkpoint:
            grad_checkpoint(model.layers[0])
        if config.from_adapter is not None:
            model.load_weights(str(config.from_adapter / "adapters.safetensors"), strict=False)
            mx.eval(model.parameters())

        self.train_examples = self._examples(self.train_records)
        valid = self._examples(self.valid_records)
        self.valid_examples = [valid[i] for i in self.rng.permutation(len(valid))[: config.valid_examples]]
        if not self.train_examples or not self.valid_examples:
            raise ValueError("training and validation both need labeled questions")

    def _check_adapter(self, adapter: Path) -> None:
        """An adapter to start from must have been trained with the requested LoRA settings, or its weights fit nothing."""
        saved = json.loads((adapter / "adapter_config.json").read_text())
        wanted = {"fine_tune_type": "lora", "num_layers": self.config.lora_layers, "lora_parameters": self.lora}
        if saved != wanted:
            raise ValueError(f"{adapter} was trained with {saved}, not the requested {wanted}")

    def _examples(self, records: list[Record]) -> list[Example]:
        """One example per labeled question the model can answer; a weightless record would not move the update."""
        examples = []
        for index, record in enumerate(records):
            if record.weight == 0:
                continue
            for qid, question in record.questions.items():
                cardinality = len(option_keys(question))
                if cardinality <= self.engine.codes.max_options and target_vector(record, qid) is not None:
                    examples.append(Example(index, qid, record.weight))
        return examples

    def _render(self, records: list[Record], example: Example, augment: bool):
        """The example's tokens, its target over the presented options, its weight, and its state when the
        state holds images (read again when its batch runs, so a chunk of batches never holds their pixels)."""
        record = records[example.record]
        question = record.questions[example.question_id]
        presented, rotation = permuted(question, self.rng) if augment else (question, 0)
        rendered = render_question(presented, self.engine.codes.codes, rotation)
        prefix = self.engine._prefix(record.state)
        tokens = prefix.tokens + self.engine._encode(rendered.text + prefix.tail)
        if len(tokens) > self.config.max_tokens:
            return None
        target = target_vector(record, example.question_id)
        seen = record.state if prefix.vision_inputs is not None else None
        return tokens, [target[key] for key in rendered.keys], example.weight, seen

    def _batch(self, rendered: list[tuple[list[int], list[float], float, object]]):
        width = math.ceil(max(len(t) for t, _, _, _ in rendered) / 32) * 32
        options = max(len(target) for _, target, _, _ in rendered)
        pad = self.engine._pad_id
        tokens = mx.array([t + [pad] * (width - len(t)) for t, _, _, _ in rendered])
        last = mx.array([len(t) - 1 for t, _, _, _ in rendered])
        ids = mx.array([self.engine.codes.token_ids[:options] for _ in rendered])
        mask = mx.array([[i < len(target) for i in range(options)] for _, target, _, _ in rendered])
        targets = mx.array([target + [0.0] * (options - len(target)) for _, target, _, _ in rendered], dtype=mx.float32)
        weights = mx.array([weight for _, _, weight, _ in rendered], dtype=mx.float32)
        return tokens, last, ids, mask, targets, weights, [seen for _, _, _, seen in rendered]

    def _vision(self, tokens: mx.array, last: mx.array, states: list) -> tuple[mx.array | None, mx.array | None]:
        """Input embeddings with each row's image features in place, and every row's multimodal positions, on a
        vision model; (None, None) on a text model. A batch without images keeps its token embeddings. The vision
        tower is frozen: nothing here takes a gradient."""
        if not self.engine.reads_images:
            return None, None
        width = tokens.shape[1]
        positions = mx.broadcast_to(mx.arange(width)[None, None, :], (3, tokens.shape[0], width))
        if all(state is None for state in states):
            # A vision model's attention takes explicit positions whenever no cache supplies them.
            return None, positions
        vlm = self.engine.vlm
        embeds = vlm.language_model.model.embed_tokens(tokens)
        rows, places = [], []
        for row, state in enumerate(states):
            if state is None:
                rows.append(embeds[row])
                places.append(positions[:, row])
                continue
            length = int(last[row]) + 1
            images = self.engine._prefix(state).vision_inputs
            ids = tokens[row:row + 1, :length]
            features = vlm.get_input_embeddings(ids, images["pixel_values"], image_grid_thw=images["image_grid_thw"],
                                                mask=mx.ones(ids.shape, dtype=mx.int32))
            # Padding sits after the last real token, so its embeddings and positions are never read.
            rows.append(mx.concatenate([features.inputs_embeds[0], embeds[row, length:]], axis=0))
            row_positions = features.position_ids[:, 0]
            tail = row_positions.max() + 1 + mx.arange(width - length)
            places.append(mx.concatenate([row_positions, mx.broadcast_to(tail[None], (3, width - length))], axis=1))
        return mx.stop_gradient(mx.stack(rows)), mx.stack(places, axis=1)

    @staticmethod
    def _log_probs(model, tokens, last, ids, mask, embeds=None, positions=None):
        logits = mx.where(mask, option_logits(model, tokens, None, last, ids, positions, embeds), -mx.inf)
        return logits - mx.logsumexp(logits, axis=-1, keepdims=True)

    def loss(self, model, tokens, last, ids, mask, targets, weights, embeds=None, positions=None):
        log_probs = self._log_probs(model, tokens, last, ids, mask, embeds, positions)
        config = self.config
        return objective(log_probs, mask, targets, config.label_smoothing, config.brier_weight,
                         config.ranking_weight, config.ranking_margin, weights)

    def _batches(self) -> BatchStream:
        return BatchStream(len(self.train_examples), self.config.batch_size, self.rng,
                           lambda i: self._render(self.train_records, self.train_examples[i], augment=True), self._batch)

    def validate(self) -> dict:
        model = self.engine.model
        model.eval()
        losses, top, correct = [], [], []
        size = self.config.batch_size
        rendered = [r for r in (self._render(self.valid_records, e, augment=False) for e in self.valid_examples) if r is not None]
        rendered.sort(key=lambda c: len(c[0]))
        for start in range(0, len(rendered), size):
            group = rendered[start:start + size]
            tokens, last, ids, mask, targets, _, states = self._batch(group)
            log_probs = self._log_probs(model, tokens, last, ids, mask, *self._vision(tokens, last, states))
            ce = -mx.where(mask, targets * log_probs, 0.0).sum(axis=-1)
            mx.eval(ce, log_probs)
            probs = np.exp(np.where(np.array(mask), np.array(log_probs), -np.inf))
            for row, (_, target, _, _), loss in zip(probs, group, np.array(ce)):
                row = row[: len(target)]
                losses.append(float(loss))
                top.append(float(row.max()))
                correct.append(int(row.argmax()) == int(np.argmax(target)))
        model.train()
        return {"loss": float(np.mean(losses)), "accuracy": float(np.mean(correct)),
                "expected_calibration_error": expected_calibration_error(top, correct),
                "coverage_at_5pct_error": coverage_at_error(top, correct, 0.05), "count": len(losses)}

    def save(self, directory: Path, history: list[dict]) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        weights = dict(tree_flatten(self.engine.model.trainable_parameters()))
        mx.save_safetensors(str(directory / "adapters.safetensors"), weights)
        (directory / "adapter_config.json").write_text(json.dumps(
            {"fine_tune_type": "lora", "num_layers": self.config.lora_layers, "lora_parameters": self.lora}, indent=2) + "\n")
        data = [{"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in self.config.data]
        meta = {"config": {**config_json(self.config), "data": data},
                "base_revision": self.engine.revision, "train_records": len(self.train_records),
                "train_examples": len(self.train_examples), "valid_examples": len(self.valid_examples), "history": history}
        (directory / TRAINING).write_text(json.dumps(meta, indent=2) + "\n")

    def run(self, log=print, sink=None) -> list[dict]:
        """Train to `steps`, returning the history; `sink` receives each history entry as it is made."""
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

        def record(entry: dict) -> None:
            history.append(entry)
            log(json.dumps(entry))
            if sink is not None:
                sink(entry)

        batches = self._batches()
        if self.checkpoint is None:
            history = []
            record({"step": 0, **self.validate()})
            self.save(config.output, history)
            first, tokens_seen, elapsed = 1, 0, 0.0
        else:
            weights, optimizer_state, state = self.checkpoint
            model.load_weights(list(weights.items()), strict=False)
            optimizer.state = optimizer_state
            mx.eval(model.parameters(), optimizer.state)
            batches.restore(state["stream"])
            history, tokens_seen, elapsed = state["history"], state["tokens_seen"], state["elapsed"]
            first = state["step"] + 1
            log(f"resuming from step {state['step']}")
        best = min(entry["loss"] for entry in history)
        started, window = time.perf_counter() - elapsed, []
        for step in range(first, config.steps + 1):
            # Dropout draws from a per-step key, so a resumed run masks exactly as the uninterrupted one would.
            mx.random.seed(int(np.random.SeedSequence((config.seed, step)).generate_state(1)[0]))
            tokens, last, ids, mask, targets, weights, states = next(batches)
            loss, grads = step_fn(model, tokens, last, ids, mask, targets, weights, *self._vision(tokens, last, states))
            grads, _ = optim.clip_grad_norm(grads, config.max_grad_norm)
            optimizer.update(model, grads)
            mx.eval(model.trainable_parameters(), optimizer.state, loss)
            window.append(float(loss))
            tokens_seen += tokens.size
            if step % config.eval_every == 0 or step == config.steps:
                elapsed = time.perf_counter() - started
                entry = {"step": step, "train_loss": float(np.mean(window)), "tokens_per_second": tokens_seen / elapsed,
                         **progress(step, config.steps, elapsed), **self.validate()}
                window = []
                record(entry)
                if entry["loss"] < best:
                    best = entry["loss"]
                    self.save(config.output, history)
                else:
                    meta = json.loads((config.output / TRAINING).read_text())
                    meta["history"] = history
                    (config.output / TRAINING).write_text(json.dumps(meta, indent=2) + "\n")
                save_checkpoint(config.output / CHECKPOINT, dict(tree_flatten(model.trainable_parameters())), optimizer.state,
                                {"step": step, "tokens_seen": tokens_seen, "elapsed": elapsed, "stream": batches.state(),
                                 "history": history, "config": config_json(config)})
        return history
