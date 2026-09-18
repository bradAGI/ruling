"""A decision encoder: every typed question answered in one bidirectional pass.

The state and every question go into a single sequence. Each option gets a
`[MASK]` slot followed by its text, and a linear head scores the slot's hidden
state; a softmax over one question's slots is that question's distribution.
Attention is restricted so every question sees the state and its own text,
every option sees only itself, its question, and the state, and all options of
a question start at the same position. Answers are therefore identical
whatever other questions are asked, in any order, with options in any order.
Nothing is decoded and there are no letter codes.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten

from ruling.calibration import Calibration, Provenance
from ruling.engine import InputTooLong, RawScore, Scored, answer
from ruling.modernbert import Config, Layer, MaskedLM, load
from ruling.questions import (
    MAX_CHOICE_OPTIONS,
    Choice,
    Noul,
    Score,
    State,
    SystemOneRequest,
    SystemOneResponse,
    Usage,
    option_keys,
    render_text,
)

CHECKPOINT_FILE = "decision.json"
WEIGHTS_FILE = "weights.safetensors"

__all__ = ["InputTooLong", "Special", "Packed", "pack", "attention_mask", "DecisionModel", "DecisionEngine"]


@dataclass(frozen=True)
class Special:
    cls: int
    sep: int
    mask: int
    pad: int


@dataclass(frozen=True)
class Packed:
    tokens: list[int]
    positions: list[int]
    segments: list[tuple[int, int]]
    """Per token: (0, 0) for the state, (q, 0) for question q's text, (q, o) for its option o."""
    slots: dict[str, list[tuple[str, int]]]
    question_index: dict[str, int]


def question_text(question: Choice | Score | Noul) -> str:
    instructions = render_text(question.instructions)
    if isinstance(question, Choice):
        return f"Choose the best option. {instructions}"
    if isinstance(question, Score):
        return f"Choose the level that fits, from lowest to highest. {instructions}"
    return f"Is this statement true? {instructions}"


def option_text(question: Choice | Score | Noul, key: str) -> str:
    if isinstance(question, Choice):
        description = question.criteria[key]
        return key if description is None else f"{key}: {render_text(description)}"
    if isinstance(question, Score):
        index = int(key)
        return f"level {index + 1} of {len(question.criteria)}: {render_text(question.criteria[index])}"
    criteria = question.criteria
    if key == "yes":
        return "true" if criteria is None or criteria.true is None else f"true: {render_text(criteria.true)}"
    return "false" if criteria is None or criteria.false is None else f"false: {render_text(criteria.false)}"


def pack(state: State, questions: dict[str, Choice | Score | Noul], encode: Callable[[str], list[int]],
         special: Special, max_tokens: int) -> Packed:
    body = state if isinstance(state, str) else json.dumps(state, indent=2, ensure_ascii=False)
    tokens = [special.cls, *encode(body), special.sep]
    positions = list(range(len(tokens)))
    segments = [(0, 0)] * len(tokens)
    state_length = len(tokens)
    slots: dict[str, list[tuple[str, int]]] = {}
    index: dict[str, int] = {}
    for q, (qid, question) in enumerate(questions.items(), start=1):
        if isinstance(question, Choice) and len(question.criteria) > MAX_CHOICE_OPTIONS:
            raise ValueError(f"question {qid!r} has more than {MAX_CHOICE_OPTIONS} options")
        index[qid] = q
        text = [*encode(question_text(question)), special.sep]
        tokens += text
        positions += range(state_length, state_length + len(text))
        segments += [(q, 0)] * len(text)
        start = state_length + len(text)
        slots[qid] = []
        for o, key in enumerate(option_keys(question), start=1):
            option = [special.mask, *encode(option_text(question, key)), special.sep]
            slots[qid].append((key, len(tokens)))
            tokens += option
            positions += range(start, start + len(option))
            segments += [(q, o)] * len(option)
    if len(tokens) > max_tokens:
        raise InputTooLong(f"request is {len(tokens)} tokens; limit is {max_tokens}")
    return Packed(tokens, positions, segments, slots, index)


def attention_mask(segment_lists: list[list[tuple[int, int]]]) -> np.ndarray:
    """Boolean [batch, width, width]; padding positions neither attend nor are attended to."""
    width = max(len(s) for s in segment_lists)
    mask = np.zeros((len(segment_lists), width, width), dtype=bool)
    for b, segments in enumerate(segment_lists):
        seg = np.array(segments)
        q, o = seg[:, 0], seg[:, 1]
        sees_state = (q == 0)[None, :]
        own_question_text = (q[:, None] == q[None, :]) & (o == 0)[None, :]
        own_option = (q[:, None] == q[None, :]) & (o[:, None] == o[None, :])
        n = len(segments)
        mask[b, :n, :n] = sees_state | own_question_text | own_option
    return mask


class DecisionModel(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.lm = MaskedLM(config)
        self.score = nn.Linear(config.hidden_size, 1)
        self.score.weight = mx.zeros_like(self.score.weight)
        self.score.bias = mx.zeros_like(self.score.bias)

    def __call__(self, tokens: mx.array, positions: mx.array, mask: mx.array, rows: mx.array, cols: mx.array) -> mx.array:
        """One score per requested slot, where slot n is at (rows[n], cols[n])."""
        hidden = self.lm.encoder(tokens, positions, mask)
        return self.score(self.lm.transform(hidden[rows, cols]))[:, 0].astype(mx.float32)


@dataclass(frozen=True)
class Checkpoint:
    base: str
    revision: str

    def save(self, directory: Path, model: DecisionModel, extra: dict) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        mx.save_safetensors(str(directory / WEIGHTS_FILE), dict(tree_flatten(model.parameters())))
        (directory / CHECKPOINT_FILE).write_text(json.dumps({**asdict(self), **extra}, indent=2) + "\n")


def checkpoint_layers(model: DecisionModel) -> None:
    """Recompute each encoder layer's activations in the backward pass instead of storing them."""
    call = Layer.__call__

    def checkpointed(layer, *args):
        def inner(params, *args):
            layer.update(params)
            return call(layer, *args)

        return mx.checkpoint(inner)(layer.trainable_parameters(), *args)

    Layer.__call__ = checkpointed


def is_checkpoint(model: str) -> bool:
    return (Path(model) / CHECKPOINT_FILE).exists()


def load_model(base: str, revision: str, checkpoint: Path | None):
    """The base encoder with a fresh zero-initialized head, or a trained checkpoint on top of it."""
    lm, tokenizer, config = load(base, revision)
    model = DecisionModel(config)
    model.lm = lm
    if checkpoint is not None:
        model.load_weights(str(checkpoint / WEIGHTS_FILE), strict=True)
    mx.eval(model.parameters())
    special = Special(cls=tokenizer.cls_token_id, sep=tokenizer.sep_token_id, mask=tokenizer.mask_token_id, pad=tokenizer.pad_token_id)
    return model, tokenizer, special


def batch_inputs(packs: list[Packed], pad: int) -> tuple[mx.array, mx.array, mx.array]:
    width = max(len(p.tokens) for p in packs)
    tokens = mx.array([p.tokens + [pad] * (width - len(p.tokens)) for p in packs])
    positions = mx.array([p.positions + [0] * (width - len(p.positions)) for p in packs])
    return tokens, positions, mx.array(attention_mask([p.segments for p in packs]))


class DecisionEngine:
    """Serves a trained decision checkpoint through the same interface as the decoder engine."""

    rotations = 1
    prior_debias = False
    max_options = MAX_CHOICE_OPTIONS

    def __init__(self, checkpoint: Path, calibration: Calibration, max_input_tokens: int):
        meta = json.loads((checkpoint / CHECKPOINT_FILE).read_text())
        self.model_id = str(checkpoint)
        self.revision = meta["revision"]
        self.adapter = hashlib.sha256((checkpoint / WEIGHTS_FILE).read_bytes()).hexdigest()[:16]
        self.model, self.tokenizer, self.special = load_model(meta["base"], meta["revision"], checkpoint)
        self.max_input_tokens = min(max_input_tokens, self.model.lm.encoder.config.max_position_embeddings)
        # The encoder packs state and every question into one sequence, so a branch is the whole input.
        self.max_branch_tokens = self.max_input_tokens
        self.model.eval()
        calibration.check(self.provenance)
        self.calibration = calibration

    @property
    def provenance(self) -> Provenance:
        return Provenance(self.model_id, self.revision, self.rotations, self.prior_debias, self.adapter)

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def score(self, state: State, questions: dict[str, Choice | Score | Noul]) -> Scored:
        packed = pack(state, questions, self.encode, self.special, self.max_input_tokens)
        tokens, positions, mask = batch_inputs([packed], self.special.pad)
        order = [(qid, key, col) for qid, slots in packed.slots.items() for key, col in slots]
        scores = self.model(tokens, positions, mask, mx.zeros((len(order),), dtype=mx.int32),
                            mx.array([col for _, _, col in order]))
        values = np.array(scores, dtype=np.float64)
        raw = {}
        cursor = 0
        for qid, question in questions.items():
            count = len(packed.slots[qid])
            logits = values[cursor:cursor + count]
            raw[qid] = RawScore(keys=option_keys(question), logits=logits - np.log(np.exp(logits - logits.max()).sum()) - logits.max())
            cursor += count
        return Scored(raw=raw, input_tokens=len(packed.tokens))

    def evaluate(self, request: SystemOneRequest) -> SystemOneResponse:
        scored = self.score(request.state, request.questions)
        answers = {qid: answer(request.questions[qid], scored.raw[qid], self.calibration) for qid in request.questions}
        return SystemOneResponse(model=self.model_id, answers=answers, usage=Usage(input_tokens=scored.input_tokens))

    def answer(self, question, raw: RawScore):
        return answer(question, raw, self.calibration)
