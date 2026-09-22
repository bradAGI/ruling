"""Score typed questions against a state with a local MLX model.

No text is generated. The state prefix is encoded once into a KV cache. Every
question suffix (one per rotation of its answer order) is then scored in
batched forward passes against replicas of that cache, and the logits at each
suffix's answer position are restricted to the allowed answer codes.

Positional bias is removed two ways: a content-free letter prior measured once
per answer count is subtracted from every log-probability, and the remaining
orderings are averaged in log space.
"""

from __future__ import annotations

import hashlib
import threading
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import mlx.core as mx
import numpy as np
from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.errors import HfHubHTTPError, HFValidationError
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

from ruling.calibration import Calibration, Provenance, choice_confidence, log_softmax, score_confidence, softmax
from ruling.codes import AnswerCodes
from ruling.prompt import (
    PRIOR_STATE,
    SPLIT_MARKER,
    SYSTEM_PROMPT,
    prior_question,
    render_question,
    render_state,
)
from ruling.questions import (
    MAX_CHOICE_OPTIONS,
    Choice,
    ChoiceAnswer,
    Noul,
    NoulAnswer,
    Score,
    ScoreAnswer,
    State,
    SystemOneRequest,
    SystemOneResponse,
    Usage,
    option_keys,
    render_text,
)

# Prefix tokens are pushed through the model in chunks so the full-vocabulary
# logits for a long state never have to exist all at once.
PREFILL_STEP = 512

# Suffix passes are batched until batch_size * (prefix + padded suffix) tokens
# would exceed this budget, which bounds the memory of the replicated cache.
BATCH_TOKEN_BUDGET = 65_536

# For a Choice with more options than single-token codes, every option is first
# scored on its own as a Noul and the strongest few go to a final Choice.
STAGE_TWO_CANDIDATES = 8

# Qwen3-family templates emit an empty thinking block when thinking is off.
# Templates without the variable ignore it.
TEMPLATE_KWARGS = {"enable_thinking": False}


class InputTooLong(ValueError):
    """Raised with a message that names the maximum context length, the phrase clients
    such as the Decision Index harness look for to classify a rejection as a capacity
    limit rather than a failure."""


@dataclass(frozen=True)
class RawScore:
    """Debiased log-probabilities over `keys`, in canonical key order, before calibration."""

    keys: list[str]
    logits: np.ndarray


@dataclass(frozen=True)
class Scored:
    raw: dict[str, RawScore]
    input_tokens: int


@dataclass(frozen=True)
class _Pass:
    question_id: str
    keys: list[str]
    tokens: list[int]


def answer(question: Choice | Score | Noul, raw: RawScore, calibration: Calibration) -> ChoiceAnswer | ScoreAnswer | NoulAnswer:
    """A typed answer from calibrated probabilities over the question's canonical keys."""
    probs = softmax(raw.logits, calibration.temperature(question.type))
    distribution = {key: float(p) for key, p in zip(raw.keys, probs)}
    if isinstance(question, Choice):
        return ChoiceAnswer(choice=raw.keys[int(probs.argmax())], probabilities=distribution,
                            confidence=choice_confidence(probs))
    if isinstance(question, Score):
        return ScoreAnswer(
            score=float((probs * np.arange(len(probs))).sum()),
            legend={str(i): render_text(level) for i, level in enumerate(question.criteria)},
            probabilities=distribution,
            confidence=score_confidence(probs),
        )
    return NoulAnswer(noul=distribution["yes"])


def option_logits(model, tokens: mx.array, cache, last: mx.array, option_ids: mx.array) -> mx.array:
    """Logits for each row's answer codes at its last real position, shape [batch, options].

    Only the hidden state at the answer position goes through the output head,
    so a batch never materializes full-vocabulary logits for every token. This
    follows the mlx-lm causal layout: an optional `language_model` wrapper, a
    `.model` that returns final hidden states, and either a separate `lm_head`
    or tied input embeddings.
    """
    text = getattr(model, "language_model", model)
    hidden = text.model(tokens, cache)
    picked = hidden[mx.arange(tokens.shape[0]), last]
    tied = getattr(text, "tie_word_embeddings", None)
    if tied is None:
        tied = text.args.tie_word_embeddings
    logits = text.model.embed_tokens.as_linear(picked) if tied else text.lm_head(picked)
    return mx.take_along_axis(logits, option_ids, axis=-1).astype(mx.float32)


def adapter_digest(adapter: Path | None) -> str | None:
    """A short content hash of LoRA weights, so calibration and reports name the exact adapter."""
    if adapter is None:
        return None
    return hashlib.sha256((adapter / "adapters.safetensors").read_bytes()).hexdigest()[:16]


def model_released(model_id: str) -> str:
    """The date the served weights were published, as `YYYY-MM-DD`.

    Read from the Hugging Face repository when the model came from one and it is
    reachable. A local directory, or an unreachable repository, falls back to the
    date the weights landed on this machine.
    """
    local = Path(model_id)
    if not local.exists():
        try:
            return HfApi().model_info(model_id).last_modified.date().isoformat()
        except (HfHubHTTPError, HFValidationError, OSError, AttributeError):
            local = Path(snapshot_download(model_id, allow_patterns=["config.json"], local_files_only=True))
    return date.fromtimestamp(local.stat().st_mtime).isoformat()


def model_revision(model_id: str) -> str | None:
    """The Hugging Face commit the weights came from, or None for a local directory."""
    if Path(model_id).exists():
        return None
    # mlx_lm.load has already fetched the weights; only the cached config is consulted here.
    path = Path(snapshot_download(model_id, allow_patterns=["config.json"], local_files_only=True))
    return path.name if path.parent.name == "snapshots" else None


class Engine:
    def __init__(
        self,
        model_id: str,
        calibration: Calibration,
        max_input_tokens: int,
        rotations: int = 1,
        prior_debias: bool = True,
        adapter: Path | None = None,
        max_branch_tokens: int | None = None,
    ):
        if rotations < 1:
            raise ValueError("rotations must be at least 1")
        self.model_id = model_id
        self.max_input_tokens = max_input_tokens
        self.max_branch_tokens = max_branch_tokens or max_input_tokens
        self.rotations = rotations
        self.prior_debias = prior_debias
        self.adapter = adapter_digest(adapter)
        self.model, self.tokenizer = load(model_id, adapter_path=str(adapter) if adapter else None)
        self.revision = model_revision(model_id)
        self.codes = AnswerCodes.from_tokenizer(self.tokenizer)
        self._pad_id = self.tokenizer.pad_token_id
        if self._pad_id is None:
            self._pad_id = self.tokenizer.eos_token_id
        self._system_role = self._template_accepts_system_role()
        self.max_options = MAX_CHOICE_OPTIONS
        self._lock = threading.Lock()
        self._priors: dict[int, np.ndarray] = {}
        calibration.check(self.provenance)
        self.calibration = calibration

    @property
    def provenance(self) -> Provenance:
        return Provenance(self.model_id, self.revision, self.rotations, self.prior_debias, self.adapter)

    # ---- prompt assembly -------------------------------------------------

    def split_prompt(self, state: State, thinking: bool = False) -> tuple[str, str]:
        """Return the chat-templated prefix (through the state) and the template tail.

        With `thinking`, the tail ends where the model starts reasoning instead of answering.
        """
        rendered_state = render_state(state)
        if SPLIT_MARKER in rendered_state:
            raise ValueError("state contains the reserved split marker")
        if self._system_role:
            messages = [{"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": rendered_state + SPLIT_MARKER}]
        else:
            messages = [{"role": "user", "content": SYSTEM_PROMPT + "\n\n" + rendered_state + SPLIT_MARKER}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=thinking
        )
        prefix, tail = text.split(SPLIT_MARKER)
        return prefix, tail

    def _template_accepts_system_role(self) -> bool:
        try:
            self.tokenizer.apply_chat_template(
                [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}],
                tokenize=False, add_generation_prompt=True, **TEMPLATE_KWARGS,
            )
        except Exception:  # templates raise jinja TemplateError subclasses; the type varies by model
            return False
        return True

    # ---- scoring -----------------------------------------------------------

    def score(self, state: State, questions: dict[str, Choice | Score | Noul]) -> Scored:
        large = {qid: q for qid, q in questions.items()
                 if isinstance(q, Choice) and len(q.criteria) > self.codes.max_options}
        tokens = 0
        reduced = dict(questions)
        finalists: dict[str, list[str]] = {}
        if large:
            stage_one = self._score_plain(state, {
                f"{qid}\x00{key}": Noul(instructions={"question": q.instructions, "candidate": {key: q.criteria[key]}})
                for qid, q in large.items() for key in q.criteria
            })
            tokens += stage_one.input_tokens
            for qid, q in large.items():
                ranked = sorted(q.criteria, key=lambda key: -stage_one.raw[f"{qid}\x00{key}"].logits[0])
                finalists[qid] = ranked[:STAGE_TWO_CANDIDATES]
                reduced[qid] = Choice(instructions=q.instructions, criteria={k: q.criteria[k] for k in finalists[qid]})
        scored = self._score_plain(state, reduced)
        tokens += scored.input_tokens
        raw = dict(scored.raw)
        for qid, keys in finalists.items():
            full = np.full(len(large[qid].criteria), -np.inf)
            for key, value in zip(keys, scored.raw[qid].logits):
                full[list(large[qid].criteria).index(key)] = value
            raw[qid] = RawScore(keys=option_keys(large[qid]), logits=full)
        return Scored(raw=raw, input_tokens=tokens)

    def _score_plain(self, state: State, questions: dict[str, Choice | Score | Noul]) -> Scored:
        prefix_text, tail = self.split_prompt(state)
        prefix_tokens = self._encode(prefix_text)
        passes = [
            _Pass(qid, rendered.keys, self._encode(rendered.text + tail))
            for qid, question in questions.items()
            for rendered in (render_question(question, self.codes.codes, r) for r in self._rotations_for(question))
        ]
        widest = max((p for p in passes), key=lambda p: len(p.tokens), default=None)
        if widest is not None and len(prefix_tokens) + len(widest.tokens) > self.max_branch_tokens:
            raise InputTooLong(f"question {widest.question_id!r} makes a branch of "
                               f"{len(prefix_tokens) + len(widest.tokens)} tokens, over the maximum context length "
                               f"of {self.max_branch_tokens} for one branch")
        total = len(prefix_tokens) + sum(len(p.tokens) for p in passes)
        if total > self.max_input_tokens:
            raise InputTooLong(f"request is {total} tokens, over the maximum context length of "
                               f"{self.max_input_tokens} for a whole request")

        per_question: dict[str, dict[str, list[float]]] = {qid: {} for qid in questions}
        with self._lock:
            prefix_cache = self._encode_prefix(prefix_tokens)
            for chunk in self._chunks(passes, len(prefix_tokens)):
                for one, logits in zip(chunk, self._forward_batch(prefix_cache, chunk)):
                    log_probs = log_softmax(logits)
                    if self.prior_debias:
                        log_probs = log_probs - self._prior(len(one.keys))
                    for key, value in zip(one.keys, log_probs):
                        per_question[one.question_id].setdefault(key, []).append(float(value))

        raw = {}
        for qid, question in questions.items():
            keys = option_keys(question)
            raw[qid] = RawScore(keys=keys, logits=np.array([np.mean(per_question[qid][k]) for k in keys]))
        return Scored(raw=raw, input_tokens=total)

    def _prior(self, cardinality: int) -> np.ndarray:
        """Log-probabilities the model assigns to each letter with no content to go on."""
        if cardinality not in self._priors:
            prefix_text, tail = self.split_prompt(PRIOR_STATE)
            rendered = render_question(prior_question(cardinality), self.codes.codes)
            cache = self._encode_prefix(self._encode(prefix_text))
            one = _Pass("prior", rendered.keys, self._encode(rendered.text + tail))
            self._priors[cardinality] = log_softmax(self._forward_batch(cache, [one])[0])
        return self._priors[cardinality]

    def _chunks(self, passes: list[_Pass], prefix_length: int) -> list[list[_Pass]]:
        chunks: list[list[_Pass]] = [[]]
        for one in passes:
            candidate = chunks[-1] + [one]
            width = max(len(p.tokens) for p in candidate)
            if chunks[-1] and len(candidate) * (prefix_length + width) > BATCH_TOKEN_BUDGET:
                chunks.append([one])
            else:
                chunks[-1] = candidate
        return chunks

    def _forward_batch(self, prefix_cache, passes: list[_Pass]) -> list[np.ndarray]:
        """One padded forward pass for every suffix, against replicas of the prefix cache."""
        batch = len(passes)
        width = max(len(p.tokens) for p in passes)
        cache = make_prompt_cache(self.model)
        for replica, original in zip(cache, prefix_cache):
            state = original.state
            if isinstance(state, tuple):
                replica.state = tuple(mx.concatenate([a] * batch, axis=0) for a in state)
            else:
                replica.state = [None if a is None else mx.concatenate([a] * batch, axis=0) for a in state]
        tokens = mx.array([p.tokens + [self._pad_id] * (width - len(p.tokens)) for p in passes])
        last = mx.array([len(p.tokens) - 1 for p in passes])
        cardinality = max(len(p.keys) for p in passes)
        ids = mx.array([self.codes.token_ids[:cardinality] for _ in passes])
        logits = np.array(option_logits(self.model, tokens, cache, last, ids), dtype=np.float64)
        return [row[: len(one.keys)] for row, one in zip(logits, passes)]

    # ---- answers -----------------------------------------------------------

    def evaluate(self, request: SystemOneRequest) -> SystemOneResponse:
        scored = self.score(request.state, request.questions)
        answers = {qid: self.answer(request.questions[qid], scored.raw[qid]) for qid in request.questions}
        return SystemOneResponse(
            model=self.model_id, answers=answers, usage=Usage(input_tokens=scored.input_tokens)
        )

    def answer(self, question: Choice | Score | Noul, raw: RawScore) -> ChoiceAnswer | ScoreAnswer | NoulAnswer:
        return answer(question, raw, self.calibration)

    # ---- helpers -----------------------------------------------------------

    def _rotations_for(self, question: Choice | Score | Noul) -> range:
        orderings = 2 if isinstance(question, Score) else len(option_keys(question))
        return range(min(self.rotations, orderings))

    def _encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def _encode_prefix(self, tokens: list[int]):
        cache = make_prompt_cache(self.model)
        array = mx.array(tokens)[None]
        for start in range(0, len(tokens), PREFILL_STEP):
            self.model(array[:, start : start + PREFILL_STEP], cache=cache)
            mx.eval([c.state for c in cache])
        return cache
