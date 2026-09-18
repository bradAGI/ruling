"""A teacher that reasons before it answers.

For each question the model writes its reasoning, the reasoning is closed
after `</think>`, and the answer is then read from the restricted letter
logits exactly as the fast engine reads it. The result is a calibrated
distribution informed by slow reasoning: the System Two answers a System One
model is trained to reproduce. It is far too slow to serve.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
from mlx_lm import batch_generate
from mlx_lm.sample_utils import make_sampler

from ruling.calibration import Provenance, log_softmax
from ruling.engine import Engine, InputTooLong, RawScore, Scored, answer, option_logits
from ruling.prompt import render_question
from ruling.questions import Choice, Noul, Score, State, SystemOneRequest, SystemOneResponse, Usage, option_keys

THINK_END = "\n</think>\n\n"


class ThinkingEngine:
    def __init__(self, engine: Engine, max_think_tokens: int):
        self.engine = engine
        self.max_think_tokens = max_think_tokens
        self.model_id = f"{engine.model_id} thinking {max_think_tokens}"
        self.revision = engine.revision
        self.adapter = engine.adapter
        self.rotations = 1
        self.prior_debias = False
        self.max_options = engine.codes.max_options
        self.max_input_tokens = engine.max_input_tokens
        self.calibration = engine.calibration

    @property
    def provenance(self) -> Provenance:
        return Provenance(self.model_id, self.revision, self.rotations, self.prior_debias, self.adapter)

    def score(self, state: State, questions: dict[str, Choice | Score | Noul]) -> Scored:
        engine = self.engine
        prefix, tail = engine.split_prompt(state, thinking=True)
        prefix_tokens = engine._encode(prefix)
        rendered = {qid: render_question(q, engine.codes.codes, 0) for qid, q in questions.items()}
        if any(len(r.keys) > engine.codes.max_options for r in rendered.values()):
            raise InputTooLong("a question has more options than the teacher's answer codes")
        prompts = [prefix_tokens + engine._encode(r.text + tail) for r in rendered.values()]
        if max(len(p) for p in prompts) + self.max_think_tokens > self.max_input_tokens:
            raise InputTooLong(f"request needs more than {self.max_input_tokens} tokens with reasoning")
        texts = batch_generate(engine.model, engine.tokenizer, prompts, max_tokens=self.max_think_tokens,
                               sampler=make_sampler(temp=0.0)).texts
        sequences = [p + engine._encode(t.split("</think>")[0].rstrip() + THINK_END) for p, t in zip(prompts, texts)]
        width = max(len(s) for s in sequences)
        tokens = mx.array([s + [engine._pad_id] * (width - len(s)) for s in sequences])
        last = mx.array([len(s) - 1 for s in sequences])
        cardinality = max(len(r.keys) for r in rendered.values())
        ids = mx.array([engine.codes.token_ids[:cardinality] for _ in sequences])
        logits = np.array(option_logits(engine.model, tokens, None, last, ids), dtype=np.float64)
        raw = {}
        for row, (qid, question) in zip(logits, questions.items()):
            keys = rendered[qid].keys
            log_probs = dict(zip(keys, log_softmax(row[: len(keys)])))
            raw[qid] = RawScore(keys=option_keys(question), logits=np.array([log_probs[k] for k in option_keys(question)]))
        return Scored(raw=raw, input_tokens=sum(len(s) for s in sequences))

    def evaluate(self, request: SystemOneRequest) -> SystemOneResponse:
        scored = self.score(request.state, request.questions)
        answers = {qid: answer(request.questions[qid], scored.raw[qid], self.calibration) for qid in request.questions}
        return SystemOneResponse(model=self.model_id, answers=answers, usage=Usage(input_tokens=scored.input_tokens))

    def answer(self, question, raw: RawScore):
        return answer(question, raw, self.calibration)
