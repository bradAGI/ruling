"""Distillation data: a large open model writes and answers typed questions about many states.

The served model is the fast decision encoder, which generalizes poorly when
trained only on a few labeled sources. Here a large open-weight decoder writes
varied Choice, Score, and Noul questions about each state, then answers them
through the decoder engine with option-order averaging. Its probabilities
become soft targets, so the encoder learns the larger model's judgment,
uncertainty included, over far more kinds of question than any labeled
dataset holds.
"""

from __future__ import annotations

import json
import re

import mlx.core as mx
from mlx_lm import batch_generate
from mlx_lm.sample_utils import make_sampler
from pydantic import TypeAdapter, ValidationError

from ruling.calibration import softmax
from ruling.engine import TEMPLATE_KWARGS, Engine, InputTooLong
from ruling.evals import Record
from ruling.questions import Choice, Question, State

WRITER_SYSTEM = ("You design evaluation questions for software that makes fast judgments about documents. "
                 "Reply with one JSON object and nothing else.")
QUESTION = TypeAdapter(Question)


def writer_messages(state: State, count: int) -> list[dict[str, str]]:
    body = state if isinstance(state, str) else json.dumps(state, indent=2, ensure_ascii=False)
    instructions = f"""Document:
{body}

Write {count} different questions a program might need answered about this document. Mix three types:
- "noul": a statement that is true or false about the document.
- "choice": pick one of 2 to 8 named options; give each option a short description.
- "score": an ordered scale of 3 to 5 levels from lowest to highest.
Cover different aspects: intent, tone, facts stated, facts not stated, risk, next action, category. Include some questions
whose answer is genuinely uncertain from the document, and some whose answer is clearly no. Use snake_case ids.

Reply with exactly this JSON shape:
{{"questions": {{
  "some_id": {{"type": "noul", "instructions": "..."}},
  "other_id": {{"type": "choice", "instructions": "...", "criteria": {{"option_a": "description", "option_b": "description"}}}},
  "third_id": {{"type": "score", "instructions": "...", "criteria": ["lowest level", "middle level", "highest level"]}}
}}}}"""
    return [{"role": "system", "content": WRITER_SYSTEM}, {"role": "user", "content": instructions}]


def parse_questions(text: str, max_options: int) -> dict:
    """The valid questions in a writer reply; anything malformed is dropped rather than repaired."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return {}
    try:
        payload = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return {}
    candidates = payload.get("questions") if isinstance(payload, dict) else None
    if not isinstance(candidates, dict):
        return {}
    questions = {}
    for qid, spec in candidates.items():
        try:
            question = QUESTION.validate_python(spec)
        except ValidationError:
            continue
        if isinstance(question, Choice) and len(question.criteria) > max_options:
            continue
        questions[str(qid)] = question
    return questions


def write_questions(states: list[tuple[str, State]], generate, questions_per_state: int, batch_size: int,
                    max_options: int, log=print):
    """Yield, batch by batch, records carrying the questions a writer model produced for each state.

    `generate` takes a list of chat message lists and returns one reply text per prompt.
    """
    kept = 0
    for start in range(0, len(states), batch_size):
        batch = states[start:start + batch_size]
        texts = generate([writer_messages(state, questions_per_state) for _, state in batch])
        records = [Record(family=f"distill_{source}", state=state, questions=questions)
                   for (source, state), text in zip(batch, texts)
                   if (questions := parse_questions(text, max_options))]
        kept += len(records)
        log(f"{min(start + batch_size, len(states))}/{len(states)} states, {kept} with questions")
        yield records


def local_generator(engine: Engine, seed: int, temperature: float):
    """A `generate` function backed by a local MLX model with batched sampling."""
    mx.random.seed(seed)
    sampler = make_sampler(temp=temperature, top_p=0.95)

    def generate(prompts: list[list[dict]]) -> list[str]:
        tokens = [engine.tokenizer.apply_chat_template(messages, add_generation_prompt=True, **TEMPLATE_KWARGS) for messages in prompts]
        return batch_generate(engine.model, engine.tokenizer, tokens, max_tokens=1200, sampler=sampler).texts

    return generate


def answer_locally(records: list[Record], engine, log=print):
    """Yield each record with the engine's answer distributions as `target` references; records too long are skipped."""
    for index, record in enumerate(records):
        try:
            scored = engine.score(record.state, record.questions)
        except InputTooLong:
            continue
        targets = {qid: dict(zip(raw.keys, (float(p) for p in softmax(raw.logits)))) for qid, raw in scored.raw.items()}
        if (index + 1) % 100 == 0:
            log(f"{index + 1}/{len(records)} answered")
        yield record.model_copy(update={"references": {qid: {"target": t} for qid, t in targets.items()}})
