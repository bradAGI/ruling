"""World synthesis: sample hidden facts first, then have a local model write text that conveys them.

A world is a kind of document (a support message, a bug report) plus hidden
attributes, each tied to a typed question. For every record the attribute
values are sampled from their weights, a writer model is told the facts as
plain hints, and the questions are asked about the finished text. The labels
are the sampled values, so they are known before the text exists. A judge pass
drops records where the writer plainly failed to convey a fact.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from mlx_lm import batch_generate
from mlx_lm.sample_utils import make_sampler
from pydantic import BaseModel, Field, model_validator

from ruling.engine import TEMPLATE_KWARGS, Engine
from ruling.evals import Record, label_index
from ruling.questions import Question, Score, option_keys

WRITER_SYSTEM = ("You write realistic sample documents for testing software. Output only the document itself, "
                 "with no title, preamble, labels, or commentary.")


class Value(BaseModel):
    hint: str
    weight: float = Field(gt=0)


class Attribute(BaseModel):
    id: str
    question: Question
    values: dict[str, Value]

    @model_validator(mode="after")
    def values_match_question(self) -> "Attribute":
        expected = {{"yes": "true", "no": "false"}.get(k, k) for k in option_keys(self.question)}
        if set(self.values) != expected:
            raise ValueError(f"attribute {self.id!r}: values must match the question's answers {sorted(expected)}")
        return self

    def label(self, value: str) -> str | int | bool:
        if self.question.type == "noul":
            return value == "true"
        if isinstance(self.question, Score):
            return int(value)
        return value


class World(BaseModel):
    id: str
    writer: str
    attributes: list[Attribute] = Field(min_length=1)


class Catalog(BaseModel):
    styles: list[str] = Field(min_length=1)
    worlds: list[World] = Field(min_length=1)

    @classmethod
    def load(cls, path: Path) -> "Catalog":
        return cls.model_validate_json(path.read_text())


def sample_latent(world: World, rng: np.random.Generator) -> dict[str, str]:
    latent = {}
    for attribute in world.attributes:
        keys = list(attribute.values)
        weights = np.array([attribute.values[k].weight for k in keys])
        latent[attribute.id] = keys[int(rng.choice(len(keys), p=weights / weights.sum()))]
    return latent


def writer_messages(world: World, latent: dict[str, str], style: str) -> list[dict[str, str]]:
    facts = "\n".join(f"- {attribute.values[latent[attribute.id]].hint}" for attribute in world.attributes)
    body = (f"Write {world.writer}.\n\nIt must make all of these true, conveyed naturally rather than stated as labels:\n"
            f"{facts}\n\nStyle: {style}. Invent any names, products, and details you need. Between one and eight sentences.")
    return [{"role": "system", "content": WRITER_SYSTEM}, {"role": "user", "content": body}]


def record_for(world: World, latent: dict[str, str], text: str) -> Record:
    return Record(
        family=f"world_{world.id}",
        state=text,
        questions={a.id: a.question for a in world.attributes},
        labels={a.id: a.label(latent[a.id]) for a in world.attributes},
    )


def clean_output(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    text = re.sub(r"^(message|document|email|ticket|review|post|notes?)\s*:\s*\n", "", text, flags=re.I).strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        text = text[1:-1].strip()
    return text


def synthesize(catalog: Catalog, engine: Engine, count: int, seed: int, batch_size: int,
               min_probability: float, temperature: float, log=print):
    """Write `count` candidate records with the engine's model and yield, batch by batch, those its judge pass accepts."""
    rng = np.random.default_rng(seed)
    sampler = make_sampler(temp=temperature, top_p=0.95)
    kept = 0
    for start in range(0, count, batch_size):
        jobs = []
        for _ in range(min(batch_size, count - start)):
            world = catalog.worlds[int(rng.integers(len(catalog.worlds)))]
            jobs.append((world, sample_latent(world, rng), catalog.styles[int(rng.integers(len(catalog.styles)))]))
        prompts = [engine.tokenizer.apply_chat_template(writer_messages(w, latent, style), add_generation_prompt=True,
                                                        **TEMPLATE_KWARGS) for w, latent, style in jobs]
        response = batch_generate(engine.model, engine.tokenizer, prompts, max_tokens=320, sampler=sampler)
        accepted: list[Record] = []
        for (world, latent, _), text in zip(jobs, response.texts):
            text = clean_output(text)
            if not text:
                continue
            record = record_for(world, latent, text)
            scored = engine.score(record.state, record.questions)
            probabilities = [_probability(scored.raw[qid].logits, label_index(record.questions[qid], label))
                             for qid, label in record.labels.items()]
            if min(probabilities) >= min_probability:
                accepted.append(record)
        kept += len(accepted)
        log(f"{min(start + batch_size, count)}/{count} written, {kept} kept")
        yield accepted


def _probability(logits: np.ndarray, index: int) -> float:
    shifted = logits - logits.max()
    return float(np.exp(shifted[index]) / np.exp(shifted).sum())
