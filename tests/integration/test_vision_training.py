"""Training reads the images in a record: an adapter on a `vision:` model learns what only the picture shows."""

import base64
import io
from dataclasses import replace

import numpy as np
import pytest
from PIL import Image

from ruling.build import write
from ruling.config import Settings
from ruling.evals import Record
from ruling.questions import SystemOneRequest
from ruling.registry import build_engine
from ruling.train import Trainer, TrainConfig
from ruling.vision import VISION_PREFIX

from .conftest import TEST_MODEL

pytestmark = [pytest.mark.integration, pytest.mark.heavy]

# A code no model knows: only training can tie it to the color in the picture.
CODES = {"red": "alpha", "blue": "beta", "green": "gamma"}
QUESTION = {"code": {"type": "choice", "instructions": "Which code goes with <image 1>?",
                     "criteria": {code: None for code in CODES.values()}}}
RGB = {"red": (200, 30, 30), "blue": (30, 40, 200), "green": (30, 170, 50)}


def records(count: int, seed: int) -> list[Record]:
    rng = np.random.default_rng(seed)
    made = []
    for i in range(count):
        color = list(CODES)[i % 3]
        pixels = np.clip(np.array(RGB[color]) + rng.integers(-25, 26, (48, 48, 3)), 0, 255).astype(np.uint8)
        buffer = io.BytesIO()
        Image.fromarray(pixels).save(buffer, format="PNG")
        photo = {"type": "image", "media_type": "image/png", "data": base64.b64encode(buffer.getvalue()).decode()}
        made.append(Record(state={"photo": photo}, questions=QUESTION, labels={"code": CODES[color]}))
    return made


def notes(count: int) -> list[Record]:
    """Text-only records: a vision model's training data may mix them with images."""
    return [Record(state=f"Ticket {i}: the {color} lamp in room {i} is broken.",
                   questions={"lamp": {"type": "noul", "instructions": "The ticket is about a lamp."}},
                   labels={"lamp": True}) for i, color in zip(range(count), ["red", "blue", "green"] * count)]


def log_likelihood(engine, held_out: list[Record]) -> float:
    """Mean log-probability the served model gives each held-out record's right code."""
    answers = [engine.evaluate(SystemOneRequest(state=r.state, questions=r.questions)).answers["code"]
               for r in held_out]
    return float(np.mean([np.log(a.probabilities[r.labels["code"]]) for a, r in zip(answers, held_out)]))


def test_a_vision_adapter_learns_from_the_images_and_serves_what_it_learned(tmp_path):
    train, holdout, adapter = tmp_path / "train.jsonl", tmp_path / "holdout.jsonl", tmp_path / "adapter"
    write([r.model_dump(mode="json") for r in records(48, seed=1) + notes(12)], train)
    held_out = records(9, seed=2)
    write([r.model_dump(mode="json") for r in held_out], holdout)
    # Measured: 300 steps take the validation loss from 2.3 to 0.4; far fewer leave it at the uniform 1.1.
    config = TrainConfig(model=VISION_PREFIX + TEST_MODEL, data=[train], output=adapter, holdout=[holdout],
                         steps=300, batch_size=4, warmup=5, learning_rate=3e-5, lora_layers=16, max_tokens=512,
                         eval_every=100, valid_fraction=0.25, max_image_tokens=64, seed=0)
    history = Trainer(config).run(log=lambda *_: None)
    assert history[-1]["loss"] < history[0]["loss"] / 2

    settings = Settings(model=VISION_PREFIX + TEST_MODEL, rotations=1, max_image_tokens=64)
    base, tuned = build_engine(settings), build_engine(replace(settings, adapter_path=adapter))
    # Only the picture says which code is right, so a gain on held-out pictures is a gain from reading them.
    assert log_likelihood(tuned, held_out) > log_likelihood(base, held_out) + 0.5


def test_a_text_model_refuses_to_train_on_records_with_images(tmp_path):
    train = tmp_path / "train.jsonl"
    write([r.model_dump(mode="json") for r in records(6, seed=3)], train)
    with pytest.raises(ValueError, match="images"):
        Trainer(TrainConfig(model=TEST_MODEL, data=[train], output=tmp_path / "out", holdout=[], steps=1))
