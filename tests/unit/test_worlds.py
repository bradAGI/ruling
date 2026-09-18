from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from ruling.evals import Record
from ruling.questions import option_keys
from ruling.worlds import Catalog, World, clean_output, record_for, sample_latent, writer_messages

CATALOG = Path(__file__).resolve().parents[2] / "datasets" / "worlds.json"


def test_bundled_catalog_is_consistent():
    catalog = Catalog.load(CATALOG)
    assert len(catalog.worlds) >= 8 and catalog.styles
    for world in catalog.worlds:
        for attribute in world.attributes:
            assert set(attribute.values) == {("true" if k == "yes" else "false" if k == "no" else k)
                                             for k in option_keys(attribute.question)}


def test_world_rejects_values_that_do_not_match_the_question():
    bad = {"id": "w", "writer": "a note", "attributes": [
        {"id": "a", "question": {"type": "choice", "instructions": "q", "criteria": {"x": None, "y": None}},
         "values": {"x": {"hint": "hx", "weight": 1}, "z": {"hint": "hz", "weight": 1}}}]}
    with pytest.raises(ValidationError, match="values must match"):
        World.model_validate(bad)


def test_sampling_follows_weights_and_is_deterministic():
    world = Catalog.load(CATALOG).worlds[0]
    a = [sample_latent(world, np.random.default_rng(i)) for i in range(400)]
    b = [sample_latent(world, np.random.default_rng(i)) for i in range(400)]
    assert a == b
    refunds = sum(1 for latent in a if latent["goal"] == "refund") / len(a)
    assert 0.18 < refunds < 0.37  # weight 3 of 11


def test_prompt_carries_every_hint_and_record_carries_every_label():
    catalog = Catalog.load(CATALOG)
    world = catalog.worlds[0]
    latent = sample_latent(world, np.random.default_rng(1))
    messages = writer_messages(world, latent, "formal and polite")
    body = messages[-1]["content"]
    for attribute in world.attributes:
        assert attribute.values[latent[attribute.id]].hint in body
    record = record_for(world, latent, "Hello, my order 123 never came.")
    assert isinstance(record, Record) and record.family == f"world_{world.id}"
    assert set(record.labels) == {a.id for a in world.attributes}
    frustration = record.labels["frustration"]
    assert frustration == int(latent["frustration"]) and isinstance(record.labels["deadline"], bool)


def test_clean_output_strips_thinking_quotes_and_labels():
    assert clean_output("<think>\nplan\n</think>\n\n\"Hi there, my laptop died.\"") == "Hi there, my laptop died."
    assert clean_output("Message:\nHi there") == "Hi there"
    assert clean_output("   ") == ""
