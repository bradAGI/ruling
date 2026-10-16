"""A registry names every model a server answers for, and shares a base between its adapters."""

import json
from pathlib import Path

import pytest

from ruling.calibration import Calibration
from ruling.config import Settings
from ruling.questions import DEFAULT_MODEL_ALIASES
from ruling import registry as registry_module
from ruling.registry import Entry, Registry, UnknownModel, load_entries


class FakeEngine:
    prior_debias = False
    max_options = 255
    reads_images = False
    rotations = 1
    max_input_tokens = max_branch_tokens = 1000
    revision = adapter = None

    def __init__(self, model_id):
        self.model_id = model_id
        self.calibration = Calibration()


def test_resolve_maps_none_and_aliases_to_the_default_and_rejects_strangers():
    registry = Registry("base", {"base": FakeEngine("base"), "tuned": FakeEngine("base")})
    assert registry.resolve(None)[0] == "base"
    for alias in DEFAULT_MODEL_ALIASES:
        assert registry.resolve(alias)[0] == "base"
    assert registry.resolve("tuned")[0] == "tuned"
    with pytest.raises(UnknownModel) as caught:
        registry.resolve("nope")
    assert caught.value.known == ["base", "tuned"] and "'tuned'" in str(caught.value)


def test_default_must_be_an_entry():
    with pytest.raises(ValueError, match="default"):
        Registry("missing", {"base": FakeEngine("base")})


def test_entries_resolve_paths_against_the_file_and_refuse_unknown_fields_and_alias_names(tmp_path):
    models = tmp_path / "models.json"
    models.write_text(json.dumps({
        "routing/v1": {"model": "some/base", "adapter": "adapters/routing", "calibration": "adapters/routing/calibration.json"},
        "big": {"model": "some/big"},
    }))
    entries = load_entries(models)
    assert entries["routing/v1"] == Entry("some/base", tmp_path / "adapters/routing", tmp_path / "adapters/routing/calibration.json")
    assert entries["big"] == Entry("some/big")
    for bad in ({"x": {"adapter": "a"}}, {"x": {"model": "m", "rotations": 3}}, {"default": {"model": "m"}}, {"": {"model": "m"}}, ["m"],
                # A local model has no address to override; ignoring one would hide a typo'd prefix.
                {"x": {"model": "some/base", "base_url": "http://127.0.0.1:8092"}}):
        models.write_text(json.dumps(bad))
        with pytest.raises(ValueError):
            load_entries(models)


def test_from_settings_shares_a_loaded_base_and_builds_the_rest(tmp_path, monkeypatch):
    """Entries on the primary's base become variants of it; others are built on their own."""
    built = []

    class SharedEngine(FakeEngine):
        def variant(self, adapter, calibration):
            other = SharedEngine(self.model_id)
            other.adapter = str(adapter)
            other.calibration = calibration
            return other

    def fake_build(settings):
        built.append(settings)
        return SharedEngine(settings.model)

    monkeypatch.setattr(registry_module, "build_engine", fake_build)
    monkeypatch.setattr(registry_module, "Engine", SharedEngine)
    (tmp_path / "models.json").write_text(json.dumps({
        "tuned": {"model": "the/base", "adapter": "adapter-a"},
        "other": {"model": "an/other"},
    }))
    settings = Settings(model="the/base", models_path=tmp_path / "models.json", cascade_to="x", cascade_threshold=0.5)
    registry = Registry.from_settings(settings)
    assert registry.default == "the/base" and sorted(registry.engines) == ["other", "the/base", "tuned"]
    assert registry.engines["tuned"].adapter == str(tmp_path / "adapter-a")
    assert [s.model for s in built] == ["the/base", "an/other"]
    # An entry is a plain engine: the primary's cascade and failover do not apply to it.
    assert built[1].cascade_to is None and built[1].adapter_path is None


def test_from_settings_refuses_an_entry_named_like_the_primary(tmp_path, monkeypatch):
    monkeypatch.setattr(registry_module, "build_engine", lambda settings: FakeEngine(settings.model))
    (tmp_path / "models.json").write_text(json.dumps({"the/base": {"model": "the/base"}}))
    with pytest.raises(ValueError, match="primary"):
        Registry.from_settings(Settings(model="the/base", models_path=tmp_path / "models.json"))


def test_settings_read_the_registry_file_and_queue_limit(monkeypatch):
    monkeypatch.setenv("RULING_MODELS", "/tmp/models.json")
    monkeypatch.setenv("RULING_MAX_PENDING", "3")
    settings = Settings.from_env()
    assert settings.models_path == Path("/tmp/models.json") and settings.max_pending == 3


def test_settings_read_the_image_token_cap(monkeypatch):
    monkeypatch.setenv("RULING_MAX_IMAGE_TOKENS", "512")
    assert Settings.from_env().max_image_tokens == 512
