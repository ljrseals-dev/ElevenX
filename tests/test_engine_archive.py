"""Additional model-free regression tests for lazy XTTS initialization."""
from __future__ import annotations

import sys
import types

from elevenx.engine import ElevenXEngine


def test_xtts_is_not_loaded_during_engine_construction(tmp_path, monkeypatch):
    config = tmp_path / "config.yaml"
    config.write_text(
        "audio:\n  sample_rate: 24000\nmodels:\n  device: cpu\n",
        encoding="utf-8",
    )
    engine = ElevenXEngine(config)
    assert engine._tts is None
    assert engine.sample_rate == 24000


def test_model_declared_output_rate_overrides_configured_rate(tmp_path, monkeypatch):
    config = tmp_path / "config.yaml"
    config.write_text(
        "audio:\n  sample_rate: 24000\nmodels:\n  device: cpu\n",
        encoding="utf-8",
    )

    class FakeModel:
        def __init__(self, **kwargs):
            self.synthesizer = types.SimpleNamespace(output_sample_rate=44100)

        def to(self, device):
            assert device == "cpu"
            return self

    fake_package = types.ModuleType("TTS")
    fake_api = types.ModuleType("TTS.api")
    fake_api.TTS = FakeModel
    fake_package.api = fake_api
    monkeypatch.setitem(sys.modules, "TTS", fake_package)
    monkeypatch.setitem(sys.modules, "TTS.api", fake_api)

    engine = ElevenXEngine(config)
    assert engine._tts is None

    model = engine._load_tts()

    assert model is engine._tts
    assert engine.sample_rate == 44100
    assert engine._load_tts() is model
