import sys
import types

import pytest

from elevenx.engine import ElevenXEngine


def test_engine_rejects_non_positive_configured_sample_rate(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text("audio:\n  sample_rate: 0\nmodels:\n  device: cpu\n", encoding="utf-8")
    with pytest.raises(ValueError, match="positive integer"):
        ElevenXEngine(config)


def test_engine_prefers_model_declared_output_sample_rate(tmp_path, monkeypatch):
    config = tmp_path / "config.yaml"
    config.write_text("audio:\n  sample_rate: 24000\nmodels:\n  device: cpu\n", encoding="utf-8")

    class FakeModel:
        def __init__(self, **kwargs):
            self.synthesizer = types.SimpleNamespace(output_sample_rate=44100)

        def to(self, device):
            return self

    fake_tts = types.ModuleType("TTS")
    fake_api = types.ModuleType("TTS.api")
    fake_api.TTS = FakeModel
    fake_tts.api = fake_api
    monkeypatch.setitem(sys.modules, "TTS", fake_tts)
    monkeypatch.setitem(sys.modules, "TTS.api", fake_api)

    engine = ElevenXEngine(config)
    engine._load_tts()

    assert engine.sample_rate == 44100
