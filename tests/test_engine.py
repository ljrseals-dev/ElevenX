import sys
import types

import pytest

from elevenx.engine import ElevenXEngine


def _write_config(tmp_path, sample_rate=24000):
    config = tmp_path / "config.yaml"
    config.write_text(
        f"audio:\n  sample_rate: {sample_rate}\nmodels:\n  device: cpu\n",
        encoding="utf-8",
    )
    return config


def _install_fake_tts(monkeypatch, model_factory):
    fake_tts = types.ModuleType("TTS")
    fake_api = types.ModuleType("TTS.api")
    fake_api.TTS = model_factory
    fake_tts.api = fake_api
    monkeypatch.setitem(sys.modules, "TTS", fake_tts)
    monkeypatch.setitem(sys.modules, "TTS.api", fake_api)


def test_engine_rejects_non_positive_configured_sample_rate(tmp_path):
    config = _write_config(tmp_path, sample_rate=0)
    with pytest.raises(ValueError, match="positive integer"):
        ElevenXEngine(config)


def test_engine_does_not_load_xtts_during_initialization(tmp_path):
    engine = ElevenXEngine(_write_config(tmp_path))
    assert engine._tts is None


def test_engine_prefers_model_declared_output_sample_rate(tmp_path, monkeypatch):
    class FakeModel:
        def __init__(self, **kwargs):
            self.synthesizer = types.SimpleNamespace(output_sample_rate=44100)

        def to(self, device):
            return self

    _install_fake_tts(monkeypatch, FakeModel)
    engine = ElevenXEngine(_write_config(tmp_path))
    engine._load_tts()
    assert engine.sample_rate == 44100


def test_engine_uses_tts_config_audio_sample_rate_fallback(tmp_path, monkeypatch):
    class FakeModel:
        def __init__(self, **kwargs):
            self.synthesizer = types.SimpleNamespace(
                output_sample_rate=None,
                tts_config=types.SimpleNamespace(audio={"sample_rate": 32000}),
            )

        def to(self, device):
            return self

    _install_fake_tts(monkeypatch, FakeModel)
    engine = ElevenXEngine(_write_config(tmp_path))
    engine._load_tts()
    assert engine.sample_rate == 32000


def test_engine_ignores_invalid_model_sample_rate_and_keeps_configured_rate(
    tmp_path, monkeypatch
):
    class FakeModel:
        def __init__(self, **kwargs):
            self.synthesizer = types.SimpleNamespace(
                output_sample_rate=0,
                tts_config=types.SimpleNamespace(audio={"sample_rate": -1}),
            )

        def to(self, device):
            return self

    _install_fake_tts(monkeypatch, FakeModel)
    engine = ElevenXEngine(_write_config(tmp_path, sample_rate=22050))
    engine._load_tts()
    assert engine.sample_rate == 22050
