"""Model-free tests for the vendored drop-in TTS engine (baseline + Harmonia slot)."""
import wave
import numpy as np
from elevenx.tts.engine import TTSEngine, available_codecs
from elevenx.tts.cli import main


def test_registry_lists_baseline():
    assert "baseline" in available_codecs()


def test_baseline_synth_non_silent():
    a = TTSEngine(sample_rate=22050).synth("hello world")
    assert a.dtype == np.float32
    assert a.size > 22050 * 0.3
    assert float(np.max(np.abs(a))) > 0.05


def test_harmonia_falls_back():
    # Harmonia slot is registered but not installed; engine must fall back to baseline
    assert TTSEngine("harmonia").codec.config.name == "baseline"


def test_harmonia_encode_fallback_regenerates():
    e = TTSEngine()
    e.swap_codec("baseline")
    a = e.synth("hello")
    assert len(a) > 0


def test_cli_wav_roundtrip(tmp_path):
    p = tmp_path / "out.wav"
    assert main(["hello world", str(p)]) == 0
    with wave.open(str(p), "rb") as w:
        assert (w.getnchannels(), w.getsampwidth(), w.getframerate()) == (1, 2, 22050)
        assert w.getnframes() > 0


def test_non_ascii_is_safe():
    assert TTSEngine().synth("héllo — world").size > 0
