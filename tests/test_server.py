import numpy as np
import pytest

from elevenx.server import pcm16_bytes, resolve_reference_path, wav_bytes


def test_pcm16_clips_and_sanitizes():
    data = pcm16_bytes(np.array([-2.0, np.nan, 0.5, 2.0], dtype=np.float32))
    decoded = np.frombuffer(data, dtype="<i2")
    assert decoded.tolist() == [-32767, 0, 16383, 32767]


def test_wav_has_expected_metadata():
    import wave
    import io

    payload = wav_bytes(np.zeros(32, dtype=np.float32), 24000)
    with wave.open(io.BytesIO(payload), "rb") as wav:
        assert wav.getnchannels() == 1
        assert wav.getframerate() == 24000
        assert wav.getsampwidth() == 2
        assert wav.getnframes() == 32


def test_reference_path_rejects_traversal(tmp_path, monkeypatch):
    import elevenx.server as server
    monkeypatch.setattr(server, "REFERENCE_ROOT", tmp_path.resolve())
    with pytest.raises(ValueError):
        resolve_reference_path("../secret.wav")


def test_reference_path_requires_existing_audio(tmp_path, monkeypatch):
    import elevenx.server as server
    monkeypatch.setattr(server, "REFERENCE_ROOT", tmp_path.resolve())
    with pytest.raises(FileNotFoundError):
        resolve_reference_path("missing.wav")
