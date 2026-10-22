from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from elevenx.api import app

client = TestClient(app)


def test_health_does_not_load_models():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "ElevenX",
        "interface": "web",
    }


def test_homepage_serves_voice_studio():
    response = client.get("/")
    assert response.status_code == 200
    assert "ElevenX — Voice Studio" in response.text
    assert 'id="generate"' in response.text
    assert 'id="reference"' in response.text


def test_synthesis_rejects_unsupported_reference_extension():
    response = client.post(
        "/v1/synthesize",
        data={"text": "Hello world", "language": "en", "emotion": "neutral"},
        files={"reference_audio": ("reference.exe", b"not audio", "application/octet-stream")},
    )
    assert response.status_code == 415


@pytest.mark.parametrize("emotion", ["unknown", "sad"])
def test_synthesis_rejects_unknown_emotion_before_loading_model(emotion):
    response = client.post(
        "/v1/synthesize",
        data={"text": "Hello world", "language": "en", "emotion": emotion},
        files={"reference_audio": ("reference.wav", b"reference", "audio/wav")},
    )
    assert response.status_code == 422


def test_synthesis_rejects_blank_text_before_loading_model():
    response = client.post(
        "/v1/synthesize",
        data={"text": "   ", "language": "en", "emotion": "neutral"},
        files={"reference_audio": ("reference.wav", b"reference", "audio/wav")},
    )
    assert response.status_code == 422


def test_synthesis_uses_engine_and_returns_wav(monkeypatch, tmp_path):
    class FakeResult:
        pass

    class FakeEngine:
        def synthesize(self, text, reference_audio, language, emotion):
            assert text == "Hello from tests"
            assert language == "en"
            assert emotion == "neutral"
            assert Path(reference_audio).read_bytes() == b"valid-for-fake"
            return FakeResult()

        def save_wav(self, result, output_path):
            Path(output_path).write_bytes(b"RIFFfake-wav")
            return Path(output_path)

    monkeypatch.setattr("elevenx.api.get_engine", lambda: FakeEngine())
    response = client.post(
        "/v1/synthesize",
        data={"text": "Hello from tests", "language": "en", "emotion": "neutral"},
        files={"reference_audio": ("reference.wav", b"valid-for-fake", "audio/wav")},
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("audio/wav")
    assert response.content == b"RIFFfake-wav"


def test_synthesis_upload_limit_is_enforced():
    payload = b"x" * (25 * 1024 * 1024 + 1)
    response = client.post(
        "/v1/synthesize",
        data={"text": "Hello world", "language": "en", "emotion": "neutral"},
        files={"reference_audio": ("reference.wav", payload, "audio/wav")},
    )
    assert response.status_code == 413
