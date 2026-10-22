from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from elevenx import api


client = TestClient(api.app)


def test_health_reports_web_and_complete_waveform_mode():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["interface"] == "web"
    assert response.json()["mode"] == "complete-waveform"


def test_homepage_serves_voice_studio():
    response = client.get("/")
    assert response.status_code == 200
    assert "ElevenX — Voice Studio" in response.text
    assert 'id="generate"' in response.text
    assert 'id="reference"' in response.text


def test_overlapping_synthesis_is_rejected_with_retry_hint():
    assert api._synthesis_slot.acquire(blocking=False)
    try:
        response = client.post(
            "/v1/synthesize",
            data={"text": "hello", "language": "en", "emotion": "neutral", "consent": "true"},
            files={"reference_audio": ("reference.wav", b"small test payload", "audio/wav")},
        )
        assert response.status_code == 503
        assert response.headers["retry-after"] == "3"
        assert "already synthesizing" in response.json()["detail"]
    finally:
        api._synthesis_slot.release()


def test_unsupported_reference_format_is_rejected():
    response = client.post(
        "/v1/synthesize",
        data={"text": "hello", "consent": "true"},
        files={"reference_audio": ("reference.exe", b"not audio", "application/octet-stream")},
    )
    assert response.status_code == 415


@pytest.mark.parametrize("emotion", ["unknown", "sad"])
def test_unknown_emotion_is_rejected_before_engine_load(monkeypatch, emotion):
    monkeypatch.setattr(api, "get_engine", lambda: pytest.fail("engine should not load"))
    response = client.post(
        "/v1/synthesize",
        data={"text": "hello", "language": "en", "emotion": emotion, "consent": "true"},
        files={"reference_audio": ("reference.wav", b"reference", "audio/wav")},
    )
    assert response.status_code == 422


def test_blank_text_is_rejected_before_engine_load(monkeypatch):
    monkeypatch.setattr(api, "get_engine", lambda: pytest.fail("engine should not load"))
    response = client.post(
        "/v1/synthesize",
        data={"text": "   ", "language": "en", "emotion": "neutral", "consent": "true"},
        files={"reference_audio": ("reference.wav", b"reference", "audio/wav")},
    )
    assert response.status_code == 422


def test_voice_permission_is_required():
    response = client.post(
        "/v1/synthesize",
        data={"text": "hello", "language": "en", "emotion": "neutral", "consent": "false"},
        files={"reference_audio": ("reference.wav", b"reference", "audio/wav")},
    )
    assert response.status_code == 403
    assert "permission" in response.json()["detail"].lower()


def test_empty_reference_is_rejected_without_loading_engine(monkeypatch):
    monkeypatch.setattr(api, "get_engine", lambda: pytest.fail("engine should not load"))
    response = client.post(
        "/v1/synthesize",
        data={"text": "hello", "consent": "true"},
        files={"reference_audio": ("reference.wav", b"", "audio/wav")},
    )
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


def test_reference_upload_limit_is_enforced(monkeypatch):
    monkeypatch.setattr(api, "get_engine", lambda: pytest.fail("engine should not load"))
    oversized = b"x" * (25 * 1024 * 1024 + 1)
    response = client.post(
        "/v1/synthesize",
        data={"text": "hello", "consent": "true"},
        files={"reference_audio": ("reference.wav", oversized, "audio/wav")},
    )
    assert response.status_code == 413


def test_synthesis_returns_wav_and_cleans_temporary_files(monkeypatch):
    captured = {}

    class FakeEngine:
        def synthesize(self, text, reference_audio, language, emotion):
            assert text == "Hello from tests"
            assert language == "en"
            assert emotion == "neutral"
            captured["reference_path"] = Path(reference_audio)
            assert captured["reference_path"].read_bytes() == b"fake reference"
            return object()

        def save_wav(self, result, output_path):
            captured["output_path"] = Path(output_path)
            captured["output_path"].write_bytes(b"RIFF-test-wav")
            return captured["output_path"]

    monkeypatch.setattr(api, "get_engine", lambda: FakeEngine())
    response = client.post(
        "/v1/synthesize",
        data={"text": "Hello from tests", "language": "en", "emotion": "neutral", "consent": "true"},
        files={"reference_audio": ("reference.wav", b"fake reference", "audio/wav")},
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("audio/wav")
    assert response.content == b"RIFF-test-wav"
    assert not captured["reference_path"].exists()
    assert not captured["output_path"].exists()
    assert not api._synthesis_slot.locked()


def test_synthesis_releases_slot_when_engine_raises(monkeypatch):
    class FailingEngine:
        def synthesize(self, *args, **kwargs):
            raise RuntimeError("synthetic model failure")

    monkeypatch.setattr(api, "get_engine", lambda: FailingEngine())
    response = client.post(
        "/v1/synthesize",
        data={"text": "Hello from tests", "language": "en", "emotion": "neutral", "consent": "true"},
        files={"reference_audio": ("reference.wav", b"fake reference", "audio/wav")},
    )

    assert response.status_code == 400
    assert "synthetic model failure" in response.json()["detail"]
    assert not api._synthesis_slot.locked()
