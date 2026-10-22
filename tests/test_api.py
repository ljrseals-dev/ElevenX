from fastapi.testclient import TestClient

from elevenx import api


client = TestClient(api.app)


def test_health_reports_complete_waveform_mode():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["mode"] == "complete-waveform"


def test_overlapping_synthesis_is_rejected_with_retry_hint():
    assert api._synthesis_slot.acquire(blocking=False)
    try:
        response = client.post(
            "/v1/synthesize",
            data={"text": "hello", "language": "en", "emotion": "neutral"},
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
        data={"text": "hello"},
        files={"reference_audio": ("reference.exe", b"not audio", "application/octet-stream")},
    )
    assert response.status_code == 415
