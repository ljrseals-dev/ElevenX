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
