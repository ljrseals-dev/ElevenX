"""Regression tests for API error-path cleanup and bounded input validation."""
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from elevenx import api


client = TestClient(api.app)


def _track_tempfiles(monkeypatch):
    created = []
    original = api.tempfile.NamedTemporaryFile

    def tracked_named_temporary_file(*args, **kwargs):
        handle = original(*args, **kwargs)
        created.append(Path(handle.name))
        return handle

    monkeypatch.setattr(api.tempfile, "NamedTemporaryFile", tracked_named_temporary_file)
    return created


@pytest.mark.parametrize("failure_stage", ["synthesize", "save_wav"])
def test_engine_failures_remove_all_temp_files_and_release_slot(monkeypatch, failure_stage):
    created = _track_tempfiles(monkeypatch)

    class FailingEngine:
        def synthesize(self, *args, **kwargs):
            if failure_stage == "synthesize":
                raise RuntimeError("injected synthesis failure")
            return object()

        def save_wav(self, result, output_path):
            if failure_stage == "save_wav":
                Path(output_path).write_bytes(b"partial output")
                raise RuntimeError("injected WAV write failure")

    monkeypatch.setattr(api, "get_engine", lambda: FailingEngine())
    response = client.post(
        "/v1/synthesize",
        data={"text": "cleanup regression", "language": "en", "emotion": "neutral", "consent": "true"},
        files={"reference_audio": ("reference.wav", b"test reference bytes", "audio/wav")},
    )

    assert response.status_code == 400
    assert "injected" in response.json()["detail"]
    assert created
    assert all(not path.exists() for path in created)
    assert not api._synthesis_slot.locked()


def test_oversized_upload_removes_partial_temp_file(monkeypatch):
    created = _track_tempfiles(monkeypatch)
    monkeypatch.setattr(api, "get_engine", lambda: pytest.fail("engine must not load for oversized input"))

    response = client.post(
        "/v1/synthesize",
        data={"text": "oversized", "consent": "true"},
        files={"reference_audio": ("reference.wav", b"x" * (25 * 1024 * 1024 + 1), "audio/wav")},
    )

    assert response.status_code == 413
    assert created
    assert all(not path.exists() for path in created)
    assert not api._synthesis_slot.locked()


def test_text_over_limit_is_rejected_by_request_validation(monkeypatch):
    monkeypatch.setattr(api, "get_engine", lambda: pytest.fail("engine must not load for invalid request"))

    response = client.post(
        "/v1/synthesize",
        data={"text": "x" * 5001, "consent": "true"},
        files={"reference_audio": ("reference.wav", b"reference", "audio/wav")},
    )

    assert response.status_code == 422
    assert not api._synthesis_slot.locked()


def test_missing_consent_is_rejected_before_engine_load(monkeypatch):
    monkeypatch.setattr(api, "get_engine", lambda: pytest.fail("engine must not load without consent"))

    response = client.post(
        "/v1/synthesize",
        data={"text": "consent is required"},
        files={"reference_audio": ("reference.wav", b"reference", "audio/wav")},
    )

    assert response.status_code == 422
    assert not api._synthesis_slot.locked()
