from __future__ import annotations

import asyncio

from fastapi.testclient import TestClient

from elevenx import api


def test_local_only_is_default_when_config_is_missing(monkeypatch, tmp_path):
    monkeypatch.setenv("ELEVENX_CONFIG", str(tmp_path / "missing.yaml"))
    assert api.remote_api_allowed() is False
    assert api.safe_bind_host() == "127.0.0.1"


def test_local_only_forces_loopback_even_if_config_requests_wildcard_bind(monkeypatch, tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text("app:\n  host: 0.0.0.0\nsecurity:\n  allow_remote_api: false\n", encoding="utf-8")
    monkeypatch.setenv("ELEVENX_CONFIG", str(config))
    assert api.safe_bind_host() == "127.0.0.1"


def test_explicit_remote_opt_in_uses_configured_host(monkeypatch, tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text("app:\n  host: 0.0.0.0\nsecurity:\n  allow_remote_api: true\n", encoding="utf-8")
    monkeypatch.setenv("ELEVENX_CONFIG", str(config))
    assert api.remote_api_allowed() is True
    assert api.safe_bind_host() == "0.0.0.0"


def test_malformed_security_config_fails_closed(monkeypatch, tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text("security: [not, a, mapping]", encoding="utf-8")
    monkeypatch.setenv("ELEVENX_CONFIG", str(config))
    assert api.remote_api_allowed() is False
    assert api.safe_bind_host() == "127.0.0.1"


def test_middleware_rejects_remote_client_and_wildcard_server(monkeypatch, tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text("security:\n  allow_remote_api: false\n", encoding="utf-8")
    monkeypatch.setenv("ELEVENX_CONFIG", str(config))
    messages = []

    async def downstream(scope, receive, send):
        raise AssertionError("remote request must not reach the application")

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        messages.append(message)

    scope = {
        "type": "http",
        "method": "GET",
        "path": "/health",
        "headers": [(b"host", b"example.com")],
        "client": ("203.0.113.10", 4321),
        "server": ("0.0.0.0", 8000),
    }
    asyncio.run(api.LocalOnlyMiddleware(downstream)(scope, receive, send))
    assert messages[0]["status"] == 403
    assert b"Remote API access is disabled" in messages[1]["body"]


def test_loopback_health_endpoint_remains_available(monkeypatch, tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text("security:\n  allow_remote_api: false\n", encoding="utf-8")
    monkeypatch.setenv("ELEVENX_CONFIG", str(config))
    response = TestClient(api.app).get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
