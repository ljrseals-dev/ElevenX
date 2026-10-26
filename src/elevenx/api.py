"""Local-only development API; remote access is denied unless explicitly enabled."""
from __future__ import annotations

import ipaddress
import os
import tempfile
import threading
from pathlib import Path
from typing import Any

import yaml
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from starlette.background import BackgroundTask
from starlette.responses import FileResponse, JSONResponse, Response

from .emotions import Emotion
from .engine import ElevenXEngine

app = FastAPI(title="ElevenX", version="0.1.3")
_engine: ElevenXEngine | None = None
_engine_lock = threading.Lock()
_synthesis_slot = threading.Lock()
_WEB_APP = Path(__file__).resolve().parent / "static" / "index.html"
_DEFAULT_CONFIG = "config.yaml"
_TEST_HOSTS = {"testclient", "testserver"}


def _config_path() -> Path:
    return Path(os.environ.get("ELEVENX_CONFIG", _DEFAULT_CONFIG)).expanduser()


def _security_config() -> dict[str, Any]:
    """Load security settings on each request; malformed config fails closed."""
    try:
        with _config_path().open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle) or {}
        security = config.get("security", {})
        if not isinstance(security, dict):
            return {}
        return security
    except (OSError, yaml.YAMLError, AttributeError):
        return {}


def remote_api_allowed() -> bool:
    """Remote access is opt-in; absent or invalid config means local-only."""
    return _security_config().get("allow_remote_api") is True


def safe_bind_host() -> str:
    """Return the configured bind host, forcing loopback unless remote is opted in."""
    if remote_api_allowed():
        try:
            with _config_path().open("r", encoding="utf-8") as handle:
                config = yaml.safe_load(handle) or {}
            host = config.get("app", {}).get("host", "127.0.0.1")
            return host if isinstance(host, str) and host.strip() else "127.0.0.1"
        except (OSError, yaml.YAMLError, AttributeError):
            pass
    return "127.0.0.1"


def _is_loopback(value: str | None) -> bool:
    if not value:
        return False
    candidate = value.strip().strip("[]")
    if candidate.lower() in {"localhost", "localhost.localdomain", *_TEST_HOSTS}:
        return True
    try:
        return ipaddress.ip_address(candidate).is_loopback
    except ValueError:
        return False


class LocalOnlyMiddleware:
    """Fail closed if a remote client or non-loopback server reaches the app."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or remote_api_allowed():
            await self.app(scope, receive, send)
            return

        client = scope.get("client")
        client_host = client[0] if client else None
        server = scope.get("server")
        server_host = server[0] if server else None
        headers = {key.lower(): value.decode("latin-1") for key, value in scope.get("headers", [])}
        host_header = headers.get(b"host") if False else headers.get("host", "")
        # TestClient uses synthetic hostnames; production requests must use loopback.
        test_request = client_host in _TEST_HOSTS or server_host in _TEST_HOSTS
        host_name = host_header.rsplit(":", 1)[0] if host_header.count(":") == 1 else host_header
        host_ok = test_request or _is_loopback(host_name)
        client_ok = test_request or _is_loopback(client_host)
        server_ok = test_request or _is_loopback(server_host)
        if not (host_ok and client_ok and server_ok):
            response = JSONResponse(
                {"detail": "Remote API access is disabled; bind to loopback or explicitly enable security.allow_remote_api."},
                status_code=403,
            )
            await response(scope, receive, send)
            return
        await self.app(scope, receive, send)


app.add_middleware(LocalOnlyMiddleware)


def get_engine() -> ElevenXEngine:
    global _engine
    if _engine is None:
        with _engine_lock:
            if _engine is None:
                _engine = ElevenXEngine(str(_config_path()))
    return _engine


@app.get("/", include_in_schema=False)
def web_app() -> FileResponse:
    if not _WEB_APP.is_file():
        raise HTTPException(status_code=500, detail="ElevenX web interface is missing")
    return FileResponse(_WEB_APP, media_type="text/html; charset=utf-8")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "ElevenX", "interface": "web", "mode": "complete-waveform"}


@app.post("/v1/synthesize")
async def synthesize(
    text: str = Form(..., max_length=5000),
    language: str = Form("en", min_length=2, max_length=12),
    emotion: str = Form("neutral"),
    consent: bool = Form(...),
    reference_audio: UploadFile = File(...),
) -> Response:
    if not consent:
        await reference_audio.close()
        raise HTTPException(status_code=403, detail="Confirm permission to use the reference voice")
    if not text.strip():
        await reference_audio.close()
        raise HTTPException(status_code=422, detail="Text must not be empty")
    try:
        selected_emotion = Emotion(emotion)
    except ValueError as exc:
        await reference_audio.close()
        choices = ", ".join(item.value for item in Emotion)
        raise HTTPException(status_code=422, detail=f"Unsupported emotion. Choose one of: {choices}") from exc

    suffix = Path(reference_audio.filename or "reference.wav").suffix.lower()
    if suffix not in {".wav", ".flac", ".ogg", ".aiff", ".aif"}:
        await reference_audio.close()
        raise HTTPException(status_code=415, detail="Unsupported reference audio format")

    reference_path: Path | None = None
    output_path: Path | None = None
    slot_acquired = False
    try:
        total = 0
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
            reference_path = Path(handle.name)
            while True:
                chunk = await reference_audio.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > 25 * 1024 * 1024:
                    raise HTTPException(status_code=413, detail="Reference audio exceeds 25 MiB")
                handle.write(chunk)
        if total == 0:
            raise HTTPException(status_code=400, detail="Reference audio file is empty")

        slot_acquired = _synthesis_slot.acquire(blocking=False)
        if not slot_acquired:
            raise HTTPException(
                status_code=503,
                detail="ElevenX is already synthesizing audio. Wait for it to finish, then retry.",
                headers={"Retry-After": "3"},
            )

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
            output_path = Path(handle.name)

        engine = get_engine()
        result = await run_in_threadpool(engine.synthesize, text.strip(), reference_path, language, selected_emotion.value)
        await run_in_threadpool(engine.save_wav, result, output_path)

        def cleanup() -> None:
            if reference_path is not None:
                reference_path.unlink(missing_ok=True)
            if output_path is not None:
                output_path.unlink(missing_ok=True)

        return FileResponse(
            path=output_path,
            media_type="audio/wav",
            filename="elevenx.wav",
            background=BackgroundTask(cleanup),
        )
    except HTTPException:
        if reference_path is not None:
            reference_path.unlink(missing_ok=True)
        if output_path is not None:
            output_path.unlink(missing_ok=True)
        raise
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        if reference_path is not None:
            reference_path.unlink(missing_ok=True)
        if output_path is not None:
            output_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception:
        if reference_path is not None:
            reference_path.unlink(missing_ok=True)
        if output_path is not None:
            output_path.unlink(missing_ok=True)
        raise
    finally:
        if slot_acquired:
            _synthesis_slot.release()
        await reference_audio.close()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("elevenx.api:app", host=safe_bind_host(), port=8000)
