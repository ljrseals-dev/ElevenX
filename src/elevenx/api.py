"""Local-only ElevenX Voice Studio API with serialized XTTS synthesis."""
from __future__ import annotations

import os
import tempfile
import threading
from pathlib import Path
from typing import Final

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask

from .emotions import Emotion
from .engine import ElevenXEngine

app = FastAPI(title="ElevenX", version="0.1.1", description="Local-first voice synthesis studio")
_engine: ElevenXEngine | None = None
_engine_lock = threading.Lock()
_synthesis_slot = threading.Lock()
_WEB_APP = Path(__file__).resolve().parent / "static" / "index.html"
_MAX_UPLOAD_BYTES: Final = 25 * 1024 * 1024
_UPLOAD_CHUNK_BYTES: Final = 1024 * 1024
_ALLOWED_SUFFIXES: Final = {".wav", ".flac", ".ogg", ".aiff", ".aif"}


def get_engine() -> ElevenXEngine:
    """Initialize the heavy model only when a synthesis request needs it."""
    global _engine
    if _engine is None:
        with _engine_lock:
            if _engine is None:
                _engine = ElevenXEngine(os.environ.get("ELEVENX_CONFIG", "config.yaml"))
    return _engine


@app.get("/", include_in_schema=False)
def web_app() -> FileResponse:
    """Serve the browser-based Voice Studio."""
    if not _WEB_APP.is_file():
        raise HTTPException(status_code=500, detail="ElevenX web interface is missing")
    return FileResponse(_WEB_APP, media_type="text/html; charset=utf-8")


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness only; this endpoint does not load model weights."""
    return {
        "status": "ok",
        "service": "ElevenX",
        "interface": "web",
        "mode": "complete-waveform",
    }


@app.post("/v1/synthesize")
async def synthesize(
    text: str = Form(..., max_length=5000),
    language: str = Form("en", min_length=2, max_length=12),
    emotion: str = Form("neutral"),
    consent: bool = Form(...),
    reference_audio: UploadFile = File(...),
) -> FileResponse:
    """Generate a WAV, reject concurrent model calls, and clean temporary files."""
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
    if suffix not in _ALLOWED_SUFFIXES:
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
                chunk = await reference_audio.read(_UPLOAD_CHUNK_BYTES)
                if not chunk:
                    break
                total += len(chunk)
                if total > _MAX_UPLOAD_BYTES:
                    raise HTTPException(status_code=413, detail="Reference file exceeds 25 MiB")
                handle.write(chunk)
        if total == 0:
            raise HTTPException(status_code=400, detail="Reference audio is empty")

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
        result = await run_in_threadpool(
            engine.synthesize, text.strip(), reference_path, language, selected_emotion.value
        )
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
