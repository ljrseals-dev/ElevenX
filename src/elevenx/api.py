"""Local-only ElevenX web app and synthesis API."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Final

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from starlette.background import BackgroundTask
from fastapi.responses import FileResponse

from .emotions import Emotion
from .engine import ElevenXEngine

app = FastAPI(
    title="ElevenX",
    version="0.1.0",
    description="Local-first voice synthesis studio",
)
_engine: ElevenXEngine | None = None
_WEB_APP = Path(__file__).resolve().parent / "static" / "index.html"
_MAX_UPLOAD_BYTES: Final = 25 * 1024 * 1024
_UPLOAD_CHUNK_BYTES: Final = 1024 * 1024
_ALLOWED_SUFFIXES: Final = {".wav", ".flac", ".ogg", ".aiff", ".aif"}


def get_engine() -> ElevenXEngine:
    """Load the inference engine only when a synthesis request needs it."""
    global _engine
    if _engine is None:
        _engine = ElevenXEngine(os.environ.get("ELEVENX_CONFIG", "config.yaml"))
    return _engine


@app.get("/", include_in_schema=False)
def web_app() -> FileResponse:
    """Serve the actual ElevenX Voice Studio UI."""
    if not _WEB_APP.is_file():
        raise HTTPException(status_code=500, detail="ElevenX web interface is missing")
    return FileResponse(_WEB_APP, media_type="text/html; charset=utf-8")


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness only; this endpoint deliberately does not load model weights."""
    return {"status": "ok", "service": "ElevenX", "interface": "web"}


@app.post("/v1/synthesize")
async def synthesize(
    text: str = Form(..., max_length=5000),
    language: str = Form("en", min_length=2, max_length=12),
    emotion: str = Form("neutral"),
    reference_audio: UploadFile = File(...),
) -> FileResponse:
    """Generate a WAV from multipart form fields and clean temporary files."""
    suffix = Path(reference_audio.filename or "reference.wav").suffix.lower()
    if suffix not in _ALLOWED_SUFFIXES:
        await reference_audio.close()
        raise HTTPException(status_code=415, detail="Unsupported reference audio format")
    if not text.strip():
        await reference_audio.close()
        raise HTTPException(status_code=422, detail="Text must not be empty")
    try:
        selected_emotion = Emotion(emotion)
    except ValueError as exc:
        await reference_audio.close()
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported emotion. Choose one of: {', '.join(item.value for item in Emotion)}",
        ) from exc

    reference_path: Path | None = None
    output_path: Path | None = None
    try:
        # Enforce the cap while reading, not after allocating an unbounded upload.
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
            reference_path = Path(handle.name)
            total = 0
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

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
            output_path = Path(handle.name)

        engine = get_engine()
        result = engine.synthesize(text, reference_path, language, selected_emotion.value)
        engine.save_wav(result, output_path)

        # FileResponse reads the file before running its background cleanup.
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
    finally:
        await reference_audio.close()
