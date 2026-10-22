"""Local-only development API. Do not expose publicly without security controls."""
from __future__ import annotations
import os
import tempfile
from pathlib import Path
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.background import BackgroundTask
from fastapi.responses import FileResponse
from .engine import ElevenXEngine

app = FastAPI(title="ElevenX", version="0.1.0")
_engine: ElevenXEngine | None = None

def get_engine() -> ElevenXEngine:
    global _engine
    if _engine is None:
        _engine = ElevenXEngine(os.environ.get("ELEVENX_CONFIG", "config.yaml"))
    return _engine

@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "ElevenX"}

@app.post("/v1/synthesize")
async def synthesize(text: str = Form(..., max_length=5000),
                     language: str = Form("en"), emotion: str = Form("neutral"),
                     reference_audio: UploadFile = File(...)) -> FileResponse:
    """Generate WAV from multipart form fields and delete temp files after sending."""
    suffix = Path(reference_audio.filename or "reference.wav").suffix.lower()
    if suffix not in {".wav", ".flac", ".ogg", ".aiff", ".aif"}:
        raise HTTPException(status_code=415, detail="Unsupported reference audio format")
    reference_path = output_path = None
    try:
        content = await reference_audio.read()
        if not content or len(content) > 25 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="Reference file is empty or too large")
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
            reference_path = Path(handle.name)
            handle.write(content)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
            output_path = Path(handle.name)
        result = get_engine().synthesize(text, reference_path, language, emotion)
        get_engine().save_wav(result, output_path)
        def cleanup() -> None:
            if reference_path is not None:
                reference_path.unlink(missing_ok=True)
            if output_path is not None:
                output_path.unlink(missing_ok=True)
        return FileResponse(path=output_path, media_type="audio/wav",
                            filename="elevenx.wav", background=BackgroundTask(cleanup))
    except HTTPException:
        if reference_path: reference_path.unlink(missing_ok=True)
        if output_path: output_path.unlink(missing_ok=True)
        raise
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        if reference_path: reference_path.unlink(missing_ok=True)
        if output_path: output_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        await reference_audio.close()
