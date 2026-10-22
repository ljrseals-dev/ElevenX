"""FastAPI service for local ElevenX synthesis.

Neural dependencies are loaded only when synthesis is requested so health checks
and unit tests can run without downloading model weights.
"""
from __future__ import annotations

import io
import os
import wave
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import Response
from pydantic import BaseModel, Field

CONFIG_PATH = Path(os.getenv("ELEVENX_CONFIG", "config.yaml")).resolve()
REFERENCE_ROOT = Path(os.getenv("ELEVENX_REFERENCE_ROOT", "./references")).resolve()
app = FastAPI(title="ElevenX Voice AI API", version="0.2.0")


def load_config() -> dict[str, Any]:
    """Load YAML config with safe defaults; fail clearly on malformed config."""
    if not CONFIG_PATH.exists():
        return {
            "system": {"sample_rate": 24000, "device": "auto", "use_codec": False},
            "tts": {"language": "en", "emotion_prompts": {}},
            "server": {"max_text_characters": 4000},
        }
    try:
        data = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise RuntimeError(f"Cannot load ElevenX config at {CONFIG_PATH}: {exc}") from exc
    if not isinstance(data, dict):
        raise RuntimeError("ElevenX config must be a YAML mapping")
    return data


def resolve_reference_path(value: str) -> Path:
    """Resolve a reference filename strictly beneath the configured root."""
    if not value or not value.strip():
        raise ValueError("reference_audio must be a filename")
    root = REFERENCE_ROOT.resolve()
    candidate = (root / value).resolve()
    if candidate == root or root not in candidate.parents:
        raise ValueError("reference_audio must remain inside the configured reference directory")
    if not candidate.is_file():
        raise FileNotFoundError("Reference audio file was not found")
    if candidate.suffix.lower() not in {".wav", ".flac", ".ogg"}:
        raise ValueError("Reference audio must be WAV, FLAC, or OGG")
    return candidate


def pcm16_bytes(audio: np.ndarray) -> bytes:
    """Convert finite floating-point audio to clipped little-endian PCM16."""
    array = np.asarray(audio, dtype=np.float32)
    if array.size == 0:
        return b""
    array = np.nan_to_num(array, nan=0.0, posinf=1.0, neginf=-1.0)
    array = np.clip(array, -1.0, 1.0)
    return (array * 32767.0).astype("<i2").tobytes()


def wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    """Encode mono or interleaved multichannel floating-point audio as WAV."""
    array = np.asarray(audio, dtype=np.float32)
    if array.ndim == 1:
        channels = 1
    elif array.ndim == 2:
        channels = array.shape[1]
    else:
        raise ValueError("audio must have shape (samples,) or (samples, channels)")
    if sample_rate <= 0 or channels < 1:
        raise ValueError("sample_rate and channel count must be positive")
    raw = pcm16_bytes(array)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as output:
        output.setnchannels(channels)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(raw)
    return buffer.getvalue()


class SynthesisRequest(BaseModel):
    text: str = Field(min_length=1, max_length=4000)
    reference_audio: str = Field(min_length=1, max_length=255)
    emotion: str = "neutral"
    spatial: bool = False


@lru_cache(maxsize=1)
def get_engine():
    """Initialize the heavyweight inference engine on first synthesis."""
    try:
        from .engine import ElevenXEngine
    except ImportError as exc:
        raise RuntimeError(
            "Inference engine is not installed/configured. Install the optional TTS "
            "dependencies and provide elevenx/engine.py."
        ) from exc
    return ElevenXEngine(str(CONFIG_PATH))


@app.get("/health")
def health():
    return {"status": "ok", "engine": "ElevenX", "model_loaded": get_engine.cache_info().currsize > 0}


@app.post("/synthesize")
def synthesize(request: SynthesisRequest):
    config = load_config()
    limit = int(config.get("server", {}).get("max_text_characters", 4000))
    if len(request.text) > limit:
        raise HTTPException(status_code=413, detail="Text exceeds configured length limit")
    try:
        reference = resolve_reference_path(request.reference_audio)
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    try:
        engine = get_engine()
        audio = engine.synthesize_full(request.text, str(reference), request.emotion, request.spatial)
        rate = int(getattr(engine, "sample_rate", config.get("system", {}).get("sample_rate", 24000)))
        return Response(content=wav_bytes(audio, rate), media_type="audio/wav")
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.websocket("/ws/tts")
async def websocket_tts(websocket: WebSocket):
    await websocket.accept()
    try:
        data = await websocket.receive_json()
        try:
            request = SynthesisRequest.model_validate(data)
            reference = resolve_reference_path(request.reference_audio)
        except (ValueError, FileNotFoundError) as exc:
            await websocket.send_json({"type": "error", "message": str(exc)})
            return
        if len(request.text) > int(load_config().get("server", {}).get("max_text_characters", 4000)):
            await websocket.send_json({"type": "error", "message": "Text exceeds configured length limit"})
            return
        try:
            engine = get_engine()
            await websocket.send_json({
                "type": "start",
                "sample_rate": int(engine.sample_rate),
                "format": "pcm_s16le",
                "channels": 1,
            })
            # Model inference is blocking; run the generator off the event loop.
            import asyncio
            chunks = await asyncio.to_thread(
                lambda: list(engine.synthesize_stream(request.text, str(reference), request.emotion))
            )
            for chunk in chunks:
                await websocket.send_bytes(pcm16_bytes(chunk))
            await websocket.send_json({"type": "done"})
        except Exception as exc:
            await websocket.send_json({"type": "error", "message": f"Synthesis failed: {type(exc).__name__}"})
    except WebSocketDisconnect:
        return
    finally:
        try:
            await websocket.close()
        except RuntimeError:
            pass
