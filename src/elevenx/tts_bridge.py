"""Optional bridge from ElevenXEngine-style calls to the drop-in TTS path.

Used when the operator wants a zero-dependency numpy formant path (or the
HarmoniaCodec slot) instead of Coqui XTTS-v2. XTTS remains the default.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from .tts.engine import TTSEngine, available_codecs


@dataclass(frozen=True)
class DropInResult:
    audio: np.ndarray
    sample_rate: int
    codec: str
    duration_seconds: float


def synthesize_dropin(
    text: str,
    *,
    codec: str = "baseline",
    sample_rate: int = 22050,
) -> DropInResult:
    """Synthesize with the vendored drop-in engine.

    Harmonia falls back to baseline with a RuntimeWarning until weights exist.
    """
    text = (text or "").strip()
    if not text:
        raise ValueError("text must not be empty")
    engine = TTSEngine(codec_name=codec, sample_rate=sample_rate)
    audio = engine.synth(text)
    return DropInResult(
        audio=audio,
        sample_rate=engine.sr,
        codec=engine.codec.config.name,
        duration_seconds=float(len(audio) / engine.sr) if audio.size else 0.0,
    )


def list_dropin_codecs() -> list[str]:
    return available_codecs()
