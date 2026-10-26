"""Numpy-only drop-in TTS engine with a HarmoniaCodec slot.

This package is vendored into ElevenX as an optional, zero-dependency
synthesis path. The primary engine remains Coqui XTTS-v2. HarmoniaCodec
remains a reserved slot that falls back to the baseline formant codec
until trained weights and a quality report are supplied.
"""
from .engine import TTSEngine, available_codecs

__all__ = ["TTSEngine", "available_codecs"]
