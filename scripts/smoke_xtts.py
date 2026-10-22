#!/usr/bin/env python3
"""Opt-in XTTS-v2 smoke test; uses only a caller-provided, consented reference recording."""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import soundfile as sf

from elevenx import ElevenXEngine


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, type=Path, help="Consented 3–30 second reference audio file")
    parser.add_argument("--text", default="This is an ElevenX audio generation smoke test.")
    parser.add_argument("--language", default="en")
    parser.add_argument("--emotion", default="neutral")
    parser.add_argument("--config", default=Path("config.yaml"), type=Path)
    parser.add_argument("--output", default=Path("elevenx-smoke.wav"), type=Path)
    args = parser.parse_args()

    if not args.reference.is_file():
        parser.error(f"reference file not found: {args.reference}")

    engine = ElevenXEngine(args.config)
    result = engine.synthesize(args.text, args.reference, args.language, args.emotion)
    if result.audio.size == 0 or not np.isfinite(result.audio).all():
        raise RuntimeError("XTTS returned empty or non-finite audio")
    if result.sample_rate <= 0 or not math.isfinite(result.duration_seconds) or result.duration_seconds <= 0:
        raise RuntimeError("XTTS returned invalid sample rate or duration")

    engine.save_wav(result, args.output)
    decoded, decoded_rate = sf.read(args.output, dtype="float32", always_2d=True)
    if decoded.size == 0 or not np.isfinite(decoded).all():
        raise RuntimeError("saved WAV is empty or contains non-finite samples")
    if decoded_rate != result.sample_rate:
        raise RuntimeError(f"WAV sample rate mismatch: expected {result.sample_rate}, got {decoded_rate}")
    if decoded.shape[1] != 1:
        raise RuntimeError(f"expected mono WAV, got {decoded.shape[1]} channels")
    duration = decoded.shape[0] / decoded_rate
    if not math.isfinite(duration) or duration <= 0:
        raise RuntimeError("saved WAV has invalid duration")

    print("XTTS smoke test passed")
    print(f"Output: {args.output.resolve()}")
    print(f"Sample rate: {decoded_rate} Hz")
    print(f"Channels: {decoded.shape[1]}")
    print(f"Duration: {duration:.2f} s")
    print("Next: play the WAV in a separate audio player to verify audible playback.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
