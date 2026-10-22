"""Opt-in integration contract for HarmoniaCodec.

This module deliberately does not implement a codec or load arbitrary model code.
The repository's HarmoniaCodec prototype has no verified trained weights/quality
results yet, so ElevenX must keep its current synthesis path unchanged until a
real adapter and reproducible quality report are supplied.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class CodecQualityReport:
    """Results from an external, reproducible encode/decode benchmark."""

    weights_sha256: str
    benchmark_id: str
    passed: bool


@runtime_checkable
class HarmoniaCodecAdapter(Protocol):
    """Interface an independently reviewed, trained codec adapter must implement."""

    def encode(self, audio: object, sample_rate: int) -> object:
        """Encode audio into the codec's representation."""
        ...

    def decode(self, encoded: object) -> object:
        """Decode a representation into audio."""
        ...


@dataclass(frozen=True)
class HarmoniaCodecReadiness:
    ready: bool
    reasons: tuple[str, ...]


def check_harmonia_readiness(
    *,
    enabled: bool,
    weights_path: str | Path | None,
    report: CodecQualityReport | None,
    expected_weights_sha256: str | None,
) -> HarmoniaCodecReadiness:
    """Check prerequisites without loading weights or changing the active engine.

    A passing report must identify the exact expected weights and a non-empty
    benchmark ID. This is a deployment guard, not a substitute for reviewing
    the benchmark methodology or validating the codec implementation.
    """
    reasons: list[str] = []
    if not enabled:
        reasons.append("HarmoniaCodec is disabled by configuration")
    path = Path(weights_path) if weights_path is not None else None
    if path is None or not path.is_file():
        reasons.append("trained weights file is missing")
    if report is None:
        reasons.append("quality report is missing")
    else:
        if not report.passed:
            reasons.append("quality report did not pass")
        if not report.benchmark_id.strip():
            reasons.append("quality report has no benchmark identifier")
        if not expected_weights_sha256:
            reasons.append("expected weights SHA-256 is not configured")
        elif report.weights_sha256.lower() != expected_weights_sha256.lower():
            reasons.append("quality report does not match the configured weights SHA-256")
    return HarmoniaCodecReadiness(ready=not reasons, reasons=tuple(reasons))
