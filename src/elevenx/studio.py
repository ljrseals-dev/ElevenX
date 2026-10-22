"""AOC-style vocal arrangement primitives for ElevenX.

This module describes independent vocal parts; it does not claim that XTTS can
sing a specified melody or produce pitch-accurate harmonies. A capable vocal
renderer is injected by the caller.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np


_ALLOWED_PART_COUNTS = {2, 3, 4, 6}


@dataclass(frozen=True)
class VocalPart:
    """One editable part in a vocal arrangement."""

    name: str
    role: str
    prompt: str
    gain: float
    pan: float
    offset_seconds: float = 0.0

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("part name must not be empty")
        if not self.role.strip():
            raise ValueError("part role must not be empty")
        if not self.prompt.strip():
            raise ValueError("part prompt must not be empty")
        if not 0.0 <= self.gain <= 2.0:
            raise ValueError("gain must be between 0 and 2")
        if not -1.0 <= self.pan <= 1.0:
            raise ValueError("pan must be between -1 and 1")
        if self.offset_seconds < 0:
            raise ValueError("offset_seconds must not be negative")


@dataclass(frozen=True)
class VocalArrangement:
    """A stable list of separately renderable vocal parts."""

    title: str
    parts: tuple[VocalPart, ...]

    def __post_init__(self) -> None:
        if not self.title.strip():
            raise ValueError("arrangement title must not be empty")
        if not self.parts:
            raise ValueError("arrangement must contain at least one part")
        names = [part.name.casefold() for part in self.parts]
        if len(names) != len(set(names)):
            raise ValueError("part names must be unique")


class VocalRenderer(Protocol):
    """Adapter contract for a renderer capable of generating the requested part."""

    def render_part(
        self, part: VocalPart, reference_audio: str | Path
    ) -> tuple[np.ndarray, int]:
        """Return mono float audio and its actual sample rate."""
        ...


def build_harmony_arrangement(
    title: str,
    lead_prompt: str,
    *,
    part_count: int = 4,
    include_adlibs: bool = True,
) -> VocalArrangement:
    """Build an editable 2/3/4/6-part arrangement plan.

    These are production directions, not generated audio or guaranteed musical
    intervals. A renderer must implement each direction for the result to sound
    like a harmony stack.
    """
    if part_count not in _ALLOWED_PART_COUNTS:
        raise ValueError("part_count must be one of 2, 3, 4, or 6")
    if not lead_prompt.strip():
        raise ValueError("lead_prompt must not be empty")

    parts: list[VocalPart] = [
        VocalPart("Lead", "lead", lead_prompt.strip(), 1.0, 0.0)
    ]
    backing = [
        ("Harmony High L", "high_harmony", "Sing a distinct higher harmony; do not double the lead.", 0.62, -0.55),
        ("Harmony Low R", "low_harmony", "Sing a distinct lower harmony; do not double the lead.", 0.62, 0.55),
        ("Double L", "double", "Perform a tight, natural double of the lead.", 0.48, -0.8),
        ("Double R", "double", "Perform a tight, natural double of the lead.", 0.48, 0.8),
    ]
    for name, role, direction, gain, pan in backing[: part_count - 1]:
        parts.append(VocalPart(name, role, f"{lead_prompt.strip()}\n\nDirection: {direction}", gain, pan))

    if include_adlibs:
        parts.append(
            VocalPart(
                "Ad-libs", "adlib",
                f"Create short call-and-response ad-libs around this lyric: {lead_prompt.strip()}",
                0.4, 0.25, 0.15,
            )
        )
    return VocalArrangement(title=title.strip(), parts=tuple(parts))


def render_arrangement(
    arrangement: VocalArrangement,
    renderer: VocalRenderer,
    reference_audio: str | Path,
) -> dict[str, tuple[np.ndarray, int]]:
    """Render each part independently and validate the renderer's output.

    The function deliberately returns stems instead of silently mixing tracks.
    Every returned waveform must be finite, non-empty, mono, and use a positive
    sample rate. A renderer may implement synthesis or route to another engine.
    """
    stems: dict[str, tuple[np.ndarray, int]] = {}
    for part in arrangement.parts:
        audio, sample_rate = renderer.render_part(part, reference_audio)
        samples = np.asarray(audio, dtype=np.float32)
        if samples.ndim != 1:
            raise ValueError(f"{part.name}: renderer must return mono audio")
        if samples.size == 0 or not np.isfinite(samples).all():
            raise ValueError(f"{part.name}: renderer returned empty or non-finite audio")
        if not isinstance(sample_rate, int) or sample_rate <= 0:
            raise ValueError(f"{part.name}: sample rate must be a positive integer")
        stems[part.name] = (samples, sample_rate)
    return stems
