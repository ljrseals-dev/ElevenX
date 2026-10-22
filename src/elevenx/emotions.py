"""Sampling presets; XTTS does not guarantee calibrated emotion control."""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum

class Emotion(str, Enum):
    NEUTRAL = "neutral"
    CONFIDENT = "confident"
    EMPATHETIC = "empathetic"
    AUTHORITATIVE = "authoritative"
    AGGRESSIVE = "aggressive"

@dataclass(frozen=True)
class EmotionPreset:
    temperature: float
    top_k: int
    top_p: float
    speed: float

    @classmethod
    def from_mapping(cls, value: dict[str, float | int]) -> "EmotionPreset":
        result = cls(float(value["temperature"]), int(value["top_k"]),
                     float(value["top_p"]), float(value["speed"]))
        if not 0.1 <= result.temperature <= 2.0:
            raise ValueError("temperature must be between 0.1 and 2.0")
        if result.top_k < 1:
            raise ValueError("top_k must be positive")
        if not 0.0 < result.top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1]")
        if not 0.5 <= result.speed <= 2.0:
            raise ValueError("speed must be between 0.5 and 2.0")
        return result
