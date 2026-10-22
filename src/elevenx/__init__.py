"""ElevenX voice synthesis package."""
from .engine import ElevenXEngine, SynthesisResult
from .emotions import Emotion, EmotionPreset
__all__ = ["ElevenXEngine", "SynthesisResult", "Emotion", "EmotionPreset"]
