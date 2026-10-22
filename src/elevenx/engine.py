"""ElevenX synthesis orchestration with lazy model loading."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import numpy as np
import yaml
from .emotions import Emotion, EmotionPreset
from .speakers import SpeakerEncoder

@dataclass(frozen=True)
class SynthesisResult:
    audio: np.ndarray
    sample_rate: int
    emotion: str
    duration_seconds: float

class ElevenXEngine:
    """Synthesize speech from text using a reference voice the caller may use."""
    def __init__(self, config_path: str | Path = "config.yaml") -> None:
        path = Path(config_path)
        if not path.is_file():
            raise FileNotFoundError(f"ElevenX config not found: {path}")
        with path.open("r", encoding="utf-8") as handle:
            self.config: dict[str, Any] = yaml.safe_load(handle) or {}
        self.sample_rate = int(self.config.get("audio", {}).get("sample_rate", 24000))
        model_cfg = self.config.get("models", {})
        requested = str(model_cfg.get("device", "auto"))
        if requested == "auto":
            import torch
            requested = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = requested
        self._tts: Any | None = None
        self._speaker_encoder: SpeakerEncoder | None = None

    def _load_tts(self) -> Any:
        if self._tts is None:
            try:
                from TTS.api import TTS
            except ImportError as exc:
                raise RuntimeError("Coqui TTS is not installed; install requirements.txt") from exc
            name = self.config.get("models", {}).get(
                "xtts_model", "tts_models/multilingual/multi-dataset/xtts_v2")
            self._tts = TTS(model_name=name, progress_bar=False).to(self.device)
        return self._tts

    def _get_speaker_encoder(self) -> SpeakerEncoder:
        if self._speaker_encoder is None:
            cfg = self.config.get("models", {})
            self._speaker_encoder = SpeakerEncoder(
                source=str(cfg.get("speaker_encoder_source", "speechbrain/spkrec-ecapa-voxceleb")),
                savedir=str(Path(cfg.get("cache_dir", ".models")) / "speechbrain-ecapa"),
                device=self.device)
        return self._speaker_encoder

    def speaker_embedding(self, reference_audio: str | Path) -> np.ndarray:
        """Return an ECAPA embedding for analysis or comparison."""
        if not self.config.get("models", {}).get("load_speaker_encoder", True):
            raise RuntimeError("Speaker encoder is disabled in config.yaml")
        return self._get_speaker_encoder().embed(reference_audio)

    def synthesize(self, text: str, reference_audio: str | Path,
                   language: str | None = None,
                   emotion: Emotion | str | None = None) -> SynthesisResult:
        """Generate a mono waveform; style presets are hints, not emotion guarantees."""
        text = text.strip()
        max_chars = int(self.config.get("security", {}).get("max_text_characters", 5000))
        if not text:
            raise ValueError("text must not be empty")
        if len(text) > max_chars:
            raise ValueError(f"text exceeds configured limit of {max_chars} characters")
        audio_cfg = self.config.get("audio", {})
        SpeakerEncoder.read_reference(
            reference_audio, target_rate=16000,
            min_seconds=float(audio_cfg.get("reference_min_seconds", 3.0)),
            max_seconds=float(audio_cfg.get("reference_max_seconds", 30.0)))
        synth_cfg = self.config.get("synthesis", {})
        chosen = Emotion(emotion or synth_cfg.get("default_emotion", "neutral"))
        raw = synth_cfg.get("emotion_presets", {}).get(chosen.value)
        if raw is None:
            raise ValueError(f"No preset configured for emotion: {chosen.value}")
        preset = EmotionPreset.from_mapping(raw)
        language = language or str(self.config.get("models", {}).get("language", "en"))
        generated = self._load_tts().tts(
            text=text, speaker_wav=str(reference_audio), language=language,
            split_sentences=bool(synth_cfg.get("split_sentences", True)),
            temperature=preset.temperature, top_k=preset.top_k,
            top_p=preset.top_p, speed=preset.speed)
        audio = np.asarray(generated, dtype=np.float32).reshape(-1)
        if audio.size == 0 or not np.isfinite(audio).all():
            raise RuntimeError("XTTS returned empty or non-finite audio")
        peak = float(np.max(np.abs(audio)))
        if peak > 1.0:
            audio = audio / peak
        return SynthesisResult(audio, self.sample_rate, chosen.value,
                               len(audio) / self.sample_rate)

    def save_wav(self, result: SynthesisResult, output_path: str | Path) -> Path:
        """Write output to a caller-selected local WAV path."""
        import soundfile as sf
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        sf.write(target, result.audio, result.sample_rate, subtype="PCM_16")
        return target
