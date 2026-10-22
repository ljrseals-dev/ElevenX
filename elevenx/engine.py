"""XTTS-v2 adapter used by the ElevenX API.

This adapter deliberately bypasses HarmoniaCodec unless a real, trained codec
adapter is explicitly supplied. XTTS streaming is chunked model inference, not a
guarantee of sub-10 ms end-to-end latency.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np
import torch
import yaml
from TTS.api import TTS


class ElevenXEngine:
    """Thin XTTS-v2 streaming wrapper with per-reference conditioning cache."""

    def __init__(self, config_path: str = "config.yaml") -> None:
        self.config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
        system = self.config.get("system", {})
        self.sample_rate = int(system.get("sample_rate", 24000))
        requested_device = system.get("device", "auto")
        if requested_device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        elif requested_device == "cuda" and not torch.cuda.is_available():
            self.device = "cpu"
        else:
            self.device = requested_device
        tts_cfg = self.config.get("tts", {})
        model_name = tts_cfg.get(
            "model_name", "tts_models/multilingual/multi-dataset/xtts_v2"
        )
        self.language = tts_cfg.get("language", "en")
        self.chunk_size = int(system.get("streaming_chunk_size", 20))
        self.use_codec = bool(system.get("use_codec", False))
        if self.use_codec:
            raise RuntimeError(
                "HarmoniaCodec is disabled: this checkout does not provide a verified "
                "trained codec adapter. Set system.use_codec: false."
            )
        self.tts = TTS(model_name=model_name).to(self.device)
        self._conditioning: dict[tuple[str, int], tuple[torch.Tensor, torch.Tensor]] = {}

    def _get_conditioning(self, reference_audio_path: str):
        path = Path(reference_audio_path).resolve(strict=True)
        if not path.is_file():
            raise ValueError("Reference audio must be a file")
        key = (str(path), path.stat().st_mtime_ns)
        cached = self._conditioning.get(key)
        if cached is not None:
            return cached
        model = self.tts.synthesizer.tts_model
        with torch.inference_mode():
            latents = model.get_conditioning_latents(audio_path=[str(path)])
        self._conditioning = {key: latents}
        return latents

    def synthesize_stream(
        self, text: str, reference_audio_path: str, emotion: str = "neutral"
    ) -> Iterator[np.ndarray]:
        """Yield mono float32 chunks; emotion labels are metadata only.

        XTTS-v2 does not expose a stable general-purpose four-axis emotion API.
        We intentionally do not prepend fake emotion tokens or imply that they
        reliably control confidence, empathy, authority, or aggression.
        """
        if not text.strip():
            raise ValueError("text must not be empty")
        allowed = {"neutral", "confidence", "empathy", "authority", "aggression"}
        if emotion not in allowed:
            raise ValueError(f"emotion must be one of: {', '.join(sorted(allowed))}")
        gpt_cond_latent, speaker_embedding = self._get_conditioning(reference_audio_path)
        model = self.tts.synthesizer.tts_model
        with torch.inference_mode():
            chunks = model.inference_stream(
                text=text,
                language=self.language,
                gpt_cond_latent=gpt_cond_latent,
                speaker_embedding=speaker_embedding,
                stream_chunk_size=self.chunk_size,
                enable_text_splitting=True,
                temperature=0.75,
                repetition_penalty=7.0,
            )
            for chunk in chunks:
                if isinstance(chunk, torch.Tensor):
                    audio = chunk.detach().float().squeeze().cpu().numpy()
                else:
                    audio = np.asarray(chunk, dtype=np.float32).squeeze()
                if audio.ndim != 1:
                    raise RuntimeError("XTTS stream yielded an unexpected audio shape")
                yield np.nan_to_num(audio.astype(np.float32, copy=False))

    def synthesize_full(
        self, text: str, reference_audio_path: str, emotion: str = "neutral",
        spatial: bool = False
    ) -> np.ndarray:
        chunks = list(self.synthesize_stream(text, reference_audio_path, emotion))
        if not chunks:
            return np.zeros(0, dtype=np.float32)
        audio = np.concatenate(chunks)
        if spatial:
            spatial_cfg = self.config.get("spatial", {})
            if not spatial_cfg.get("enabled", False):
                raise ValueError("Spatial rendering was requested but is disabled in config")
            try:
                from .spatial_audio import SpatialAudioRenderer
            except ImportError as exc:
                raise RuntimeError("Install optional pyroomacoustics to enable spatial audio") from exc
            renderer = SpatialAudioRenderer(self.config)
            audio = renderer.render(audio)
        return audio
