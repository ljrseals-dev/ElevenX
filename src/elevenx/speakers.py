"""Reference audio validation and optional ECAPA-TDNN embeddings."""
from __future__ import annotations
from pathlib import Path
from typing import Any, Optional, Tuple, Union
from math import gcd
import numpy as np
import soundfile as sf

class SpeakerEncoder:
    """Compute speaker embeddings without persisting reference audio or vectors."""
    def __init__(self, source: str = "speechbrain/spkrec-ecapa-voxceleb",
                 savedir: str = ".models/speechbrain-ecapa", device: str = "cpu") -> None:
        self.source, self.savedir, self.device = source, savedir, device
        self._encoder: Optional[Any] = None

    def _load(self) -> Any:
        if self._encoder is None:
            try:
                from speechbrain.inference.speaker import EncoderClassifier
            except ImportError:
                from speechbrain.pretrained import EncoderClassifier
            self._encoder = EncoderClassifier.from_hparams(
                source=self.source, savedir=self.savedir, run_opts={"device": self.device})
        return self._encoder

    @staticmethod
    def read_reference(path: Union[str, Path], target_rate: int = 16000,
                       min_seconds: float = 3.0, max_seconds: float = 30.0
                       ) -> Tuple[np.ndarray, int]:
        audio_path = Path(path)
        if not audio_path.is_file():
            raise FileNotFoundError(f"Speaker reference not found: {audio_path}")
        audio, rate = sf.read(audio_path, dtype="float32", always_2d=True)
        audio = audio.mean(axis=1)
        duration = len(audio) / float(rate)
        if duration < min_seconds:
            raise ValueError(f"Reference must be at least {min_seconds:g} seconds")
        if duration > max_seconds:
            raise ValueError(f"Reference must be no longer than {max_seconds:g} seconds")
        if not len(audio) or not np.isfinite(audio).all():
            raise ValueError("Reference audio must contain finite, non-empty samples")
        if rate != target_rate:
            from scipy.signal import resample_poly
            divisor = gcd(int(rate), int(target_rate))
            audio = resample_poly(audio, target_rate // divisor, rate // divisor)
            rate = target_rate
        peak = float(np.max(np.abs(audio)))
        if peak > 1.0:
            audio = audio / peak
        return np.asarray(audio, dtype=np.float32), int(rate)

    def embed(self, path: str | Path) -> np.ndarray:
        """Return a normalized ECAPA vector; it is not an XTTS conditioning latent."""
        import torch
        audio, rate = self.read_reference(path)
        if rate != 16000:
            raise RuntimeError("Speaker encoder expects 16 kHz audio")
        waveform = torch.from_numpy(audio).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            embedding = self._load().encode_batch(waveform)
            embedding = torch.nn.functional.normalize(embedding, p=2, dim=-1)
        return embedding.squeeze().detach().cpu().numpy().astype(np.float32)
