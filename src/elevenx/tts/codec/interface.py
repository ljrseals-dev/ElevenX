"""Frozen codec contract for the drop-in TTS engine."""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
import numpy as np

@dataclass
class CodecConfig:
    name: str
    latent_dim: int
    frame_rate: int
    hop_length: int
    supports_streaming: bool = False
    min_python: str = "3.9"

@dataclass
class CodecLatent:
    tokens: np.ndarray
    sample_rate: int
    frame_hop: int
    speaker_id: Optional[int] = None
    metadata: Optional[dict] = None

class BaseCodec(ABC):
    @property
    @abstractmethod
    def config(self) -> CodecConfig:
        raise NotImplementedError

    @abstractmethod
    def load(self, model_path: Optional[str] = None) -> None:
        raise NotImplementedError

    @abstractmethod
    def encode(self, phonemes: np.ndarray, prosody: dict) -> CodecLatent:
        raise NotImplementedError

    @abstractmethod
    def decode(self, latent: CodecLatent) -> np.ndarray:
        raise NotImplementedError

    def unload(self) -> None:
        pass
