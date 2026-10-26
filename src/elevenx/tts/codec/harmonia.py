"""Reserved HarmoniaCodec slot implementing the frozen interface."""
from .interface import BaseCodec, CodecLatent, CodecConfig
from .registry import register_codec

@register_codec("harmonia")
class HarmoniaCodec(BaseCodec):
    def __init__(self, sr: int = 44100, frame_rate: int = 50):
        self.sr=int(sr); self.frame_rate=int(frame_rate); self.hop=max(1,self.sr//self.frame_rate)
    @property
    def config(self):
        return CodecConfig("harmonia",512,self.frame_rate,self.hop,True,"3.9")
    def load(self, model_path=None):
        raise NotImplementedError("HarmoniaCodec weights/runtime are not installed")
    def encode(self, phonemes, prosody):
        raise NotImplementedError("HarmoniaCodec encoder is not installed")
    def decode(self, latent):
        raise NotImplementedError("HarmoniaCodec decoder is not installed")
