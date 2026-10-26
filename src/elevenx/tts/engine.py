"""Codec-agnostic TTS orchestration with safe codec fallback."""
import warnings
import numpy as np
from .codec.registry import get_codec, autoload, list_codecs
from .frontend import text_to_phonemes, plan_prosody
from .dsp import NoiseBed, MultibandCompressor, ProDither

class TTSEngine:
    def __init__(self,codec_name="baseline",codec_path=None,sample_rate=22050,fallback="baseline"):
        autoload(); self.sr=int(sample_rate); self.fallback=fallback; self._warned=set(); self.codec=self._instantiate(codec_name,codec_path)
        self.noise_bed=NoiseBed(); self.mb_comp=MultibandCompressor(self.sr); self.dither=ProDither(24)
    def _instantiate(self,name,path=None):
        try:
            cls=get_codec(name); inst=cls(sr=self.sr); inst.load(path); return inst
        except Exception as exc:
            if name==self.fallback: raise
            self._warn_once(f"init:{name}",f"Codec '{name}' failed ({exc!r}); falling back to '{self.fallback}'.")
            cls=get_codec(self.fallback); inst=cls(sr=self.sr); inst.load(None); return inst
    def _warn_once(self,key,msg):
        if key not in self._warned: warnings.warn(msg,RuntimeWarning,stacklevel=2); self._warned.add(key)
    def _fallback_render(self,phonemes,prosody):
        old=self.codec; self.codec=self._instantiate(self.fallback,None)
        try:
            latent=self.codec.encode(phonemes,prosody); audio=self.codec.decode(latent); return latent,audio
        finally:
            if old is not self.codec:
                try: old.unload()
                except Exception: pass
    def synth(self,text):
        phonemes=text_to_phonemes(text); prosody=plan_prosody(phonemes,self.codec.config.frame_rate,text)
        try:
            latent=self.codec.encode(phonemes,prosody)
        except NotImplementedError as exc:
            self._warn_once("encode",f"{self.codec.config.name}.encode unavailable; using '{self.fallback}'.")
            latent,audio=self._fallback_render(phonemes,prosody)
        else:
            try:
                audio=self.codec.decode(latent)
            except NotImplementedError:
                self._warn_once("decode",f"{self.codec.config.name}.decode unavailable; regenerating with '{self.fallback}'.")
                latent,audio=self._fallback_render(phonemes,prosody)
        original_dtype=getattr(audio,"dtype",None); audio=np.asarray(audio,dtype=np.float32)
        if original_dtype is not None and original_dtype != np.dtype(np.float32): self._warn_once("dtype",f"Codec returned {original_dtype}; cast to float32.")
        if audio.size==0:return np.zeros(0,dtype=np.float32)
        source_sr=int(getattr(latent,"sample_rate",self.sr))
        if source_sr!=self.sr:
            new_len=max(1,int(round(len(audio)*self.sr/source_sr)))
            audio=np.interp(np.linspace(0,len(audio)-1,new_len),np.arange(len(audio)),audio).astype(np.float32)
        audio=self.noise_bed.process(audio,self.sr); audio=self.mb_comp.process(audio); audio=self.dither.process(audio)
        peak=float(np.max(np.abs(audio)))
        return (audio*.95/peak if peak>1e-9 else audio).astype(np.float32)
    def swap_codec(self,name,path=None):
        try:self.codec.unload()
        except Exception:pass
        self.codec=self._instantiate(name,path)

def available_codecs(): autoload(); return list_codecs()
