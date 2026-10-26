"""Numpy-only Klatt-style baseline formant synthesizer."""
import numpy as np
from .interface import BaseCodec, CodecLatent, CodecConfig
from .registry import register_codec
from ..frontend import phoneme_name, VOWELS, FRICATIVES, PLOSIVES, NASALS, APPROX

@register_codec("baseline")
class BaselineCodec(BaseCodec):
    def __init__(self, sr: int = 22050, frame_rate: int = 50):
        self.sr = int(sr); self.frame_rate = int(frame_rate); self.hop = max(1, self.sr // self.frame_rate)
        self.rng = np.random.default_rng(12345)
    @property
    def config(self):
        return CodecConfig("baseline", 80, self.frame_rate, self.hop, False, "3.9")
    def load(self, model_path=None):
        return None
    def encode(self, phonemes, prosody):
        dur = np.asarray(prosody.get("duration", np.ones(len(phonemes))), dtype=np.float32)
        f0 = np.asarray(prosody.get("f0", np.full(len(phonemes), 120.0)), dtype=np.float32)
        energy = np.asarray(prosody.get("energy", np.full(len(phonemes), .7)), dtype=np.float32)
        frames = np.maximum(1, np.rint(dur).astype(np.int64))
        total = int(frames.sum())
        tokens = np.zeros((max(total, 1), self.config.latent_dim), dtype=np.float32)
        pos = 0
        for i, pid in enumerate(np.asarray(phonemes, dtype=np.int64)):
            count = int(frames[i]); sl = slice(pos, pos + count)
            tokens[sl, 0] = f0[i] if i < len(f0) else 120.0
            tokens[sl, 1] = float(pid)
            tokens[sl, 2] = energy[i] if i < len(energy) else .7
            pos += count
        return CodecLatent(tokens, self.sr, self.hop, metadata={"phoneme_ids": np.asarray(phonemes,dtype=np.int64).copy()})
    def decode(self, latent):
        tokens = np.asarray(latent.tokens)
        if tokens.size == 0: return np.zeros(0, dtype=np.float32)
        parts = []
        for row in tokens:
            parts.append(self._render_segment(float(row[0]), int(round(row[1])), float(row[2]), latent.frame_hop))
        return self._radiate(np.concatenate(parts).astype(np.float32, copy=False))
    def _source(self, f0, n):
        t = np.arange(n, dtype=np.float64) / self.sr
        if f0 <= 1: return np.zeros(n, dtype=np.float32)
        # Harmonic-rich glottal source with mild spectral rolloff.
        y = np.zeros(n, dtype=np.float64)
        phase = 2*np.pi*f0*t
        for h in range(1, min(60, int((self.sr/2)/f0)+1)):
            y += np.sin(h*phase) / (h ** 0.85)
        return np.tanh(y * .55).astype(np.float32)
    def _resonator(self, x, fc, bw):
        fc = min(max(30.0, float(fc)), self.sr*.45); r = np.exp(-np.pi*float(bw)/self.sr); th = 2*np.pi*fc/self.sr
        a1 = -2*r*np.cos(th); a2 = r*r; b0 = 1-r
        y=np.zeros_like(x,dtype=np.float32); y1=y2=0.0
        for i,xi in enumerate(x):
            yi = b0*float(xi) - a1*y1 - a2*y2; y[i]=yi; y2,y1=y1,yi
        return y
    def _vowel(self, f0, formants, n):
        src=self._source(f0,n); y=src
        for fc,bw,gain in ((formants[0],90,.9),(formants[1],120,.65),(formants[2],180,.45)):
            y=self._resonator(y,fc,bw)*gain
        return np.tanh(y*1.8).astype(np.float32)
    def _render_segment(self,f0,pid,energy,n):
        ph=phoneme_name(pid); y=np.zeros(n,dtype=np.float32)
        if ph in VOWELS:
            formants={"AA":(800,1200,2500),"AE":(660,1720,2410),"AH":(640,1190,2390),"AO":(570,840,2410),"AW":(640,1230,2550),"AY":(660,1700,2400),"EH":(530,1840,2480),"ER":(490,1350,1690),"EY":(530,1840,2480),"IH":(390,1990,2550),"IY":(270,2290,3010),"OW":(570,840,2410),"OY":(570,840,2410),"UH":(440,1020,2240),"UW":(300,870,2240)}[ph]
            y=self._vowel(f0,formants,n)
        elif ph in NASALS:
            y=self._resonator(self._source(f0,n),250,80)*.8; y=self._resonator(y,1000,140)*.5
        elif ph in APPROX:
            y=self._resonator(self._source(f0,n),400,100)*.7; y=self._resonator(y,1300,150)*.5
        elif ph in FRICATIVES:
            noise=self.rng.standard_normal(n).astype(np.float32); centers={"S":6000,"SH":3000,"F":4000,"TH":5000,"V":3500,"Z":5000,"HH":1500,"DH":3000}; y=self._resonator(noise,centers.get(ph,3000),1200)*.35
        elif ph in PLOSIVES:
            bl=max(1,n//5); burst=self.rng.standard_normal(bl).astype(np.float32); centers={"P":1200,"T":4000,"K":2500,"B":1200,"D":4000,"G":2500,"CH":3500,"JH":3000}; y[:bl]=self._resonator(burst,centers.get(ph,2500),900)*.6
        # ADSR per phoneme, preserving a release tail.
        if n:
            a=min(n,max(1,int(.010*self.sr))); d=min(n-a,max(1,int(.020*self.sr))); r=min(n,max(1,int(.030*self.sr)))
            env=np.ones(n,dtype=np.float32)
            env[:a]=np.linspace(0,1,a,endpoint=False,dtype=np.float32)
            if d and a+d<=n: env[a:a+d]=np.linspace(1,.82,d,endpoint=False,dtype=np.float32)
            env[max(0,n-r):]=np.linspace(float(env[max(0,n-r)]),0,r,dtype=np.float32)
            y*=env*float(np.clip(energy,0,1))
        return y.astype(np.float32)
    def _radiate(self,x):
        if not len(x): return x.astype(np.float32)
        y=np.empty_like(x); px=py=0.0
        for i,v in enumerate(x):
            y[i]=.97*(py+float(v)-px); px=float(v); py=float(y[i])
        peak=np.max(np.abs(y))
        return (y/(peak+1e-9)*.8).astype(np.float32)
