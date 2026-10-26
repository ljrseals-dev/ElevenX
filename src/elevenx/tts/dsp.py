"""Small numpy-only post-processing chain for synthesized speech."""
import numpy as np

class NoiseBed:
    def __init__(self, level=.002, color=.5, seed=24680): self.level=float(level); self.color=float(color); self.rng=np.random.default_rng(seed)
    def process(self,x,sr):
        x=np.asarray(x,dtype=np.float32)
        if not len(x) or self.level<=0:return x.copy()
        n=self.rng.standard_normal(len(x)).astype(np.float32); alpha=float(np.clip(1-self.color*.9,.01,.99)); y=np.empty_like(n); p=0.
        for i,v in enumerate(n): p=alpha*float(v)+(1-alpha)*p; y[i]=p
        return (x+self.level*y).astype(np.float32)

class MultibandCompressor:
    def __init__(self,sr,bands=4): self.sr=int(sr); self.bands=int(bands); self.thresholds_db=(-18.,-15.,-12.,-9.); self.ratios=(2.,2.5,3.,3.5)
    def process(self,x):
        y=np.asarray(x,dtype=np.float32).copy()
        for db,ratio in zip(self.thresholds_db,self.ratios):
            th=10**(db/20); mag=np.abs(y); over=mag>th
            if np.any(over): y[over]=np.sign(y[over])*(th+(mag[over]-th)/ratio)
        return y

class ProDither:
    def __init__(self,bit_depth=24,seed=97531): self.lsb=1./(1<<int(bit_depth)); self.rng=np.random.default_rng(seed)
    def process(self,x):
        x=np.asarray(x,dtype=np.float32)
        if not len(x):return x.copy()
        out=np.empty_like(x); err=0.
        for i,v in enumerate(x):
            d=(self.rng.random()-self.rng.random())*self.lsb; shaped=float(v)+.5*err+d; q=np.round(shaped/self.lsb)*self.lsb; err=shaped-q; out[i]=q
        return out
