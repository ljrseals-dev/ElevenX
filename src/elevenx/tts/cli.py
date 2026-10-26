"""CLI: python -m elevenx.tts.cli \"hello world\" out.wav [codec]."""
import os,sys,wave
import numpy as np
from .engine import TTSEngine

def write_wav(path,audio,sr):
    path=os.fspath(path); parent=os.path.dirname(os.path.abspath(path)); os.makedirs(parent,exist_ok=True)
    pcm=np.clip(np.asarray(audio,dtype=np.float32)*32767.,-32768,32767).astype("<i2")
    with wave.open(path,"wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(int(sr)); w.writeframes(pcm.tobytes())

def main(argv=None):
    argv=sys.argv[1:] if argv is None else argv
    if len(argv)<2: print("usage: python -m elevenx.tts.cli <text> <out.wav> [codec]",file=sys.stderr); return 1
    text,out=argv[0],argv[1]; codec=argv[2] if len(argv)>2 else "baseline"; e=TTSEngine(codec,sample_rate=22050); a=e.synth(text); write_wav(out,a,e.sr); print(f"wrote {out} ({len(a)} samples, codec={e.codec.config.name})"); return 0
if __name__=="__main__":sys.exit(main())
