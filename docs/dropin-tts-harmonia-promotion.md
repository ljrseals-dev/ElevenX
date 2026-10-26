# Drop-in TTS + HarmoniaCodec promotion

## What landed

The self-contained numpy TTS engine from `dropin-tts-harmonia.zip` is now
vendored under `src/elevenx/tts/`.

Architecture (unchanged from the drop-in contract):

```
text -> frontend -> codec registry -> encode -> decode
     -> NoiseBed -> MultibandCompressor -> ProDither -> WAV
```

- `baseline` — deterministic Klatt-style formant synthesizer (always available).
- `harmonia` — reserved slot implementing the frozen `BaseCodec` interface.
  Until weights/runtime are installed it raises `NotImplementedError` and the
  engine falls back to `baseline` without ever mixing latents across codecs.

## Why this is a promotion, not a replacement

1. **XTTS remains the default synthesis path** in `ElevenXEngine` and the
   FastAPI `/v1/synthesize` endpoint.
2. **HarmoniaCodec is still gated** by `check_harmonia_readiness` in
   `elevenx.harmoniacodec`. The drop-in only provides the *slot* and a safe
   fallback; it does not claim trained quality.
3. The drop-in path is usable today for:
   - offline / no-GPU environments
   - CI that must not download multi-GB XTTS weights
   - smoke tests of the codec registry and post-processing chain

## How to use the drop-in path

```python
from elevenx.tts import TTSEngine, available_codecs

print(available_codecs())  # ['baseline', 'harmonia']

engine = TTSEngine(codec_name="baseline", sample_rate=22050)
audio = engine.synth("Welcome to ElevenX.")
# audio: float32 mono, peak-normalized
```

CLI:

```bash
python -m elevenx.tts.cli "hello world" out.wav
python -m elevenx.tts.cli "hello world" out.wav harmonia   # falls back with warning
```

## Completing the HarmoniaCodec application

To finish the real codec path (still required before any production enablement):

1. Implement `load` / `encode` / `decode` in `elevenx/tts/codec/harmonia.py`
   against a trained checkpoint from the HarmoniaCodec repository.
2. Produce a `CodecQualityReport` with `weights_sha256`, a non-empty
   `benchmark_id`, and `passed=True`.
3. Set `codec.enabled: true` and the matching `expected_weights_sha256` in
   `config.yaml` only after the readiness guard returns `ready=True`.
4. Keep XTTS as the fallback until end-to-end quality gates pass.

See also: `docs/harmoniacodec-integration.md`.
