## Optional drop-in TTS (numpy baseline + HarmoniaCodec slot)

ElevenX vendors a zero-dependency formant TTS under `elevenx.tts` for offline
and CI use. It does **not** replace XTTS-v2.

```python
from elevenx.tts import TTSEngine
audio = TTSEngine().synth("Welcome to ElevenX.")  # float32 mono @ 22.05 kHz
```

```bash
python -m elevenx.tts.cli "hello world" out.wav
```

The `harmonia` codec name is registered. Until trained weights are installed
the engine emits one warning and renders with the baseline codec. See
`docs/dropin-tts-harmonia-promotion.md` and `docs/harmoniacodec-integration.md`.
