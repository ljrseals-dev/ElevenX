# ElevenX

![ElevenX product poster](docs/elevenx-poster.svg)

**Local-first voice synthesis prototype** built around Coqui XTTS-v2, with optional SpeechBrain ECAPA-TDNN speaker embeddings and a small FastAPI interface.

> I don't compete. I replace.

That line is branding, not a measured performance claim. ElevenX is an early implementation milestone—not a validated production voice platform.

[Repository](https://github.com/ljrseals-dev/ElevenX) · [Feature branch](https://github.com/ljrseals-dev/ElevenX/tree/feat/elevenx-core-tts) · [Voice Studio implementation PR](https://github.com/ljrseals-dev/ElevenX/pull/3)

## Contents

- [What works in this milestone](#what-works-in-this-milestone)
- [Architecture](#architecture)
- [Requirements](#requirements)
- [Installation](#installation)
- [Configuration](#configuration)
- [Python quick start](#python-quick-start)
- [Local HTTP API](#local-http-api)
- [Speaker embeddings](#speaker-embeddings)
- [Tests and validation](#tests-and-validation)
- [Privacy, security, and consent](#privacy-security-and-consent)
- [Known limitations](#known-limitations)
- [Roadmap](#roadmap)
- [Repository layout](#repository-layout)

## What works in this milestone

- Lazy-loaded text-to-speech synthesis using the configured XTTS-v2 model and a user-provided reference recording.
- WAV export as mono 16-bit PCM at the configured 24 kHz output rate.
- Five named sampling presets: `neutral`, `confident`, `empathetic`, `authoritative`, and `aggressive`.
- Optional ECAPA-TDNN speaker embeddings for analysis/comparison.
- FastAPI endpoints for a service health check and multipart WAV synthesis.
- YAML configuration for model, audio, synthesis, codec, and basic request-limit settings.
- Product poster artwork in [`docs/elevenx-poster.svg`](docs/elevenx-poster.svg).

### HarmoniaCodec status

[HarmoniaCodec](https://github.com/ljrseals-dev/HarmoniaCodec) is **not in the active synthesis path**. Its current project documentation indicates that trained weights are not shipped and that random-initialization round trips do not provide an acceptable quality basis. ElevenX keeps the codec disabled rather than silently degrading output. Revisit integration only after trained weights, reproducible evaluation data, and explicit quality gates are available.

## Architecture

```text
Caller text + permitted reference recording
                  |
                  v
        ElevenXEngine (engine.py)
          |                 |
          |                 +--> EmotionPreset validation
          |
          +--> Lazy XTTS-v2 model loading
          |
          +--> SynthesisResult (mono float waveform)
                         |
                         v
                 Local WAV export

Reference recording --> validation/resampling --> optional ECAPA embedding

FastAPI client --> /v1/synthesize --> temporary input/output files --> WAV response
```

| Module | Responsibility |
|---|---|
| `src/elevenx/engine.py` | Reads configuration, loads XTTS on first synthesis, validates requests, returns a `SynthesisResult`, writes WAV files. |
| `src/elevenx/emotions.py` | Defines the supported preset names and validates sampling parameters. |
| `src/elevenx/speakers.py` | Validates reference audio and lazily loads SpeechBrain ECAPA for speaker embeddings. |
| `src/elevenx/api.py` | Exposes the local FastAPI health and synthesis endpoints. |
| `config.yaml` | Defaults for model, sample rate, reference duration, presets, codec state, and text length. |
| `tests/test_emotions.py` | Model-free tests for preset values and validation boundaries. |

## Requirements

- Python **3.10 or 3.11** (the package metadata excludes Python 3.12+).
- A compatible PyTorch and torchaudio installation for your operating system and accelerator.
- Enough memory and disk space for the XTTS-v2 model and optional SpeechBrain model.
- A clean reference recording, 3–30 seconds by default, in a supported audio format.

Model downloads and accelerator compatibility depend on the upstream packages and your environment. First-time model initialization can be slow. CPU inference is supported by the code path but may be much slower than GPU inference.

## Installation

Create and activate a virtual environment:

```bash
python -m venv .venv
# Windows PowerShell:
.venv\\Scripts\\Activate.ps1
# macOS/Linux:
source .venv/bin/activate
```

Upgrade packaging tools and install ElevenX:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
```

If pip cannot resolve a compatible PyTorch/torchaudio pair, install the matching build for your platform from the official [PyTorch installation selector](https://pytorch.org/get-started/locally/) first, then retry the ElevenX installation.

The first model use may download model files. For an offline or restricted-network deployment, pre-cache the required weights, inspect their provenance and licenses, and verify the application in the intended disconnected environment. Installing dependencies alone does not prove offline operation.

## Configuration

The default file is `config.yaml`. The engine reads it when `ElevenXEngine` is created.

| Setting | Default | Meaning |
|---|---|---|
| `audio.sample_rate` | `24000` | Output sample rate used by the result and WAV writer. |
| `audio.output_channels` | `1` | Intended mono output; the engine currently flattens generated audio to a one-dimensional waveform. |
| `audio.reference_min_seconds` | `3.0` | Minimum reference duration. |
| `audio.reference_max_seconds` | `30.0` | Maximum reference duration. |
| `models.xtts_model` | XTTS-v2 model identifier | Model name passed to Coqui TTS. |
| `models.device` | `auto` | Chooses CUDA when available, otherwise CPU; explicit device values are also accepted by the code. |
| `models.load_speaker_encoder` | `true` | Enables the optional ECAPA embedding feature. |
| `models.cache_dir` | `./.models` | Local directory used for the SpeechBrain model cache. |
| `synthesis.default_emotion` | `neutral` | Preset used when no emotion is supplied. |
| `codec.enabled` | `false` | Codec stays disabled in this milestone. |
| `security.max_text_characters` | `5000` | Engine-level maximum text length. |

Presets are sampling configurations, not trained emotion labels. You can adjust their `temperature`, `top_k`, `top_p`, and `speed` values in YAML; invalid values are rejected by `EmotionPreset`.

The API currently uses `config.yaml` from the process working directory unless `ELEVENX_CONFIG` points to another configuration file.

## Python quick start

Place a reference recording you are authorized to use at `reference.wav`, then run:

```python
from elevenx import ElevenXEngine

engine = ElevenXEngine("config.yaml")
result = engine.synthesize(
    text="Welcome to ElevenX.",
    reference_audio="reference.wav",
    language="en",
    emotion="neutral",
)
saved_path = engine.save_wav(result, "output/elevenx.wav")

print(f"Saved: {saved_path}")
print(f"Sample rate: {result.sample_rate} Hz")
print(f"Duration: {result.duration_seconds:.2f} seconds")
```

The returned `SynthesisResult` contains:

- `audio`: mono NumPy waveform in float32 format.
- `sample_rate`: configured output rate.
- `emotion`: selected preset name.
- `duration_seconds`: generated waveform duration.

The `save_wav` method creates the output directory if needed and writes 16-bit PCM WAV. The first call to `synthesize` lazily initializes XTTS and may download/load model weights.

## Local HTTP API

Start the development server from the repository root:

```bash
uvicorn elevenx.api:app --host 127.0.0.1 --port 8000
```

Open **http://127.0.0.1:8000/** in your browser to use the ElevenX Voice Studio interface. Enter a script, choose a delivery preset and language, upload a permitted voice reference, accept the usage confirmation, then generate, preview, and download the WAV. The web interface is served by the same local FastAPI process; it does not require a separate frontend build step.

### Health check

```bash
curl http://127.0.0.1:8000/health
```

This endpoint reports that the API process responds. It **does not** prove that model weights are downloaded, the model loads successfully, or synthesis is ready.

### Generate a WAV

The endpoint expects multipart form data. Example:

```bash
curl -X POST http://127.0.0.1:8000/v1/synthesize \
  -F 'text=Welcome to ElevenX.' \
  -F 'language=en' \
  -F 'emotion=neutral' \
  -F 'reference_audio=@reference.wav' \
  --output elevenx.wav
```

Supported reference filename extensions are `.wav`, `.flac`, `.ogg`, `.aiff`, and `.aif`. The uploaded reference file is limited to 25 MiB, and the text field is limited to 5,000 characters at the API layer. Audio duration is validated by the engine.

Common responses include:

- `200`: generated WAV returned as `audio/wav`.
- `413`: uploaded reference is empty or too large.
- `415`: unsupported reference filename extension.
- `400`: synthesis validation or runtime error surfaced by the current handler.

The API does not currently provide authentication, authorization, request quotas, a job queue, concurrency control, or a streaming audio protocol. Keep it bound to loopback. Do not expose it to a LAN or the public internet without a security review and appropriate access controls.

## Speaker embeddings

The optional `engine.speaker_embedding("reference.wav")` method returns a normalized ECAPA-TDNN embedding when `models.load_speaker_encoder` is enabled and the SpeechBrain model can be loaded.

```python
from elevenx import ElevenXEngine

engine = ElevenXEngine("config.yaml")
embedding = engine.speaker_embedding("reference.wav")
print(embedding.shape)
```

An embedding is a speaker-analysis representation. It is **not** an XTTS conditioning latent, a proof of identity, or a guarantee that generated speech matches a person. Treat embeddings as sensitive biometric-like data; do not publish or retain them without a clear, lawful purpose and appropriate consent.

## Tests and validation

Run the model-free tests:

```bash
python -m pytest
```

The model-free suite covers emotion preset validation, the homepage and health endpoint, API request validation, upload limits, and a mocked synthesis response. It does not download model weights or validate real speech quality. GitHub Actions runs it on Python 3.10 and 3.11.

**Validation status:** these tests and real XTTS inference must be run in a compatible environment before calling this milestone verified. No passing test run, GPU benchmark, speaker-similarity score, or real-time latency result is claimed by this README.

## Privacy, security, and consent

- Use only reference recordings and voices you own or have explicit permission to use. Follow applicable laws, platform policies, and the consent of the person whose voice is represented.
- Do not use ElevenX to impersonate someone, mislead listeners about who is speaking, or create deceptive audio.
- Model weights may be downloaded by upstream libraries on first use. Review the model and dependency licenses and network behavior for your deployment.
- The local API creates temporary files for the uploaded reference and generated WAV, then schedules cleanup after the response. This is not a substitute for an operational privacy audit, disk-encryption policy, crash cleanup, or access controls.
- The API has no authentication or rate limiting. Keep it on loopback unless you add and test the necessary protections.
- Do not log uploaded audio, voice embeddings, or sensitive text in a production wrapper.
- This project does not advertise telemetry or remote-control functionality; still inspect dependencies and runtime network behavior before making strong privacy guarantees.

## Known limitations

- **Not real-time streaming:** synthesis returns a completed waveform. No incremental chunk delivery or sub-10-ms end-to-end latency is implemented or demonstrated.
- **Emotion is not guaranteed:** preset names only select sampling/speed parameters. They are not calibrated emotional controls and may not produce a consistent audible emotion.
- **Speaker similarity is unmeasured:** no 99.97% similarity claim has been demonstrated. ECAPA embeddings are separate analysis outputs, not XTTS conditioning.
- **Output bandwidth is limited:** 24 kHz sampling has a 12 kHz Nyquist frequency; this configuration cannot promise a 20 kHz upper band.
- **Spatial audio is not implemented:** no HRTF/binaural renderer is included in this milestone.
- **Codec is bypassed:** HarmoniaCodec is disabled until trained weights and repeatable quality tests are available.
- **API hardening is incomplete:** no auth, rate limiting, or concurrency management.
- **Build and model validation are pending:** installation and inference need to be exercised on target hardware and supported Python versions.

## Roadmap

The following are future tasks, not current capabilities:

1. Add CI for supported Python versions and run unit tests on each change.
2. Add deterministic engine/API tests with mocked model loaders, including request validation and cleanup paths.
3. Add a model-readiness status distinct from the process health check.
4. Add consent-aware voice asset handling, resource limits, and a documented security review.
5. Benchmark latency, memory, output quality, and speaker similarity with published evaluation procedures and reproducible datasets.
6. Implement genuine chunked streaming only if the model/runtime supports it and measurements justify the design.
7. Evaluate spatial/HRTF processing as a separate optional component.
8. Reconsider HarmoniaCodec integration only after trained weights, licensing review, and quality gates exist.

## Repository layout

```text
ElevenX/
├── config.yaml
├── requirements.txt
├── pyproject.toml
├── README.md
├── docs/
│   └── elevenx-poster.svg
├── src/
│   └── elevenx/
│       ├── __init__.py
│       ├── emotions.py
│       ├── engine.py
│       ├── speakers.py
│       └── api.py
└── tests/
    └── test_emotions.py
```

## License and model terms

Check the repository's license files and the terms of each upstream model and dependency before redistribution or commercial use. The ability to install a package does not automatically grant permission to use its model weights or generated voice output for every purpose.
