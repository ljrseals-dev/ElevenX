Copyright © 2026 Lewis James Seals Jr. All rights reserved.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT.
# ElevenX

![ElevenX product poster](docs/elevenx-poster.svg)

**Local-first voice synthesis prototype** built around Coqui XTTS-v2, with optional SpeechBrain ECAPA speaker embeddings and a small FastAPI interface.

> I don't compete. I replace.

The slogan is branding, not a measured performance claim. This repository is a starter implementation, not a validated production voice platform.

## Current milestone

- Lazy-loaded XTTS-v2 synthesis using a reference recording.
- Optional ECAPA-TDNN embeddings as a separate analysis feature; SpeechBrain is not required for normal XTTS synthesis.
- Configurable presets named neutral, confident, empathetic, authoritative, and aggressive.
- Local WAV output and an optional HTTP endpoint.
- Synthesis requests are serialized: concurrent requests receive HTTP 503 with a `Retry-After` hint instead of piling up in an unbounded model-call queue.
- Blocking model inference runs in a worker thread so health checks and other API requests do not stall the event loop during synthesis.
- Output sample rate is taken from the loaded model when the model exposes it, rather than blindly assuming the configured rate.
- Product poster artwork included at [docs/elevenx-poster.svg](docs/elevenx-poster.svg).
- HarmoniaCodec is deliberately not in the active synthesis path. Its README states that no trained model ships and round-trip quality is poor on random initialization. Integrate it only after trained weights and quality gates exist.

## Repository layout

```text
ElevenX/
├── config.yaml
├── requirements.txt
├── pyproject.toml
├── README.md
├── docs/
│   └── elevenx-poster.svg
├── scripts/
│   └── smoke_xtts.py
├── src/
│   └── elevenx/
│       ├── __init__.py
│       ├── emotions.py
│       ├── engine.py
│       ├── speakers.py
│       └── api.py
└── tests/
    ├── test_emotions.py
    └── test_api.py
```

## Voice Studio

Run the local API, then open `http://127.0.0.1:8000/` to use the bundled browser interface. The UI supports script entry, language and delivery-style selection, an explicit voice-permission confirmation, reference audio upload, in-browser playback, and WAV download. Keep the service bound to loopback; this interface does not add authentication or make the API safe to expose publicly.

## Install

Use Python 3.10 or 3.11 in a clean virtual environment. Deep-learning package compatibility varies by OS/GPU; install a matching PyTorch build first if needed.

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
```

For optional ECAPA speaker embeddings, install the extra:

```bash
python -m pip install -e ".[speaker]"
```

Model weights may download on first use. After models are cached, synthesis runs locally. For offline deployment, pre-cache weights and audit model-loading paths before disconnecting the machine.

## Python usage

```python
from elevenx import ElevenXEngine

engine = ElevenXEngine("config.yaml")
result = engine.synthesize(
    text="Welcome to ElevenX.",
    reference_audio="reference.wav",
    language="en",
    emotion="neutral",
)
engine.save_wav(result, "output/elevenx.wav")
print(result.sample_rate, result.duration_seconds)
```

Use only voices you own or have permission to synthesize. The reference should be clean speech; duration is validated from 3 to 30 seconds.

## Local API

Preferred launch command (reads `config.yaml` and forces loopback unless remote access is explicitly opted into):

```bash
python -m elevenx.api
```

The default `security.allow_remote_api: false` is enforced at request time. The app rejects requests with a non-loopback client, non-loopback server bind, or non-loopback Host header, even if someone starts Uvicorn with a wildcard bind. Missing or malformed security configuration fails closed. Set `security.allow_remote_api: true` only as a deliberate deployment decision; this switch is **not** a substitute for authentication, TLS, quotas, and consent controls. The API has no authentication or rate limiting; do not expose it publicly without adding authentication, TLS, request quotas, and consent controls. Keep remote access disabled for normal local use.

POST multipart form data to `/v1/synthesize` with `text`, `language`, `emotion`, `consent`, and `reference_audio`. GET `/health` is a process check, not a model-readiness check. One synthesis request is accepted at a time; if one is running, a second request receives HTTP 503 and `Retry-After: 3`. The client should wait and retry rather than opening repeated concurrent requests.

For regression coverage of the local-only policy, run `python -m pytest tests/test_api_privacy.py tests/test_api.py`.

## Model-backed smoke test

The normal CI suite intentionally does not download large model weights or require a GPU. To test actual inference manually, use a reference recording you have permission to use and run this from the repository root after installing the project:

```bash
python scripts/smoke_xtts.py --reference /path/to/consented-reference.wav --output elevenx-smoke.wav
```

The reference must be 3–30 seconds. The script loads XTTS-v2, synthesizes a short phrase, saves a WAV, and validates that the decoded file is non-empty, finite, mono, has a positive duration, and matches the model's declared sample rate. Model weights may be downloaded on first run. After it passes, play the resulting WAV in a separate audio player; the script cannot certify your browser, audio driver, or playback device.

### If playback reports that the audio queue has too few elements

This release reduces one possible cause: concurrent requests competing for the same XTTS model/GPU and blocking the API event loop. It also uses the model's declared output sample rate where available. The app still synthesizes a complete WAV; it does **not** implement a real-time streaming playback queue. If the error continues while playing a downloaded WAV, test the WAV in another player and capture the complete error text plus whether it appears during synthesis or playback. That distinction is needed to identify a browser/audio-driver issue versus a synthesis-side issue.

## Limitations and roadmap

- This milestone returns a completed waveform; it does not provide true streaming. WebSockets alone cannot guarantee sub-10-ms synthesis or end-to-end latency.
- XTTS-v2 has no calibrated emotion control here. Presets change sampling and speed as style hints; audible results must be evaluated.
- ECAPA embeddings are for speaker comparison and are not XTTS conditioning latents.
- 24 kHz audio has a 12 kHz Nyquist limit; it cannot guarantee a 20 kHz upper band.
- No 99.97% similarity or sub-10-ms latency has been demonstrated.
- HarmoniaCodec integration, spatial HRTF processing, and streaming are later milestones. The codec must not be used until trained weights and quality tests are available.

## Tests and verification

Run `python -m pip install -e ".[test]"` then `python -m pytest`. The model-free suite covers health responses, overlapping-request rejection, unsupported and empty reference files, upload-size enforcement, successful WAV responses, temporary-file cleanup, synthesis-failure cleanup, emotion preset validation, configured sample-rate validation, and preference for the loaded model's declared output sample rate. It does not verify real XTTS inference, TTS quality, speaker similarity, GPU performance, or real-time latency. Use the opt-in smoke test above for model-backed validation; do not treat model-free tests as proof that the model runs on a particular machine.
