# ElevenX

Voice synthesis service scaffold for XTTS-v2, with optional spatial rendering and an explicit codec bypass.

## Status

This repository is an integration scaffold, not a validated production voice-cloning system. XTTS-v2 and SpeechBrain require model downloads and compatible PyTorch/audio dependencies. HarmoniaCodec must match the real repository API and use trained weights before enabling codec processing.

The poster's **99.97% similarity** and **sub-10 ms end-to-end latency** are aspirational claims, not measured guarantees. XTTS-v2's streaming inference still has model-compute and first-token latency; a WebSocket cannot make synthesis itself sub-10 ms.

## Setup

Use Python 3.10 or 3.11 in a virtual environment. Install the CPU or CUDA build of PyTorch appropriate for your machine, then:

```bash
python -m pip install -r requirements.txt
cp config.example.yaml config.yaml
```

Model weights may be downloaded on first use by the selected upstream libraries. Review their licenses and terms before deployment.

## Run

```bash
uvicorn elevenx.server:app --host 127.0.0.1 --port 8000
```

The server starts without loading large neural models. The first synthesis request initializes the engine. Configure `ELEVENX_REFERENCE_ROOT` to a directory of reference recordings; clients submit a filename relative to that directory, not an arbitrary filesystem path.

- `GET /health`: process health, without claiming model readiness.
- `POST /synthesize`: generate a WAV response.
- `WS /ws/tts`: send one JSON request, receive a JSON stream header, PCM16 binary chunks, and a final JSON event.

## Safety and privacy

Only clone voices when you have authorization from the speaker. Keep reference recordings local and access-controlled. This service does not provide identity verification or consent verification. Do not expose the development server publicly without authentication, rate limiting, TLS, and resource limits.

## Tests

```bash
python -m pytest
```

The tests exercise configuration, path validation, audio encoding, and API behavior without downloading neural models. They do not validate synthesis quality or GPU latency.
