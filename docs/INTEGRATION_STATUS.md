# ElevenX archive integration status

Branch: `test/elevenx-archive-integration`

## Scope and safety rules

This branch is a staging area for selectively integrating `Source/ElevenX.zip` into the existing Python application. The existing XTTS-v2 synthesis API remains the compatibility baseline; archive code must not silently replace it.

- Keep `synthesize()` and `POST /v1/synthesize` compatible.
- Keep model loading lazy and inference serialized. A busy inference slot must continue to return HTTP 503 with `Retry-After`.
- Keep synthesis local-first. Do not add telemetry, remote model calls, or automatic update checks.
- Require explicit consent for voice cloning.
- Treat SpeechBrain ECAPA embeddings as optional analysis only unless an explicitly validated conditioning path is implemented.
- Do not enable HarmoniaCodec training/inference as a production codec without trained weights, reproducible quality tests, and documented model provenance.
- Keep TensorRT, Rust, TypeScript rendering, and the C++ spatial bridge optional and independently buildable; none may become a required runtime dependency without a separate design decision.

## Current baseline

The repository already contains a Python/FastAPI service, lazy XTTS-v2 engine, optional speaker-analysis code, request validation, upload-size limits, a nonblocking synthesis lock, background-thread inference, WAV response handling, and API tests. These are the baseline to preserve and extend, not rewrite wholesale.

The archive inventory indicates Python server/engine/spatial/watermark/TensorRT/training files, a C++ spatial bridge, Rust sources, and a TypeScript renderer. An inventory alone is not evidence that these components compile, interoperate, or meet quality/security requirements.

## Integration gate

Before importing archive source into runtime paths:

1. Extract the archive and verify its ZIP integrity.
2. Inspect every source file and dependency/license declaration; identify duplicate implementations and entry points.
3. Compare the archive server/engine with `src/elevenx/api.py` and `src/elevenx/engine.py`; port behavior selectively rather than replacing the existing API wholesale.
4. Add regression tests before changing request, consent, concurrency, temporary-file cleanup, or audio-output behavior.
5. Keep experimental subsystems disabled by default and document their prerequisites.
6. Run the Python test suite and any optional C++/TypeScript/Rust checks that the available environment supports. Record unavailable checks honestly.

## Work completed on the staging branch

- Added `SECURITY.md` describing the current loopback-only deployment boundary, consent limitations, sensitive audio handling, and prerequisites for any remote deployment.
- Added `DEPENDENCIES.md` documenting the declared Python runtime dependencies and the audit gates for optional archive subsystems.
- Added model-free engine tests covering invalid configured sample rates and preference for the loaded model's declared output sample rate.

These changes document and test existing behavior; they do not activate new inference subsystems.

## Remaining archive integration blocker

The repository connector can read text files but does not provide the ZIP bytes as an extractable local file in this workflow. The archive source itself has therefore not been audited or merged into runtime code. No test-suite or build result is claimed here.

Next implementation step: make `Source/ElevenX.zip` available as a chat attachment or in an extractable build environment, audit the source and licenses, then commit small, test-backed integration changes to this branch.
