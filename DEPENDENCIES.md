# Dependency inventory

This inventory reflects the current Python project configuration, not a license audit of every transitive dependency. Check installed package versions and licenses before redistributing a bundled build.

## Current Python runtime

The project declares Python `>=3.10,<3.12` in `pyproject.toml`.

| Dependency | Purpose | Notes |
| --- | --- | --- |
| NumPy | Waveform arrays and validation | Include required license notices. |
| PyYAML | Read local configuration | Configuration is loaded with `yaml.safe_load`. |
| soundfile | WAV output and audio handling | Native libsndfile availability varies by platform. |
| SciPy | Audio utilities/resampling support | Verify wheel/platform compatibility. |
| PyTorch / torchaudio | XTTS inference and audio tensor support | Large platform-specific packages; choose compatible builds. |
| Coqui TTS | XTTS-v2 model API | Review package license and XTTS-v2 model terms separately. |
| SpeechBrain | Optional ECAPA speaker embeddings | Embeddings are analysis-only; they are not XTTS conditioning latents in the current architecture. |
| FastAPI / Uvicorn | Local HTTP API and server | Bind to loopback for local use. |
| python-multipart | Parse multipart form uploads | Required by the synthesis endpoint. |
| Pytest / HTTPX | Test extras | Declared under the `test` optional dependency group. |

## Archive components not enabled as runtime requirements

The source archive inventory references a TensorRT helper, HarmoniaCodec training code, Rust sources, a TypeScript renderer, and a C++ spatial bridge. These are not required runtime dependencies in the current `pyproject.toml`.

They must not be silently activated as mandatory dependencies. Before enabling any component, audit its source, package manifests, licenses, platform requirements, model/weight provenance, and tests. HarmoniaCodec remains experimental until trained weights and reproducible quality gates are available.

## License verification checklist

- Record direct dependency versions used for each release.
- Review direct and transitive licenses and required notices.
- Verify XTTS-v2 model and downloaded weights terms independently of the Python package license.
- Include notices for bundled native libraries and generated assets where required.
- Re-run this review when dependency versions or archive components change.
