# ElevenX Python 3.8 Compatibility Contract

**Status:** migration contract / implementation gate  
**Target:** CPython 3.8.x  
**Applies to:** ElevenX Python package, API, Voice Studio backend, tests, and supported install instructions.

## Goal

Make ElevenX installable and testable on Python 3.8 without silently removing the existing XTTS synthesis workflow, consent checks, bounded uploads, local-only API defaults, serialized inference, or temporary-file cleanup.

The owner reports having a working Python 3.8 environment. Treat that as a valuable reproduction target, but do not infer which dependency versions or patches make it work. Record those exact versions before changing package metadata.

## Non-negotiable acceptance criteria

1. **Interpreter truthfulness**
   - Every file advertised as Python 3.8 compatible must parse and import under CPython 3.8.
   - Run `python3.8 -m compileall src tests scripts` and the model-free test suite under Python 3.8.
   - Do not advertise a supported version range until dependency installation and CI checks pass.

2. **Dependency lock and provenance**
   - Capture the working environment with `python3.8 -m pip freeze` and record OS/architecture, Python patch version, CPU/GPU, and PyTorch/CUDA build.
   - Pin or bound dependencies to releases that actually publish compatible Python 3.8 wheels.
   - Install PyTorch and torchaudio as a matched pair appropriate to the target platform.
   - Verify the selected Coqui TTS distribution's Requires-Python metadata and XTTS-v2 availability. The current `TTS>=0.22.0` declaration is a known compatibility risk; do not assume it supports Python 3.8. If the working setup uses a fork, wheel, local patch, or older release, document the exact source, version, license, and XTTS model identifier.
   - Do not weaken dependency constraints merely to make the resolver finish.

3. **Source compatibility**
   - Keep `from __future__ import annotations` where annotations use postponed evaluation.
   - Avoid runtime-only APIs introduced after Python 3.8 unless a tested compatibility shim is added.
   - Review Path, typing, dataclass, FastAPI/Starlette, NumPy, SciPy, PyYAML, SoundFile, and multipart APIs against the chosen Python 3.8 dependency set.
   - Do not add syntax or standard-library APIs from Python 3.9+ to production code.

4. **Behavior preservation**
   - XTTS remains lazily loaded.
   - The API remains bound to 127.0.0.1 by default.
   - Reference-audio permission confirmation remains mandatory.
   - Upload-size and duration limits remain enforced.
   - Synthesis stays serialized; overlapping requests return HTTP 503 with Retry-After.
   - Blocking inference remains off the async event loop.
   - Model-declared sample rate takes precedence when available.
   - HarmoniaCodec remains out of the active synthesis path until trained weights and quality gates exist.

5. **Verification**
   - Run the complete model-free test suite on Python 3.8 without downloading model weights or requiring a GPU.
   - Run an opt-in XTTS smoke test using a reference recording the operator has permission to use.
   - Test a clean installation in a fresh Python 3.8 virtual environment on the supported OS/platform.
   - Add a Python 3.8 CI job if the repository's CI runner and selected dependency set support it.
   - Report exact commands, environment, pass/fail results, and skipped tests. Never claim tests passed if they were not run.

## Required deliverables

- A pinned or constrained Python 3.8 dependency set with documented provenance.
- Python 3.8 CI and model-free regression tests.
- Updated pyproject.toml, requirements.txt, and README installation instructions that agree with one another.
- A short compatibility note listing unsupported platforms or optional features.
- A clean-install and test report.

## Current blocker

The checked-in pyproject.toml currently declares `requires-python = ">=3.10,<3.12"` and depends on packages including `numpy>=1.26,<3`, `scipy>=1.11`, `torch>=2.2`, `torchaudio>=2.2`, and `TTS>=0.22.0`. Those constraints are not evidence of a working Python 3.8 environment. In particular, the TTS distribution and its transitive dependencies must be verified before the project metadata is changed.

**Do not mark this contract complete or claim Python 3.8 support until the clean install, import/compile checks, model-free tests, and XTTS smoke test have been recorded.**
