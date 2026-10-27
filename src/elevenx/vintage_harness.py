"""Evidence-first logging primitives for vintage compressor experiments.

Local artifacts only: no network access, worker launch, audio capture, or training.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import re
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

SCHEMA_VERSION = 1
REQUIRED_REFERENCE_FIELDS = ("model", "revision_or_serial", "source_url", "source_locator", "measurement_conditions", "raw_capture_sha256", "calibration_notes", "uncertainty")
REQUIRED_COMPONENTS = ("transfer_curve", "envelope", "frequency_response", "harmonics", "intermodulation", "transients", "stereo_behavior")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
SECRET_KEY_RE = re.compile(r"(password|passwd|token|secret|api[_-]?key|private[_-]?key|authorization)", re.I)


class EvidenceError(ValueError):
    """Raised when evidence is incomplete or invalid for scoring."""


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def config_sha256(config: Mapping[str, Any]) -> str:
    return sha256_bytes(canonical_json(config).encode("utf-8"))


def _scrub(value: Any, key: str = "") -> Any:
    if SECRET_KEY_RE.search(key):
        return "[REDACTED]"
    if isinstance(value, Mapping):
        return {str(k): _scrub(v, str(k)) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_scrub(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise EvidenceError("non-finite numbers are forbidden in evidence artifacts")
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise EvidenceError("unsupported value type in artifact: " + type(value).__name__)


class JsonlWriter:
    """Append-only JSONL writer with UTC timestamps, redaction, and flush."""
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, event: Mapping[str, Any]) -> None:
        safe = _scrub(dict(event))
        safe.setdefault("timestamp_utc", datetime.now(timezone.utc).isoformat())
        with self.path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(canonical_json(safe) + "\n")
            stream.flush()


def git_commit(cwd: str | Path | None = None) -> str | None:
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(cwd) if cwd else None, check=True, capture_output=True, text=True, timeout=2)
        value = result.stdout.strip()
        return value if re.fullmatch(r"[0-9a-f]{40,64}", value) else None
    except (OSError, subprocess.SubprocessError):
        return None


def validate_reference(reference: Mapping[str, Any]) -> None:
    missing = [key for key in REQUIRED_REFERENCE_FIELDS if not reference.get(key)]
    if missing:
        raise EvidenceError("missing reference evidence fields: " + ", ".join(missing))
    if not SHA256_RE.fullmatch(str(reference["raw_capture_sha256"]).lower()):
        raise EvidenceError("raw_capture_sha256 must be a 64-character SHA-256 hex digest")
    if not str(reference["source_url"]).startswith(("https://", "http://", "file://")):
        raise EvidenceError("source_url must be an explicit source URI")
    if not isinstance(reference["measurement_conditions"], (str, Mapping)):
        raise EvidenceError("measurement_conditions must be text or a mapping")
    if not isinstance(reference["uncertainty"], (str, int, float, Mapping)):
        raise EvidenceError("uncertainty must describe measurement uncertainty or repeatability")


def validate_metrics(metrics: Mapping[str, Any], required: Iterable[str] = REQUIRED_COMPONENTS) -> None:
    missing = [name for name in required if name not in metrics or metrics[name] is None]
    if missing:
        raise EvidenceError("missing component metrics: " + ", ".join(missing))
    for name in required:
        value = metrics[name]
        values = value.values() if isinstance(value, Mapping) else (value,)
        for item in values:
            if isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(float(item)):
                raise EvidenceError("metric " + name + " must contain finite numeric values")


def composite_error(metrics: Mapping[str, Any], weights: Mapping[str, float]) -> float:
    """Weighted error primitive for already-normalized metrics; not a perceptual validation."""
    validate_metrics(metrics)
    if set(weights) != set(REQUIRED_COMPONENTS):
        raise EvidenceError("weights must name exactly the required component metrics")
    vals = [float(weights[name]) for name in REQUIRED_COMPONENTS]
    if any(not math.isfinite(v) or v < 0 for v in vals):
        raise EvidenceError("weights must be finite and non-negative")
    if not math.isclose(sum(vals), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise EvidenceError("weights must sum to 1.0")
    if any(isinstance(metrics[name], Mapping) for name in REQUIRED_COMPONENTS):
        raise EvidenceError("composite metric values must be normalized scalars")
    return sum(float(metrics[name]) * float(weights[name]) for name in REQUIRED_COMPONENTS)


def create_run_manifest(*, config: Mapping[str, Any], reference: Mapping[str, Any], input_sha256: str, objective_name: str, objective_version: str, calibration_id: str, holdout_id: str, cwd: str | Path | None = None) -> dict[str, Any]:
    validate_reference(reference)
    if not SHA256_RE.fullmatch(str(input_sha256).lower()):
        raise EvidenceError("input_sha256 must be a 64-character SHA-256 hex digest")
    if not calibration_id or not holdout_id or calibration_id == holdout_id:
        raise EvidenceError("calibration_id and holdout_id must be present and distinct")
    execution = config.get("execution", {})
    if execution.get("mode", "dry_run") != "dry_run":
        if execution.get("allow_accelerator_launch") is not True:
            raise EvidenceError("accelerator execution requires explicit allow_accelerator_launch=true")
        ceiling = execution.get("max_cost_usd")
        if not isinstance(ceiling, (int, float)) or not math.isfinite(float(ceiling)) or ceiling <= 0:
            raise EvidenceError("non-dry-run execution requires an explicit positive cost ceiling")
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": str(uuid.uuid4()),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(cwd),
        "config_sha256": config_sha256(config),
        "input_sha256": str(input_sha256).lower(),
        "reference": _scrub(dict(reference)),
        "objective": {"name": objective_name, "version": objective_version},
        "split": {"calibration_id": calibration_id, "holdout_id": holdout_id},
        "environment": {"python": sys.version.split()[0], "platform": platform.platform()},
        "execution": {
            "mode": execution.get("mode", "dry_run"),
            "allow_accelerator_launch": execution.get("allow_accelerator_launch", False),
            "max_cost_usd": execution.get("max_cost_usd", 0.0),
            "max_nodes": execution.get("max_nodes", 1),
            "max_combinations": execution.get("max_combinations", 1000),
        },
        "evidence_status": "provenance_recorded_not_hardware_validated",
    }


def write_manifest(path: str | Path, manifest: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(canonical_json(_scrub(dict(manifest))) + "\n", encoding="utf-8")
    os.replace(temporary, target)


def reject_holdout_leakage(calibration_ids: Iterable[str], holdout_ids: Iterable[str]) -> None:
    overlap = set(calibration_ids) & set(holdout_ids)
    if overlap:
        raise EvidenceError("calibration/holdout leakage detected: " + ", ".join(sorted(overlap)))


if __name__ == "__main__":
    # Run the focused evidence-contract tests and propagate pytest's exit status.
    import pytest

    tests = Path(__file__).resolve().parents[2] / "tests" / "test_vintage_harness.py"
    raise SystemExit(pytest.main(["-q", str(tests)]))
