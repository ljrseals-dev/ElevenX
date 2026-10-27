"""Unit tests for the evidence-first vintage compressor logging harness."""
import json

import pytest

from elevenx.vintage_harness import (
    EvidenceError,
    JsonlWriter,
    REQUIRED_COMPONENTS,
    composite_error,
    config_sha256,
    create_run_manifest,
    reject_holdout_leakage,
    validate_metrics,
    validate_reference,
)

GOOD_REF = {"model": "Reference Unit A", "revision_or_serial": "rev-A / serial-redacted", "source_url": "https://example.org/manual.pdf", "source_locator": "p. 12, Fig. 4", "measurement_conditions": {"sample_rate_hz": 48000, "channels": 2}, "raw_capture_sha256": "a" * 64, "calibration_notes": "Fixture only; not a real hardware claim", "uncertainty": "repeatability not established in fixture"}
GOOD_METRICS = {name: 0.1 for name in REQUIRED_COMPONENTS}
GOOD_WEIGHTS = {"transfer_curve": 0.2, "envelope": 0.2, "frequency_response": 0.15, "harmonics": 0.15, "intermodulation": 0.1, "transients": 0.1, "stereo_behavior": 0.1}


def test_config_hash_is_key_order_independent():
    assert config_sha256({"a": 1, "b": 2}) == config_sha256({"b": 2, "a": 1})


def test_reference_requires_source_and_capture_provenance():
    validate_reference(GOOD_REF)
    with pytest.raises(EvidenceError, match="missing reference evidence"):
        validate_reference({"model": "Unknown"})
    with pytest.raises(EvidenceError, match="SHA-256"):
        validate_reference({**GOOD_REF, "raw_capture_sha256": "not-a-hash"})


def test_missing_metric_is_not_silently_zeroed():
    incomplete = dict(GOOD_METRICS)
    del incomplete["harmonics"]
    with pytest.raises(EvidenceError, match="missing component metrics"):
        validate_metrics(incomplete)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_metrics_fail_closed(bad):
    with pytest.raises(EvidenceError, match="finite numeric"):
        validate_metrics({**GOOD_METRICS, "harmonics": bad})


def test_composite_error_requires_complete_weights_and_normalization():
    assert composite_error(GOOD_METRICS, GOOD_WEIGHTS) == pytest.approx(0.1)
    with pytest.raises(EvidenceError, match="sum to 1"):
        composite_error(GOOD_METRICS, {**GOOD_WEIGHTS, "stereo_behavior": 0.2})


def test_holdout_overlap_is_rejected():
    with pytest.raises(EvidenceError, match="leakage"):
        reject_holdout_leakage(["cal-1", "shared"], ["shared", "test-1"])
    reject_holdout_leakage(["cal-1"], ["test-1"])


def test_manifest_binds_config_reference_and_distinct_splits():
    config = {"execution": {"mode": "dry_run", "max_cost_usd": 0.0}}
    manifest = create_run_manifest(config=config, reference=GOOD_REF, input_sha256="b" * 64, objective_name="vintage_reference_composite_v1", objective_version="0.1.0", calibration_id="cal-01", holdout_id="holdout-01")
    assert manifest["config_sha256"] == config_sha256(config)
    assert manifest["evidence_status"] == "provenance_recorded_not_hardware_validated"
    with pytest.raises(EvidenceError, match="distinct"):
        create_run_manifest(config=config, reference=GOOD_REF, input_sha256="b" * 64, objective_name="x", objective_version="1", calibration_id="same", holdout_id="same")


def test_jsonl_writer_redacts_secrets_and_writes_utc(tmp_path):
    path = tmp_path / "run.log.jsonl"
    JsonlWriter(path).write({"event": "test", "api_token": "dont-write-this"})
    content = path.read_text(encoding="utf-8")
    row = json.loads(content.strip())
    assert row["api_token"] == "[REDACTED]"
    assert row["timestamp_utc"].endswith("+00:00")
    assert "dont-write-this" not in content
