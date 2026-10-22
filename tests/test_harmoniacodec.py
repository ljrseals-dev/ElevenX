from pathlib import Path

from elevenx.harmoniacodec import CodecQualityReport, check_harmonia_readiness


def test_codec_is_not_ready_when_disabled_even_with_prerequisites(tmp_path: Path) -> None:
    weights = tmp_path / "trained.bin"
    weights.write_bytes(b"fixture")
    report = CodecQualityReport(
        weights_sha256="abc123",
        benchmark_id="offline-roundtrip-v1",
        passed=True,
    )
    result = check_harmonia_readiness(
        enabled=False,
        weights_path=weights,
        report=report,
        expected_weights_sha256="abc123",
    )
    assert not result.ready
    assert "HarmoniaCodec is disabled by configuration" in result.reasons


def test_codec_requires_weights_report_and_matching_hash(tmp_path: Path) -> None:
    missing = check_harmonia_readiness(
        enabled=True,
        weights_path=tmp_path / "missing.bin",
        report=None,
        expected_weights_sha256="abc123",
    )
    assert not missing.ready
    assert "trained weights file is missing" in missing.reasons
    assert "quality report is missing" in missing.reasons

    weights = tmp_path / "trained.bin"
    weights.write_bytes(b"fixture")
    mismatch = check_harmonia_readiness(
        enabled=True,
        weights_path=weights,
        report=CodecQualityReport("wrong-hash", "offline-roundtrip-v1", True),
        expected_weights_sha256="abc123",
    )
    assert not mismatch.ready
    assert "quality report does not match the configured weights SHA-256" in mismatch.reasons


def test_codec_readiness_requires_passing_named_benchmark(tmp_path: Path) -> None:
    weights = tmp_path / "trained.bin"
    weights.write_bytes(b"fixture")
    result = check_harmonia_readiness(
        enabled=True,
        weights_path=weights,
        report=CodecQualityReport("abc123", "", False),
        expected_weights_sha256="abc123",
    )
    assert not result.ready
    assert "quality report did not pass" in result.reasons
    assert "quality report has no benchmark identifier" in result.reasons


def test_codec_ready_only_when_all_declared_prerequisites_pass(tmp_path: Path) -> None:
    weights = tmp_path / "trained.bin"
    weights.write_bytes(b"fixture")
    result = check_harmonia_readiness(
        enabled=True,
        weights_path=weights,
        report=CodecQualityReport("abc123", "offline-roundtrip-v1", True),
        expected_weights_sha256="abc123",
    )
    assert result.ready
    assert result.reasons == ()
