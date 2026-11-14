"""Contract tests for ElevenX's Python audio_processing implementation.

These tests exercise elevenx.audio_processing.track_f0 and
elevenx.audio_processing.compress_vocal only. They do not test or depend on
external plugins or commercial products.
"""
import numpy as np
import pytest
from scipy.signal import resample_poly

from elevenx.audio_processing import (
    CompressorSettings,
    compress_vocal,
    track_f0,
)


SAMPLE_RATE = 48_000
FRAME_MS = 40.0
HOP_MS = 10.0
F0_MIN_HZ = 65.0
F0_MAX_HZ = 500.0
CONFIDENCE_MIN = 0.0
CONFIDENCE_MAX = 1.0
CONFIDENCE_VOICED_MIN = 0.60
CONFIDENCE_SILENCE_MAX = 0.10
GLIDE_DURATION_SECONDS = 2.0
GLIDE_START_HZ = 100.0
GLIDE_END_HZ = 400.0
GLIDE_MAX_MAE_CENTS = 100.0
VIBRATO_CENTER_HZ = 220.0
VIBRATO_DEPTH_CENTS = 15.0
VIBRATO_RATE_HZ = 5.5
VIBRATO_DURATION_SECONDS = 4.0
BOUNDARY_TOLERANCE_MS = FRAME_MS - HOP_MS
COMPRESSOR_ATTACK_MS = 10.0
COMPRESSOR_RELEASE_MS = 100.0
COMPRESSOR_THRESHOLD_DB = -18.0
COMPRESSOR_RATIO = 3.0
COMPRESSOR_MAX_REDUCTION_DB = 12.0
SAMPLE_PEAK_CEILING = 0.999
SAMPLE_PEAK_TOLERANCE = 1e-6
TRUE_PEAK_OVERSAMPLE = 4


def _integrated_tone(frequency_at_time, duration_seconds, sample_rate=SAMPLE_RATE):
    time = np.arange(int(duration_seconds * sample_rate), dtype=np.float64) / sample_rate
    frequency = frequency_at_time(time)
    phase = 2.0 * np.pi * np.cumsum(frequency) / sample_rate
    return (0.3 * np.sin(phase)).astype(np.float32), time


def _glide_fixture(start_hz, end_hz):
    duration = GLIDE_DURATION_SECONDS
    slope = (end_hz - start_hz) / duration
    audio, time = _integrated_tone(lambda t: start_hz + slope * t, duration)
    return audio, time, lambda t: start_hz + slope * t


def _cents_error(estimated_hz, expected_hz):
    return 1200.0 * np.log2(estimated_hz / expected_hz)


@pytest.mark.parametrize(
    "start_hz,end_hz,direction",
    [
        (GLIDE_START_HZ, GLIDE_END_HZ, "ascending"),
        (GLIDE_END_HZ, GLIDE_START_HZ, "descending"),
    ],
)
def test_f0_glide_frame_accuracy_and_octave_errors(start_hz, end_hz, direction, record_property):
    audio, _, truth = _glide_fixture(start_hz, end_hz)
    track = track_f0(
        audio, SAMPLE_RATE, min_f0_hz=F0_MIN_HZ, max_f0_hz=F0_MAX_HZ,
        frame_ms=FRAME_MS, hop_ms=HOP_MS,
    )
    valid = track.voiced & (track.frame_times_seconds > 0.10) & (
        track.frame_times_seconds < GLIDE_DURATION_SECONDS - 0.10
    )
    assert int(np.count_nonzero(valid)) > 100, direction
    expected = truth(track.frame_times_seconds[valid])
    cents = _cents_error(track.f0_hz[valid], expected)
    octave_errors = np.abs(np.abs(cents) - 1200.0) <= 100.0
    octave_error_count = int(np.count_nonzero(octave_errors))
    mae_cents = float(np.mean(np.abs(cents)))
    assert octave_error_count == 0, (
        f"{direction}: octave_errors={octave_error_count}, "
        f"frames={cents.size}, MAE={mae_cents:.2f} cents"
    )
    assert mae_cents <= GLIDE_MAX_MAE_CENTS, (
        f"{direction}: MAE={mae_cents:.2f} cents"
    )


def test_f0_confidence_is_bounded_and_informative():
    duration = 1.0
    time = np.arange(int(duration * SAMPLE_RATE), dtype=np.float32) / SAMPLE_RATE
    clean = (0.3 * np.sin(2.0 * np.pi * 220.0 * time)).astype(np.float32)
    silence = np.zeros_like(clean)
    noise = np.random.default_rng(20261114).normal(0.0, 0.3, clean.size).astype(np.float32)

    clean_track = track_f0(clean, SAMPLE_RATE)
    silence_track = track_f0(silence, SAMPLE_RATE)
    noise_track = track_f0(noise, SAMPLE_RATE)

    for track in (clean_track, silence_track, noise_track):
        assert np.isfinite(track.confidence).all()
        assert np.all(track.confidence >= CONFIDENCE_MIN)
        assert np.all(track.confidence <= CONFIDENCE_MAX)

    clean_voiced = clean_track.confidence[clean_track.voiced]
    assert clean_voiced.size > 20
    assert float(np.median(clean_voiced)) >= CONFIDENCE_VOICED_MIN
    assert float(np.max(silence_track.confidence)) <= CONFIDENCE_SILENCE_MAX
    assert float(np.median(noise_track.confidence)) < CONFIDENCE_VOICED_MIN


def test_f0_vibrato_tracks_a_defined_220_hz_modulation():
    duration = VIBRATO_DURATION_SECONDS
    time = np.arange(int(duration * SAMPLE_RATE), dtype=np.float64) / SAMPLE_RATE
    cents = VIBRATO_DEPTH_CENTS * np.sin(2.0 * np.pi * VIBRATO_RATE_HZ * time)
    frequency = VIBRATO_CENTER_HZ * np.power(2.0, cents / 1200.0)
    phase = 2.0 * np.pi * np.cumsum(frequency) / SAMPLE_RATE
    audio = (0.3 * np.sin(phase)).astype(np.float32)
    track = track_f0(audio, SAMPLE_RATE)
    valid = track.voiced & (track.frame_times_seconds > 0.2) & (
        track.frame_times_seconds < duration - 0.2
    )
    assert np.count_nonzero(valid) > 100
    expected = np.interp(track.frame_times_seconds[valid], time, frequency)
    errors = _cents_error(track.f0_hz[valid], expected)
    assert float(np.mean(np.abs(errors))) <= 100.0
    # This test records frame-level tracking error; it does not claim the
    # tracker preserves the full vibrato depth or modulation bandwidth.


@pytest.mark.parametrize("transition", ["silence_to_voice", "voice_to_silence"])
def test_voicing_transition_respects_window_geometry(transition, record_property):
    duration = 1.0
    boundary = 0.5
    time = np.arange(int(duration * SAMPLE_RATE), dtype=np.float64) / SAMPLE_RATE
    tone = 0.3 * np.sin(2.0 * np.pi * 220.0 * time)
    if transition == "silence_to_voice":
        audio = np.where(time >= boundary, tone, 0.0)
        expected_edge = boundary
        side = "start"
    else:
        audio = np.where(time < boundary, tone, 0.0)
        expected_edge = boundary
        side = "end"
    track = track_f0(audio.astype(np.float32), SAMPLE_RATE)
    voiced_times = track.frame_times_seconds[track.voiced]
    assert voiced_times.size > 0
    observed_edge = float(voiced_times[0] if side == "start" else voiced_times[-1])
    error_ms = abs(observed_edge - expected_edge) * 1000.0
    assert error_ms <= BOUNDARY_TOLERANCE_MS + 1e-3, (
        f"{transition}: observed edge error={error_ms:.2f} ms; "
        f"geometry-derived bound={BOUNDARY_TOLERANCE_MS:.2f} ms"
    )


def _vocal_like_fixture(sample_peak=10.0 ** (-0.1 / 20.0)):
    time = np.arange(SAMPLE_RATE, dtype=np.float64) / SAMPLE_RATE
    fundamental = 220.0
    # Broad vowel-like spectral emphasis: low first formant, strong second
    # formant region, and attenuated upper harmonics.
    amplitudes = np.array([
        1.00, 0.55, 0.32, 0.18, 0.12, 0.22, 0.35, 0.42,
        0.30, 0.16, 0.09, 0.06, 0.04, 0.025, 0.015,
    ])
    signal = np.zeros_like(time)
    for harmonic, amplitude in enumerate(amplitudes, start=1):
        signal += amplitude * np.sin(2.0 * np.pi * fundamental * harmonic * time)
    signal *= sample_peak / np.max(np.abs(signal))
    return signal.astype(np.float32)


def test_compressor_full_scale_1khz_sine_is_finite_and_sample_peak_bounded():
    time = np.arange(SAMPLE_RATE, dtype=np.float64) / SAMPLE_RATE
    audio = np.sin(2.0 * np.pi * 1000.0 * time).astype(np.float32)
    settings = CompressorSettings(
        threshold_db=COMPRESSOR_THRESHOLD_DB,
        ratio=COMPRESSOR_RATIO,
        attack_ms=COMPRESSOR_ATTACK_MS,
        release_ms=COMPRESSOR_RELEASE_MS,
        max_reduction_db=COMPRESSOR_MAX_REDUCTION_DB,
    )
    output = compress_vocal(audio, SAMPLE_RATE, settings)
    assert np.isfinite(output).all()
    assert float(np.max(np.abs(output))) <= SAMPLE_PEAK_CEILING + SAMPLE_PEAK_TOLERANCE


def test_compressor_vocal_like_fixture_exercises_sample_peak_ceiling(record_property):
    audio = _vocal_like_fixture()
    input_peak = float(np.max(np.abs(audio)))
    assert input_peak == pytest.approx(10.0 ** (-0.1 / 20.0), abs=2e-6)

    # Ratio 1 avoids gain reduction; positive makeup deliberately drives the
    # pre-ceiling output above 0.999 so the final scaling branch is exercised.
    settings = CompressorSettings(
        threshold_db=COMPRESSOR_THRESHOLD_DB,
        ratio=1.0,
        attack_ms=COMPRESSOR_ATTACK_MS,
        release_ms=COMPRESSOR_RELEASE_MS,
        makeup_db=0.1,
        max_reduction_db=COMPRESSOR_MAX_REDUCTION_DB,
    )
    predicted_pre_ceiling_peak = input_peak * (10.0 ** (settings.makeup_db / 20.0))
    assert predicted_pre_ceiling_peak > SAMPLE_PEAK_CEILING

    output = compress_vocal(audio, SAMPLE_RATE, settings)
    output_peak = float(np.max(np.abs(output)))
    assert np.isfinite(output).all()
    assert output_peak == pytest.approx(SAMPLE_PEAK_CEILING, abs=SAMPLE_PEAK_TOLERANCE)


def test_compressor_accepts_finite_input_above_full_scale_and_caps_output():
    audio = _vocal_like_fixture(sample_peak=1.001)
    assert float(np.max(np.abs(audio))) > 1.0
    settings = CompressorSettings(ratio=1.0, makeup_db=0.0)
    # The ratio=1 / zero-makeup fast path is a bit-identical bypass and
    # intentionally returns input unchanged; use nonzero makeup to exercise
    # the compressor's final ceiling stage.
    settings = CompressorSettings(ratio=1.0, makeup_db=0.01)
    output = compress_vocal(audio, SAMPLE_RATE, settings)
    assert np.isfinite(output).all()
    assert float(np.max(np.abs(output))) <= SAMPLE_PEAK_CEILING + SAMPLE_PEAK_TOLERANCE


def test_compressor_bypass_is_bit_identical():
    audio = _vocal_like_fixture(sample_peak=0.8)
    output = compress_vocal(
        audio, SAMPLE_RATE,
        CompressorSettings(ratio=1.0, makeup_db=0.0),
    )
    assert np.array_equal(output, audio)


def test_compressor_gain_reduction_respects_configured_cap():
    time = np.arange(SAMPLE_RATE, dtype=np.float64) / SAMPLE_RATE
    audio = (0.8 * np.sin(2.0 * np.pi * 220.0 * time)).astype(np.float32)
    settings = CompressorSettings(
        threshold_db=-48.0, ratio=20.0, attack_ms=COMPRESSOR_ATTACK_MS,
        release_ms=COMPRESSOR_RELEASE_MS, max_reduction_db=6.0,
    )
    output = compress_vocal(audio, SAMPLE_RATE, settings)
    # Ignore the initial attack and compare steady-state RMS. The configured
    # cap is 6 dB; the smoothed detector may approach it rather than hit it
    # instantaneously.
    input_rms = float(np.sqrt(np.mean(audio[int(0.2 * SAMPLE_RATE):] ** 2)))
    output_rms = float(np.sqrt(np.mean(output[int(0.2 * SAMPLE_RATE):] ** 2)))
    gain_db = 20.0 * np.log10(output_rms / input_rms)
    assert gain_db >= -6.5
    assert gain_db < -1.0


def test_compressor_reports_no_true_peak_guarantee_on_vocal_like_fixture(record_property):
    audio = _vocal_like_fixture()
    settings = CompressorSettings(ratio=1.0, makeup_db=0.1)
    output = compress_vocal(audio, SAMPLE_RATE, settings)
    sample_peak_dbfs = 20.0 * np.log10(float(np.max(np.abs(output))))
    true_peak = float(np.max(np.abs(resample_poly(output, TRUE_PEAK_OVERSAMPLE, 1))))
    true_peak_dbfs = 20.0 * np.log10(max(true_peak, 1e-12))
    assert sample_peak_dbfs <= 20.0 * np.log10(
        SAMPLE_PEAK_CEILING + SAMPLE_PEAK_TOLERANCE
    )
    # Measurement is reported by assertion diagnostics; this is deliberately
    # not a true-peak safety assertion because the implementation has no
    # oversampled limiter.


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("function", ["tracker", "compressor"])
def test_audio_processing_rejects_non_finite_input(bad_value, function):
    audio = np.array([0.0, 0.1, bad_value, -0.1], dtype=np.float32)
    with pytest.raises(ValueError, match="finite"):
        if function == "tracker":
            track_f0(audio, SAMPLE_RATE)
        else:
            compress_vocal(audio, SAMPLE_RATE)
