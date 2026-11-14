import numpy as np
import pytest

from elevenx.audio_processing import CompressorSettings, compress_vocal, track_f0


def test_f0_tracker_finds_clean_sine_pitch():
    sample_rate = 24000
    time = np.arange(sample_rate, dtype=np.float32) / sample_rate
    audio = 0.3 * np.sin(2.0 * np.pi * 220.0 * time)
    track = track_f0(audio, sample_rate)
    voiced = track.f0_hz[track.voiced]
    assert voiced.size > 20
    assert float(np.median(voiced)) == pytest.approx(220.0, abs=3.0)


def test_f0_tracker_marks_silence_unvoiced():
    track = track_f0(np.zeros(12000, dtype=np.float32), 24000)
    assert not bool(np.any(track.voiced))
    assert bool(np.all(track.f0_hz == 0.0))


def test_f0_tracker_rejects_non_finite_audio():
    with pytest.raises(ValueError, match="finite"):
        track_f0(np.array([0.0, np.nan]), 24000)


def test_compressor_reduces_high_level_signal_and_stays_peak_safe():
    sample_rate = 24000
    time = np.arange(sample_rate, dtype=np.float32) / sample_rate
    audio = 0.9 * np.sin(2.0 * np.pi * 220.0 * time)
    processed = compress_vocal(audio, sample_rate)
    assert processed.shape == audio.shape
    assert np.max(np.abs(processed)) <= 0.999
    assert np.sqrt(np.mean(processed ** 2)) < np.sqrt(np.mean(audio ** 2))


def test_compressor_is_finite_and_does_not_modify_input():
    audio = np.linspace(-0.2, 0.2, 1024, dtype=np.float32)
    original = audio.copy()
    processed = compress_vocal(audio, 24000, CompressorSettings(ratio=1.0))
    assert np.array_equal(audio, original)
    assert np.isfinite(processed).all()


@pytest.mark.parametrize("kwargs", [
    {"ratio": 0.5}, {"threshold_db": -90.0}, {"attack_ms": 0.0},
    {"release_ms": 3000.0}, {"makeup_db": 20.0},
])
def test_invalid_compressor_settings_are_rejected(kwargs):
    with pytest.raises(ValueError):
        compress_vocal(np.ones(100, dtype=np.float32) * 0.1, 24000,
                       CompressorSettings(**kwargs))
