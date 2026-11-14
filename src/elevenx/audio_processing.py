"""Offline vocal analysis and bounded post-synthesis dynamics for ElevenX.

The F0 tracker is a NumPy normalized-autocorrelation implementation inspired by
SingerFlow's documented approach. The compressor is a deterministic classical
inference path; Neural-Compressor-Script remains a training project and its
checkpoints are not loaded here.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PitchTrack:
    f0_hz: np.ndarray
    confidence: np.ndarray
    voiced: np.ndarray
    frame_times_seconds: np.ndarray


@dataclass(frozen=True)
class CompressorSettings:
    threshold_db: float = -18.0
    ratio: float = 3.0
    attack_ms: float = 10.0
    release_ms: float = 100.0
    makeup_db: float = 0.0
    max_reduction_db: float = 12.0

    def validate(self) -> None:
        if not -60.0 <= self.threshold_db <= 0.0:
            raise ValueError("threshold_db must be between -60 and 0")
        if not 1.0 <= self.ratio <= 20.0:
            raise ValueError("ratio must be between 1 and 20")
        if not 0.1 <= self.attack_ms <= 200.0:
            raise ValueError("attack_ms must be between 0.1 and 200")
        if not 1.0 <= self.release_ms <= 2000.0:
            raise ValueError("release_ms must be between 1 and 2000")
        if not -12.0 <= self.makeup_db <= 12.0:
            raise ValueError("makeup_db must be between -12 and 12")
        if not 0.0 <= self.max_reduction_db <= 24.0:
            raise ValueError("max_reduction_db must be between 0 and 24")


def track_f0(
    audio: np.ndarray,
    sample_rate: int,
    *,
    min_f0_hz: float = 65.0,
    max_f0_hz: float = 500.0,
    frame_ms: float = 40.0,
    hop_ms: float = 10.0,
    voiced_threshold: float = 0.60,
) -> PitchTrack:
    """Estimate monophonic F0 using windowed normalized autocorrelation."""
    signal = np.asarray(audio, dtype=np.float64).reshape(-1)
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if signal.size == 0 or not np.isfinite(signal).all():
        raise ValueError("audio must be non-empty and finite")
    if not 0.0 < min_f0_hz < max_f0_hz < sample_rate / 2:
        raise ValueError("F0 range must be positive and below Nyquist")
    if frame_ms <= 0.0 or hop_ms <= 0.0:
        raise ValueError("frame_ms and hop_ms must be positive")

    frame_size = max(16, int(round(sample_rate * frame_ms / 1000.0)))
    hop_size = max(1, int(round(sample_rate * hop_ms / 1000.0)))
    min_lag = max(1, int(sample_rate / max_f0_hz))
    max_lag = min(frame_size - 2, int(sample_rate / min_f0_hz))
    window = np.hanning(frame_size)
    starts = range(0, max(1, signal.size - frame_size + 1), hop_size)
    starts = list(starts)
    if not starts or starts[-1] + frame_size < signal.size:
        starts.append(max(0, signal.size - frame_size))

    f0_values: list[float] = []
    confidence_values: list[float] = []
    voiced_values: list[bool] = []
    times: list[float] = []
    for start in starts:
        frame = signal[start:start + frame_size]
        if frame.size < frame_size:
            frame = np.pad(frame, (0, frame_size - frame.size))
        frame = (frame - float(np.mean(frame))) * window
        energy = float(np.dot(frame, frame))
        if energy <= 1e-12:
            f0_values.append(0.0)
            confidence_values.append(0.0)
            voiced_values.append(False)
            times.append((start + frame_size / 2) / sample_rate)
            continue

        fft_size = 1 << (2 * frame_size - 1).bit_length()
        spectrum = np.fft.rfft(frame, n=fft_size)
        corr = np.fft.irfft(spectrum * np.conjugate(spectrum), n=fft_size)[:frame_size]
        corr = corr / max(float(corr[0]), 1e-12)
        segment = corr[min_lag:max_lag + 1]
        if segment.size == 0:
            peak_lag = 0
            peak = 0.0
        else:
            offset = int(np.argmax(segment))
            peak_lag = min_lag + offset
            peak = float(segment[offset])
            # Parabolic interpolation improves pitch resolution between lags.
            if 0 < peak_lag < corr.size - 1:
                left, center, right = corr[peak_lag - 1:peak_lag + 2]
                denom = left - 2.0 * center + right
                if abs(float(denom)) > 1e-12:
                    peak_lag += float(0.5 * (left - right) / denom)
        is_voiced = peak >= voiced_threshold and peak_lag > 0
        f0_values.append(float(sample_rate / peak_lag) if is_voiced else 0.0)
        confidence_values.append(max(0.0, min(1.0, peak)))
        voiced_values.append(bool(is_voiced))
        times.append((start + frame_size / 2) / sample_rate)

    return PitchTrack(
        f0_hz=np.asarray(f0_values, dtype=np.float32),
        confidence=np.asarray(confidence_values, dtype=np.float32),
        voiced=np.asarray(voiced_values, dtype=np.bool_),
        frame_times_seconds=np.asarray(times, dtype=np.float32),
    )


def compress_vocal(audio: np.ndarray, sample_rate: int,
                   settings: CompressorSettings | None = None) -> np.ndarray:
    """Apply an offline peak-safe feed-forward compressor to a mono waveform."""
    cfg = settings or CompressorSettings()
    cfg.validate()
    signal = np.asarray(audio, dtype=np.float32).reshape(-1)
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if signal.size == 0 or not np.isfinite(signal).all():
        raise ValueError("audio must be non-empty and finite")
    if cfg.ratio == 1.0 and cfg.makeup_db == 0.0:
        return signal.copy()

    attack = np.exp(-1.0 / (sample_rate * cfg.attack_ms / 1000.0))
    release = np.exp(-1.0 / (sample_rate * cfg.release_ms / 1000.0))
    threshold = cfg.threshold_db
    makeup = 10.0 ** (cfg.makeup_db / 20.0)
    max_reduction = cfg.max_reduction_db
    envelope = 0.0
    gain = 1.0
    output = np.empty_like(signal)

    for index, sample in enumerate(signal):
        level = abs(float(sample))
        coeff = attack if level > envelope else release
        envelope = coeff * envelope + (1.0 - coeff) * level
        level_db = 20.0 * np.log10(max(envelope, 1e-8))
        over_db = max(0.0, level_db - threshold)
        reduction_db = min(max_reduction, over_db * (1.0 - 1.0 / cfg.ratio))
        target_gain = 10.0 ** (-reduction_db / 20.0)
        gain_coeff = attack if target_gain < gain else release
        gain = gain_coeff * gain + (1.0 - gain_coeff) * target_gain
        output[index] = sample * gain * makeup

    peak = float(np.max(np.abs(output))) if output.size else 0.0
    if peak > 0.999:
        output *= np.float32(0.999 / peak)
    return output
