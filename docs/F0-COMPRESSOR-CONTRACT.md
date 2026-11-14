# F0 and compressor verification contract

## Scope and implementation identity

This verification suite exercises only:

- `elevenx.audio_processing::track_f0`
- `elevenx.audio_processing::compress_vocal`

It does not test, reference, or depend on any external plugin or commercial product.

The inspected baseline implementation is `src/elevenx/audio_processing.py`, blob SHA `760131bcec75975e8e56565751acd3de08346fea`. The compressor's sample-peak ceiling is `0.999` linear, applied after its feed-forward gain stage and makeup gain. It is not a true-peak limiter. The verification change preserves this ceiling; changing it is a separate sound-design/API decision.

## PitchTrack confidence and future HNR

- `PitchTrack.confidence` is the normalized autocorrelation peak, bounded to [0, 1], with higher values indicating a stronger periodic autocorrelation peak. It is not HNR. Treating it directly as HNR is incorrect.
- The future `hnr_dB` definition is Boersma's mapping from a normalized autocorrelation peak (r), for voiced frames, only when that peak is the normalized autocorrelation quantity described above:

  `HNR_dB = 10 * log10(r / (1 - r))`

- The mapping is undefined at exactly (r=0) and (r=1). A future implementation must clamp before evaluating the logarithm and clamp its result to the declared range [-20, +40] dB. The corresponding confidence bounds are (r_{min}=1/101) and (r_{max}=10000/10001). The exact upper bound is approximately 0.999900009999.
- No HNR estimator exists in this milestone. `hnr_dB` is absent, not zero-filled. Confidence bounds and clean/noise/silence informativeness are separate test properties.
- Reference: Boersma, P. (1993), “Accurate short-term analysis of the fundamental frequency and the harmonics-to-noise ratio of a sampled sound,” *Proceedings of the Institute of Phonetic Sciences*, University of Amsterdam, 17, 97–110.

## F0 fixture methodology

The glide fixtures are linear 100-to-400 Hz and 400-to-100 Hz sweeps over 2 seconds at 48 kHz. Frame-level estimates are compared with the instantaneous synthetic fundamental at each reported frame center; octave-error counts and mean absolute cents error are asserted separately for each direction. The vibrato fixture is centered at 220 Hz with ±15 cents depth at 5.5 Hz over 4 seconds. The tests record errors but do not claim a measured accuracy result until executed.

## Voicing boundary geometry

The tracker uses 40 ms frames and 10 ms hops: 30 ms overlap. The frame-time labels are frame centers. The implementation classifies each frame using the normalized autocorrelation peak threshold (`peak >= voiced_threshold`, default 0.60); it does not apply a separate post-decision hangover.

The test uses a geometry-derived 30 ms maximum boundary tolerance (frame duration minus hop duration) for the synthetic silence/voice transitions. This is an explicit test bound for this implementation and fixture, not a universal property of all possible classifiers. If the classifier rule, window convention, frame size, or hop changes, the tolerance and its derivation must change with it.

## Compressor contract

Current defaults are threshold -18 dB, ratio 3:1, attack 10 ms, release 100 ms, makeup 0 dB, and maximum gain reduction 12 dB. The sample-peak ceiling is 0.999 linear (approximately -0.0087 dBFS). Since the scale factor and output multiplication are float32 operations, tests allow a 1e-6 absolute tolerance around the ceiling.

The compressor accepts finite input samples above full scale; it does not pre-clip the input. The final ceiling operation runs after compression and makeup. Tests include a vocal-like 15-harmonic 220 Hz fixture normalized to -0.1 dBFS and a separate >1.0 finite-input fixture. The ceiling-triggering fixture is constructed so the pre-ceiling peak exceeds 0.999; a separate below-ceiling/no-trigger case must be reported by execution evidence before the ceiling path can be considered fully characterized.

The compressor does not oversample its detector or final limiter. A 4x resampled true-peak measurement is diagnostic only; this implementation makes no true-peak safety guarantee. Downstream stages requiring inter-sample-peak safety must provide their own true-peak control.

The ratio=1, makeup=0 fast path returns an exact copy. Tests require bit-identical output with `np.array_equal`. Non-finite audio is rejected with `ValueError` by both the tracker and compressor.

## Future API boundary

Current `SynthesisResult.analysis` values are summary diagnostics only (median F0, voiced fraction, pitch-frame count). Frame-level F0/voicing/HNR export is out of scope for this verification change; the eventual contract will live in a separate `VocalAnalysisResult` object, and `SynthesisResult` will hold a summary reference only. HNR must not be inferred from a differently defined confidence measure.

## What this verification does and does not prove

It is intended to verify synthetic F0 glide behavior, confidence bounds/informativeness, synthetic voicing transitions, and bounded behavior of the named Python compressor on specified fixtures, including bit-identical bypass and non-finite input rejection.

It does not prove F0 tracking on real singing, voicing accuracy on real breathy or whispered speech, compressor sound quality on real vocals, the existence or accuracy of an HNR estimator, true-peak safety, or correctness of downstream consumers' assumptions.

## Measurement status

This document defines test fixtures, assertions, and implementation limits; it contains no measured F0 errors, octave-error rates, observed boundary timings, compressor peak measurements, or true-peak results. Those values remain unreported until the test suite is executed against a named commit SHA. No expected or illustrative value is presented as a measurement.
