# HarmoniaCodec integration boundary

ElevenX currently uses Coqui XTTS-v2 for speech synthesis. HarmoniaCodec is not
a drop-in replacement for a text-to-speech model: a codec encodes and decodes
audio representations, and needs a compatible trained model plus a defined
conditioning/inference pipeline to contribute to speech generation.

The new `elevenx.harmoniacodec` module defines an adapter contract and a small
readiness guard. It does **not** implement encoding/decoding, load checkpoints,
or connect HarmoniaCodec to the active XTTS path.

Before enabling a production adapter:

1. Review the actual HarmoniaCodec source and license, dependency tree, checkpoint
   format, and model-loading behavior.
2. Supply trained weights from a documented training run; do not treat random
   initialization as a trained codec.
3. Run a reproducible encode/decode benchmark on representative audio and record
   the benchmark ID, pass/fail decision, and SHA-256 of the exact weights file.
4. Define measurable quality and safety thresholds (for example, round-trip
   fidelity, sample-rate/channel handling, clipping, finite output, and latency).
5. Implement and review a concrete `HarmoniaCodecAdapter`, add integration tests
   with real weights, and explicitly opt in through deployment configuration.
6. Keep XTTS as the default/fallback until the adapter is validated end to end.

`check_harmonia_readiness` is a deployment guard, not cryptographic attestation
of a benchmark or proof of codec quality. A matching hash only binds the report
to a configured weights identifier; the report and benchmark process still need
review. This PR intentionally does not make audio-quality or performance claims.
