# Security and privacy notes

ElevenX is a local-first voice-synthesis prototype. The API is intended for use on the same machine, not as an internet-facing service.

## Safe operating boundary

- Bind Uvicorn to `127.0.0.1` for local use. Do not bind to `0.0.0.0` or expose port 8000 to a LAN or the public internet without a separate security design.
- The current API does not provide user authentication, authorization, request rate limiting, or TLS termination. The voice-permission checkbox is an explicit consent signal, not identity verification or proof of rights.
- Treat reference recordings and generated WAV files as sensitive. Keep them on trusted local storage and do not upload them to third-party services.
- Do not enable telemetry, remote inference, automatic update checks, or background capture.
- Model downloads may require network access on first use. Once required assets are cached, offline operation depends on the installed libraries and local cache being available.

## Request handling

The API requires positive consent, bounds text length and reference upload size, validates the filename suffix, rejects concurrent synthesis with HTTP 503 and `Retry-After`, and removes temporary files after response delivery or error handling. Filename suffix checks are not a substitute for decoding and validating media contents. Do not treat uploaded audio as trusted input.

## Before remote deployment

Do not expose the current service directly. A remote deployment must first add and test authentication, authorization, rate limiting, request-size limits at the reverse proxy, TLS, controlled model/cache directories, logging policy, retention/deletion policy, abuse prevention, and a robust consent/rights workflow.

## Voice cloning

Only use reference audio when you have permission to use that voice. The UI/API consent field records the caller's confirmation; it cannot establish ownership, identity, or legal permission independently.
