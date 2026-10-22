#!/usr/bin/env python3
"""Check README local links and core project documentation invariants."""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"

REQUIRED_PHRASES = (
    "Local-first voice synthesis prototype",
    "XTTS-v2",
    "SpeechBrain",
    "model-free",
    "authentication",
    "smoke test",
)
MARKDOWN_LINK = re.compile(r"!?(?:\[[^\]]*\])\(([^)]+)\)")


def main() -> int:
    errors: list[str] = []
    if not README.is_file():
        print("ERROR: README.md is missing")
        return 1

    content = README.read_text(encoding="utf-8")
    for phrase in REQUIRED_PHRASES:
        if phrase.casefold() not in content.casefold():
            errors.append(f"README.md is missing required documentation phrase: {phrase!r}")

    for raw_target in MARKDOWN_LINK.findall(content):
        target = raw_target.strip().split("#", 1)[0]
        if not target or re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*:", target) or target.startswith("#"):
            continue
        # Ignore templated or non-path Markdown destinations.
        if target.startswith("{") or target.startswith("<"):
            continue
        path = (ROOT / target).resolve()
        if ROOT.resolve() not in path.parents and path != ROOT.resolve():
            errors.append(f"Markdown link escapes repository: {raw_target!r}")
        elif not path.exists():
            errors.append(f"Broken local Markdown link: {raw_target!r}")

    # Guard against documenting an unauthenticated API as safe for public exposure.
    low = content.casefold()
    if "no authentication" not in low or "do not expose it publicly" not in low:
        errors.append("README must clearly disclose the unauthenticated API and warn against public exposure")

    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1

    print(f"README check passed: {len(MARKDOWN_LINK.findall(content))} Markdown links/images inspected.")
    print("Required project scope and API security caveats are documented.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
