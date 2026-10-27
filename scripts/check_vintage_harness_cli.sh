#!/usr/bin/env bash
# Verify the harness CLI propagates pytest's exit status on both failure and success.
# The test file is restored even if either command is interrupted or fails.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

test_file="tests/test_vintage_harness.py"
backup="$(mktemp)"
cp "$test_file" "$backup"
restore() {
  cp "$backup" "$test_file"
  rm -f "$backup"
}
trap restore EXIT INT TERM

# Force one known assertion to fail without changing the checked-in test afterward.
python - "$test_file" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
source = path.read_text(encoding="utf-8")
needle = 'pytest.approx(0.1)'
replacement = 'pytest.approx(0.2)'
if source.count(needle) != 1:
    raise SystemExit(
        "expected exactly one pytest.approx(0.1) target; refusing to mutate tests"
    )
path.write_text(source.replace(needle, replacement), encoding="utf-8")
PY

set +e
python -m elevenx.vintage_harness
failure_status=$?
set -e
if [[ "$failure_status" -eq 0 ]]; then
  echo "ERROR: injected assertion failure unexpectedly returned exit=0" >&2
  exit 1
fi
echo "Verified injected failure propagates nonzero exit=$failure_status"

# Restore before the success-path run, then remove the trap's temporary backup.
restore
trap - EXIT INT TERM

python -m elevenx.vintage_harness
echo "Verified restored test suite returns exit=0"
