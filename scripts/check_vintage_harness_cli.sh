#!/usr/bin/env bash
# Verify the harness CLI propagates pytest's exit status on both failure and success.
# Mutates one assertion in-place, checks the expected pytest result, restores the tree,
# then checks the clean run. The original test file is restored even on interruption.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

test_file="tests/test_vintage_harness.py"
tmp="$(mktemp -d)"
backup="$tmp/backup"
failure_log="$tmp/failure.log"
success_log="$tmp/success.log"
cp -- "$test_file" "$backup"

cleanup() {
  status=$?
  trap - EXIT INT TERM
  cp -- "$backup" "$test_file" || status=1
  if ! cmp -s -- "$backup" "$test_file"; then
    echo "ERROR: test file restore failed byte-for-byte" >&2
    status=1
  fi
  rm -rf -- "$tmp"
  find tests -name __pycache__ -type d -prune -exec rm -rf {} +
  exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# Avoid stale bytecode while the source is temporarily mutated.
export PYTHONDONTWRITEBYTECODE=1

python - "$test_file" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
source = path.read_text(encoding="utf-8")
needle = "pytest.approx(0.1)"
replacement = "pytest.approx(0.2)"
count = source.count(needle)
if count != 1:
    raise SystemExit(
        f"expected exactly one {needle!r} target, found {count}; refusing to mutate tests"
    )
path.write_text(source.replace(needle, replacement), encoding="utf-8")
PY

set +e
python -m elevenx.vintage_harness >"$failure_log" 2>&1
failure_status=$?
set -e

if [[ "$failure_status" -eq 0 || "$failure_status" -gt 125 ]]; then
  echo "ERROR: injected failure returned unexpected status=$failure_status" >&2
  cat "$failure_log" >&2
  exit 1
fi
if ! grep -Eq '9 passed.*1 failed|1 failed.*9 passed' "$failure_log"; then
  echo "ERROR: harness did not report the expected 9 passed / 1 failed result" >&2
  cat "$failure_log" >&2
  exit 1
fi
echo "Verified injected assertion failure returns exit=$failure_status (9 passed, 1 failed)"

# Restore the original test before checking the success path; keep the backup until EXIT.
cp -- "$backup" "$test_file"
set +e
python -m elevenx.vintage_harness >"$success_log" 2>&1
success_status=$?
set -e

if [[ "$success_status" -ne 0 ]] || ! grep -Eq '10 passed' "$success_log"; then
  echo "ERROR: restored harness did not report 10 passed with exit=0 (exit=$success_status)" >&2
  cat "$success_log" >&2
  exit 1
fi
echo "Verified restored test suite returns exit=0 (10 passed)"
