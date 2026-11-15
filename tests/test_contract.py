"""Source-contract regression tests; all tests are model-free."""
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "elevenx"
CHECKER = ROOT / "scripts" / "check_py39.py"


def test_py39_guard_scans_runtime_and_checker_itself() -> None:
    result = subprocess.run(
        [sys.executable, str(CHECKER), str(SRC), str(ROOT / "scripts")],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "source",
    [
        "def f(x: int | None): pass\\n",
        "def f(x: int | str): pass\\n",
        "match value:\\n    case 1: pass\\n",
    ],
)
def test_py39_guard_rejects_python_310_syntax(tmp_path: Path, source: str) -> None:
    bad = tmp_path / "bad.py"
    bad.write_text(source, encoding="utf-8")
    result = subprocess.run(
        [sys.executable, str(CHECKER), str(bad)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "violation" in result.stderr or "syntax error" in result.stderr


@pytest.mark.parametrize(
    "source",
    [
        "def f(x: list[int]): pass\\n",
        "def f(x: dict[str, int]): pass\\n",
        "import asyncio\\nasync def f(): await asyncio.to_thread(print)\\n",
        "import functools\\n@functools.cache\\ndef f(): return 1\\n",
        "def f(s: str): return s.removeprefix('x')\\n",
    ],
)
def test_py39_guard_accepts_python_39_features(tmp_path: Path, source: str) -> None:
    good = tmp_path / "good.py"
    good.write_text(source, encoding="utf-8")
    result = subprocess.run(
        [sys.executable, str(CHECKER), str(good)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_missing_consent_returns_400() -> None:
    from elevenx.api import app

    response = TestClient(app).post(
        "/v1/synthesize",
        data={"text": "hello"},
        files={"reference_audio": ("reference.wav", b"audio", "audio/wav")},
    )
    assert response.status_code == 400


def test_false_consent_returns_400() -> None:
    from elevenx.api import app

    response = TestClient(app).post(
        "/v1/synthesize",
        data={"text": "hello", "consent": "false"},
        files={"reference_audio": ("reference.wav", b"audio", "audio/wav")},
    )
    assert response.status_code == 400
