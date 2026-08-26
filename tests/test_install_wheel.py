"""Build the wheel, install it in an isolated venv, and run the full test suite there.

This guards against regressions that only surface for installed (non-editable)
packages, e.g. missing data files or path assumptions that hold in editable mode.

The wheel and isolated venv live under `.install-test/` and are removed in a
teardown step so no build artifacts are left behind. When this file runs inside
the installed-wheel suite (nested run), it is skipped to avoid infinite recursion.
"""

import os
import shutil
import subprocess
from collections.abc import Iterator
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
INSTALL_DIR = ROOT / ".install-test"
VENV = INSTALL_DIR / "venv"
DIST = INSTALL_DIR / "dist"
NESTED_ENV = "ML_TOOLS_TEST_INSTALL"

pytestmark = pytest.mark.skipif(
    os.environ.get(NESTED_ENV) == "1",
    reason="already running inside the installed-wheel test",
)


def _venv_python() -> Path:
    return VENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def _run_uv(uv: str, args: list[str]) -> None:
    subprocess.run([uv, *args], check=True, cwd=ROOT, capture_output=True, text=True)


@pytest.fixture(name="installed_python")
def _installed_python() -> Iterator[Path]:
    uv = shutil.which("uv")
    assert uv is not None, "uv is required to install the wheel"

    _run_uv(uv, ["build", "--wheel", "-o", str(DIST), "--no-build-logs"])
    wheel = next(DIST.glob("*.whl"))
    _run_uv(uv, ["venv", "--clear", str(VENV)])
    _run_uv(
        uv,
        ["pip", "install", "--python", str(_venv_python()), str(wheel), "pytest"],
    )

    yield _venv_python()

    if INSTALL_DIR.exists():
        shutil.rmtree(INSTALL_DIR)


def test_full_suite_passes_against_installed_wheel(installed_python: Path) -> None:
    """The entire test suite passes against the non-editable installed package."""
    env = dict(os.environ)
    env["MPLBACKEND"] = "Agg"
    env[NESTED_ENV] = "1"
    result = subprocess.run(
        [str(installed_python), "-m", "pytest", "tests", "-q", "-p", "no:cacheprovider"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
