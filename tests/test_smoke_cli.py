import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"


def test_comet_help():
    result = subprocess.run(
        [sys.executable, "-m", "comet", "--help"], capture_output=True, cwd=SRC_DIR
    )
    assert result.returncode == 0


def test_comet_run_help():
    result = subprocess.run(
        [sys.executable, "-m", "comet", "run", "--help"],
        capture_output=True,
        cwd=SRC_DIR,
    )
    assert result.returncode == 0
