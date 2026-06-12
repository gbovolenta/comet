import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from comet.config.io import load_config


def test_yaml_loader_preserves_keys():
    pytest.importorskip("yaml")
    config = load_config(PROJECT_ROOT / "examples" / "config.yaml")
    assert isinstance(config, dict)

    expected_keys = {
        "bdir",
        "model_dir",
        "h2_path",
        "traj_path",
        "restart_path",
        "box_path",
        "elements",
        "z_cutoff",
        "temperature",
        "chemical_potential",
        "mass",
        "h2_energy",
        "steps",
    }

    assert expected_keys.issubset(set(config.keys()))
