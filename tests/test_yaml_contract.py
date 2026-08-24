import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from comet.config.io import load_config


def test_yaml_loader_preserves_keys():
    pytest.importorskip("yaml")
    config = load_config(PROJECT_ROOT / "examples" / "pure_H2" / "config.yaml")
    assert isinstance(config, dict)

    # Keys the workflow actually consumes (see comet.workflows.run).
    # `slab` and `gas_masses` are optional (redundant consistency checks) and
    # intentionally absent from the example.
    expected_keys = {
        "energy_backend",
        "bdir",
        "model_dir",
        "restart_path",
        "elements",
        "gas_list",
        "gas_template_dir",
        "z_cutoff",
        "temperature",
        "pressure_unit",
        "steps",
    }

    assert expected_keys.issubset(set(config.keys()))

    # μ targets come from either per-species partial pressures (preferred) or
    # the legacy total-pressure fallback.
    assert ("partial_pressures" in config) or ("pressure" in config)
