import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from comet.config.io import load_config
from comet.io.region import bounds_from_restart, read_region


def test_load_config_reads_nested_yaml(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("root:\n  child: 1\nitems:\n  - H2\n", encoding="utf8")

    config = load_config(str(config_path))

    assert config == {"root": {"child": 1}, "items": ["H2"]}


def test_read_region_missing_file_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        read_region(tmp_path / "missing.region")


def test_bounds_from_restart_missing_file_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        bounds_from_restart(tmp_path / "missing.data", z_cutoff=5.0)
