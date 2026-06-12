import sys
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from comet.config.io import load_config
from comet.io.region import bounds_from_restart, read_region
from comet.system.matching import (
    filter_dict,
    get_matching_gas_mols,
    get_matching_gas_mols_with_symbols,
)


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


def test_get_matching_gas_mols_splits_common_and_new_entries():
    xyz_arr1 = [np.array([0, 0, 0]), np.array([1, 1, 1])]
    xyz_arr2 = [np.array([1, 1, 1]), np.array([2, 2, 2])]

    common, not_common = get_matching_gas_mols(xyz_arr1, xyz_arr2)

    assert len(common) == 1
    assert np.array_equal(common[0], np.array([1, 1, 1]))
    assert len(not_common) == 1
    assert np.array_equal(not_common[0], np.array([0, 0, 0]))


def test_get_matching_gas_mols_with_symbols_respects_tolerance():
    box_gas = Atoms(
        symbols=["H", "N"],
        positions=[[0.0, 0.0, 0.0], [5.0, 5.0, 5.0]],
    )
    old_positions = np.array([[0.0, 0.0, 0.0], [9.0, 9.0, 9.0]])

    common, new_pos, new_syms = get_matching_gas_mols_with_symbols(
        box_gas,
        old_positions,
        tol=1e-6,
    )

    assert common.shape == (1, 3)
    assert np.allclose(common[0], [0.0, 0.0, 0.0])
    assert new_pos.shape == (1, 3)
    assert np.allclose(new_pos[0], [5.0, 5.0, 5.0])
    assert new_syms == ["N"]


def test_filter_dict_returns_matching_keys_only():
    original = {
        "first": np.array([0.0, 0.0, 0.0]),
        "second": np.array([1.0, 1.0, 1.0]),
    }

    filtered_keys = set(filter_dict(original, [np.array([1.0, 1.0, 1.0])]))

    assert filtered_keys == {"second"}
