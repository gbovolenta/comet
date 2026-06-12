import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from comet.io.region import read_region


def test_read_region_roundtrip(tmp_path: Path):
    bounds = [0.0, 10.0, -5.0, 5.0, 1.0, 2.0]
    region_file = tmp_path / "region.txt"
    region_file.write_text(" ".join(map(str, bounds)))

    parsed = read_region(region_file)

    assert parsed == tuple(bounds)


def test_import_lammps_datafile():
    pytest.importorskip("comet.io.lammps_datafile")
