import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from comet.potentials.orca import _orca_blocks, _orca_simpleinput


def test_orca_simpleinput_defaults():
    assert _orca_simpleinput({}) == "PBE def2-SVP"


def test_orca_simpleinput_explicit_string_wins():
    assert _orca_simpleinput({"orca_simpleinput": "B3LYP def2-TZVP TightSCF"}) == "B3LYP def2-TZVP TightSCF"


def test_orca_blocks_include_pal_and_extra_blocks():
    blocks = _orca_blocks({"orca_nprocs": 4, "orca_blocks": "%scf MaxIter 200 end"})
    assert "%pal nprocs 4 end" in blocks
    assert "%scf MaxIter 200 end" in blocks
