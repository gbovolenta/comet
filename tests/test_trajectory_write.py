"""Trajectory writes must be immune to shared-calculator cross-talk.

Regression test for the v0.3.1 calculator-cache bug: one calculator instance
attached to several structures serves up the *last* evaluation's arrays, so
writing an earlier (smaller) structure crashed the extxyz writer with a shape
mismatch after any rejected GCMC move.
"""

import sys
from pathlib import Path

from ase import Atoms
from ase.calculators.lj import LennardJones
from ase.io import read

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from comet.io.trajectory import write_extxyz_sequence


def test_write_extxyz_ignores_stale_shared_calculator(tmp_path: Path):
    # One shared calculator, attached to two structures of different sizes —
    # exactly what the pre-fix MACE backend did to box_gas and a trial box.
    shared = LennardJones()

    box = Atoms("H2", [[0, 0, 0], [0, 0, 0.9]], cell=[10, 10, 10], pbc=True)
    box.calc = shared
    box.get_potential_energy()

    trial = Atoms("H4", [[0, 0, 0], [0, 0, 0.9], [3, 3, 3], [3, 3, 3.9]],
                  cell=[10, 10, 10], pbc=True)
    trial.calc = shared
    trial.get_potential_energy()   # shared calculator now holds 4-atom results

    out = tmp_path / "traj.extxyz"
    write_extxyz_sequence(out, box)   # pre-fix: ValueError (4-atom arrays on 2-atom frame)

    frames = read(str(out), index=":")
    assert len(frames) == 1
    assert len(frames[0]) == 2


def test_write_extxyz_appends_frames(tmp_path: Path):
    out = tmp_path / "traj.extxyz"
    box = Atoms("H2", [[0, 0, 0], [0, 0, 0.9]], cell=[10, 10, 10], pbc=True)
    box.set_tags(3)

    write_extxyz_sequence(out, box)
    write_extxyz_sequence(out, box)

    frames = read(str(out), index=":")
    assert len(frames) == 2
    # Per-atom data (molecule-id tags) survives the calculator-free copy.
    assert list(frames[0].get_tags()) == [3, 3]
