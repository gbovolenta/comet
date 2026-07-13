import sys
from pathlib import Path

import numpy as np
from ase import Atoms

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from comet.io.lammps_datafile import assign_ids_preserve_slab


def test_assign_ids_preserve_slab_reuses_gaps_for_gas():
    old_atoms_by_id = {
        1: (1, np.array([0.0, 0.0, 0.0]), (0, 0, 0)),
        2: (1, np.array([1.0, 0.0, 0.0]), (0, 0, 0)),
        3: (2, np.array([0.0, 1.0, 0.0]), (0, 0, 0)),
        4: (2, np.array([1.0, 1.0, 0.0]), (0, 0, 0)),
        5: (3, np.array([0.0, 0.0, 5.0]), (0, 0, 0)),
        6: (3, np.array([1.0, 0.0, 5.0]), (0, 0, 0)),
    }
    new_struct = Atoms(
        symbols=["Fe", "Fe", "N", "N", "H", "H"],
        positions=[
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [2.0, 0.0, 5.0],
            [3.0, 0.0, 5.0],
        ],
    )

    new_ids, matched_old_id = assign_ids_preserve_slab(
        old_atoms_by_id=old_atoms_by_id,
        new_struct=new_struct,
        elements=["Fe", "N", "H"],
        n_slab=2,
        tol=1e-8,
        reuse_old_ids_for_gas=False,
    )

    assert list(new_ids[:2]) == [1, 2]
    assert sorted(new_ids.tolist()) == [1, 2, 3, 4, 5, 6]
    assert set(new_ids[2:].tolist()) == {3, 4, 5, 6}
    assert list(matched_old_id[:2]) == [1, 2]


def test_assign_ids_preserve_slab_can_still_preserve_old_gas_ids():
    old_atoms_by_id = {
        1: (1, np.array([0.0, 0.0, 0.0]), (0, 0, 0)),
        2: (3, np.array([0.0, 0.0, 5.0]), (0, 0, 0)),
        3: (3, np.array([1.0, 0.0, 5.0]), (0, 0, 0)),
    }
    new_struct = Atoms(
        symbols=["Fe", "H", "H"],
        positions=[
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 5.0],
            [1.0, 0.0, 5.0],
        ],
    )

    new_ids, _ = assign_ids_preserve_slab(
        old_atoms_by_id=old_atoms_by_id,
        new_struct=new_struct,
        elements=["Fe", "H"],
        n_slab=1,
        tol=1e-8,
        reuse_old_ids_for_gas=True,
    )

    assert list(new_ids) == [1, 2, 3]
