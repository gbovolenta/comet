"""Sampling correctness of the MC move proposals.

Insertion orientations must be uniform on SO(3) and deletion must pick a
molecule uniformly at random — both are assumed by the GCMC acceptance rule
(detailed balance).
"""

import random
import sys
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from comet.mc.moves import _random_rotation_matrix, deletion_mc, insertion_mc


def test_random_rotation_matrix_is_proper_rotation():
    np.random.seed(1)
    for _ in range(10):
        R = _random_rotation_matrix()
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-12)
        assert np.isclose(np.linalg.det(R), 1.0)


def test_random_rotation_orientations_cover_sphere_uniformly():
    np.random.seed(2)
    z = np.array([0.0, 0.0, 1.0])
    dirs = np.array([_random_rotation_matrix() @ z for _ in range(4000)])
    # Uniform on the sphere: each Cartesian component has mean 0 and
    # variance 1/3.
    assert np.all(np.abs(dirs.mean(axis=0)) < 0.05)
    assert np.allclose((dirs**2).mean(axis=0), 1.0 / 3.0, atol=0.03)


def test_insertion_preserves_bond_length_and_places_com_in_bounds():
    random.seed(4)
    np.random.seed(4)
    bond = 1.10
    templates = {"N2": Atoms("NN", [[0, 0, 0], [0, 0, bond]])}
    bounds = (2.0, 18.0, 2.0, 18.0, 5.0, 15.0)
    x0, x1, y0, y1, z0, z1 = bounds

    for _ in range(50):
        box = Atoms(cell=[20.0, 20.0, 20.0], pbc=[True, True, False])
        new_mol, new_box, name = insertion_mc(templates, bounds, box, mol_id=1)
        assert name == "N2"
        assert set(new_mol.get_tags()) == {1}
        p = new_mol.get_positions()
        assert np.isclose(np.linalg.norm(p[0] - p[1]), bond)
        com = new_mol.get_center_of_mass()
        assert x0 <= com[0] <= x1
        assert y0 <= com[1] <= y1
        assert z0 <= com[2] <= z1


def _h2_at(center, bond):
    cx, cy, cz = center
    return Atoms("HH", [[cx, cy, cz - bond / 2], [cx, cy, cz + bond / 2]])


def test_deletion_chooses_uniformly_among_molecules():
    random.seed(3)
    # Four H2 molecules, tagged 1-4; deletion picks a mol_id uniformly.
    centers = [(3.0, 3.0, 3.0), (3.0, 12.0, 3.0), (12.0, 3.0, 3.0), (12.0, 12.0, 3.0)]
    mol_species = {1: "H2", 2: "H2", 3: "H2", 4: "H2"}

    n_trials = 400
    counts = {1: 0, 2: 0, 3: 0, 4: 0}
    for _ in range(n_trials):
        box = Atoms(cell=[15.0, 15.0, 15.0], pbc=[True, True, False])
        for mol_id, center in enumerate(centers, start=1):
            mol = _h2_at(center, 0.74)
            mol.set_tags(mol_id)
            box += mol

        gas_mol, new_box, name, del_id = deletion_mc(box, "H2", dict(mol_species))
        assert name == "H2"
        assert len(gas_mol) == 2
        assert len(new_box) == 6
        assert del_id not in set(new_box.get_tags())
        counts[del_id] += 1

    # Every molecule must be deletable with roughly equal frequency —
    # uniform 1/n choice is what the deletion acceptance prefactor assumes.
    assert all(c > 0 for c in counts.values())
    # Expected 100 each, std ~8.7; allow a generous band.
    assert all(60 <= c <= 140 for c in counts.values())


def test_deletion_skips_spectators_and_other_species():
    random.seed(5)
    box = Atoms(cell=[15.0, 15.0, 15.0], pbc=[True, True, False])
    h2 = _h2_at((3.0, 3.0, 3.0), 0.74)
    h2.set_tags(1)
    spectator = _h2_at((12.0, 12.0, 3.0), 0.74)
    spectator.set_tags(2)
    box += h2
    box += spectator
    mol_species = {1: "H2", 2: None}  # id 2 is a frozen spectator

    for _ in range(20):
        _, _, _, del_id = deletion_mc(box, "H2", mol_species)
        assert del_id == 1

    with pytest.raises(RuntimeError):
        deletion_mc(box, "N2", mol_species)
