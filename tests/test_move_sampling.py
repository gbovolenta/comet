"""Sampling correctness of the MC move proposals.

Insertion orientations must be uniform on SO(3) and deletion must pick a
molecule uniformly at random — both are assumed by the GCMC acceptance rule
(detailed balance).
"""

import random
import sys
from pathlib import Path

import numpy as np
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
        new_mol, new_box, name = insertion_mc(templates, bounds, box)
        assert name == "N2"
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
    templates = {"H2": Atoms("HH", [[0, 0, 0], [0, 0, 0.74]])}
    # Four well-separated H2 molecules with distinct bond lengths, all within
    # the deletion cutoff (0.74 * 1.25 = 0.925 Å).
    centers = [(3.0, 3.0, 3.0), (3.0, 12.0, 3.0), (12.0, 3.0, 3.0), (12.0, 12.0, 3.0)]
    bonds = [0.70, 0.72, 0.74, 0.76]

    n_trials = 400
    counts = [0, 0, 0, 0]
    for _ in range(n_trials):
        box = Atoms(cell=[15.0, 15.0, 15.0], pbc=[True, True, False])
        for center, bond in zip(centers, bonds):
            box += _h2_at(center, bond)

        _, new_box, name = deletion_mc(box, templates)
        assert name == "H2"
        assert len(new_box) == 6

        remaining = new_box.get_positions()
        gone = [
            m
            for m, center in enumerate(centers)
            if not np.any(np.linalg.norm(remaining - np.array(center), axis=1) < 1.0)
        ]
        assert len(gone) == 1
        counts[gone[0]] += 1

    # Every molecule must be deletable — in particular the choice must not
    # collapse onto the shortest bond (the old prefer-closest behavior, which
    # broke detailed balance).
    assert counts[0] < n_trials
    assert all(c > 0 for c in counts)
    # Roughly uniform: expected 100 each, std ~8.7; allow a generous band.
    assert all(60 <= c <= 140 for c in counts)
