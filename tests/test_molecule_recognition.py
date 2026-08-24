"""Molecule recognition and COM-based gas/slab partitioning."""

import sys
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from comet.system.molecules import partition_by_molecule, template_compositions

# A reasonable gas-phase CH3OH geometry (Å).
MEOH = Atoms(
    "COHHHH",
    [
        [-0.046423, 0.662445, 0.000000],
        [-0.046423, -0.758170, 0.000000],
        [0.986955, 1.021560, 0.000000],
        [-0.541793, 1.062351, 0.889607],
        [-0.541793, 1.062351, -0.889607],
        [0.864827, -1.058866, 0.000000],
    ],
)

H2O = Atoms("OHH", [[0.0, 0.0, 0.0], [0.7572, 0.5865, 0.0], [-0.7572, 0.5865, 0.0]])
H2 = Atoms("HH", [[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]])


def _place(mol: Atoms, center) -> Atoms:
    out = mol.copy()
    out.positions += np.asarray(center) - out.get_center_of_mass()
    return out


def _cu_slab() -> Atoms:
    pos = [[x, y, z] for z in (0.0, 1.8) for x in (1.0, 6.0) for y in (1.0, 6.0)]
    return Atoms("Cu8", pos)


def test_partition_mixed_polyatomic_gas_over_metal_slab():
    system = _cu_slab()
    system += _place(MEOH, (3.0, 3.0, 15.0))
    system += _place(MEOH, (8.0, 8.0, 20.0))
    system += _place(H2O, (3.0, 8.0, 18.0))
    system += _place(H2, (8.0, 3.0, 22.0))
    system += _place(MEOH, (5.0, 5.0, 4.0))   # adsorbed: COM below cutoff -> frozen slab side
    system.set_cell([12.0, 12.0, 30.0])
    system.set_pbc([True, True, False])

    templates = {"CH3OH": MEOH.copy(), "H2O": H2O.copy(), "H2": H2.copy()}
    box_gas, slab_ads, gas_counts, mol_species, next_mol_id = partition_by_molecule(
        system, 10.0, templates
    )

    assert gas_counts == {"CH3OH": 2, "H2O": 1, "H2": 1}
    assert len(box_gas) == 6 + 6 + 3 + 2
    assert len(slab_ads) == 8 + 6          # slab + adsorbed CH3OH stay frozen
    assert next_mol_id == 5
    assert sorted(mol_species.values()) == ["CH3OH", "CH3OH", "H2", "H2O"]

    # Tags group atoms into molecules of the right size.
    tags = box_gas.get_tags()
    sizes = {t: int((tags == t).sum()) for t in set(tags)}
    by_species = {mol_species[t]: n for t, n in sizes.items()}
    assert by_species["H2"] == 2 and by_species["H2O"] == 3 and by_species["CH3OH"] == 6
    assert set(slab_ads.get_tags()) == {0}


def test_partition_assigns_straddling_molecule_by_com():
    system = _cu_slab()
    straddler = Atoms("HH", [[3.0, 3.0, 9.9], [3.0, 3.0, 10.64]])  # COM above z=10
    system += straddler
    system.set_cell([12.0, 12.0, 30.0])
    system.set_pbc([True, True, False])

    box_gas, slab_ads, gas_counts, mol_species, _ = partition_by_molecule(
        system, 10.0, {"H2": H2.copy()}
    )

    # The whole molecule lands in the gas box (including the atom below the plane).
    assert gas_counts == {"H2": 1}
    assert len(box_gas) == 2
    assert len(slab_ads) == 8


def test_partition_flags_unrecognized_fragment_as_spectator(caplog):
    system = _cu_slab()
    system += Atoms("H", [[3.0, 3.0, 15.0]])       # lone H atom: matches no template
    system += _place(H2, (8.0, 8.0, 20.0))
    system.set_cell([12.0, 12.0, 30.0])
    system.set_pbc([True, True, False])

    with caplog.at_level("WARNING"):
        box_gas, _, gas_counts, mol_species, _ = partition_by_molecule(
            system, 10.0, {"H2": H2.copy()}
        )

    assert gas_counts == {"H2": 1}                 # spectator excluded from counts
    assert len(box_gas) == 3                       # but present in the gas box
    assert None in mol_species.values()
    assert "spectator" in caplog.text


def test_template_compositions_rejects_isomers():
    ethanol_like = Atoms("COHHHH", MEOH.get_positions())  # same formula as CH3OH
    with pytest.raises(ValueError, match="composition"):
        template_compositions({"CH3OH": MEOH.copy(), "other": ethanol_like})


def test_partition_supports_slab_sharing_elements_with_gas():
    # "Ice-like" O/H slab under H2O gas: slab and gas share ALL elements.
    slab_pos = [[x, y, z] for z in (0.0, 1.5) for x in (1.0, 5.0) for y in (1.0, 5.0)]
    system = Atoms("O8", slab_pos)
    system += Atoms("H16", [[p[0] + 0.6, p[1], p[2] + 0.4] for p in slab_pos]
                    + [[p[0] - 0.6, p[1], p[2] + 0.4] for p in slab_pos])
    system += _place(H2O, (3.0, 3.0, 15.0))
    system.set_cell([8.0, 8.0, 25.0])
    system.set_pbc([True, True, False])

    box_gas, slab_ads, gas_counts, mol_species, _ = partition_by_molecule(
        system, 8.0, {"H2O": H2O.copy()}
    )

    assert gas_counts == {"H2O": 1}
    assert len(box_gas) == 3
    assert len(slab_ads) == 24
