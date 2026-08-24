"""Molecule recognition and molecule-aware gas/slab partitioning.

Molecules are identified ONCE, when the restart is loaded: atoms are clustered
into fragments by covalent-bond connectivity and each fragment is matched to a
gas template by element composition. From then on identity is carried on the
gas box as per-atom integer tags (ASE tags = molecule ids) plus a
``{mol_id: species}`` mapping — it is never re-derived from element symbols or
bond lengths. Any molecular formula works: species are not assumed diatomic,
homonuclear, or distinct from the slab elements.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from ase import Atoms
from ase.data import atomic_numbers, covalent_radii
from ase.neighborlist import neighbor_list
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

logger = logging.getLogger(__name__)

# Baseline bond criterion: covalent-radii sum scaled by this factor. Template
# bonds override it (see _system_pair_cutoffs), so this only has to hold the
# slab together and detect bonds in equilibrium template geometries.
_RADII_MULT = 1.35


def composition_key(symbols) -> Tuple[str, ...]:
    """Order-independent composition key for a set of chemical symbols."""
    return tuple(sorted(symbols))


def template_compositions(gas_templates: Dict[str, Atoms]) -> Dict[Tuple[str, ...], str]:
    """Map element composition -> species name; compositions must be unique.

    Species are recognized by composition, so two templates with the same
    formula (isomers) cannot be distinguished and are rejected up front.
    """
    comp_to_species: Dict[Tuple[str, ...], str] = {}
    for name, templ in gas_templates.items():
        key = composition_key(templ.get_chemical_symbols())
        if key in comp_to_species:
            raise ValueError(
                f"Gas templates '{comp_to_species[key]}' and '{name}' have the same "
                f"composition {key}. Species are identified by composition, so "
                "isomers cannot be distinguished in one run."
            )
        comp_to_species[key] = name
    return comp_to_species


def _bonded_template_pairs(templ: Atoms) -> List[Tuple[str, str, float]]:
    """Return (sym_a, sym_b, distance) for each bonded pair in a template.

    A pair is bonded when its distance is below the scaled covalent-radii sum;
    on an equilibrium template geometry this is reliable, and it excludes
    non-bonded intramolecular pairs (e.g. H···H in CH3OH at ~1.8 Å).
    """
    syms = templ.get_chemical_symbols()
    pos = templ.get_positions()
    pairs = []
    for i in range(len(templ)):
        for j in range(i + 1, len(templ)):
            d = float(np.linalg.norm(pos[i] - pos[j]))
            r_sum = covalent_radii[atomic_numbers[syms[i]]] + covalent_radii[atomic_numbers[syms[j]]]
            if d <= r_sum * _RADII_MULT:
                pairs.append((syms[i], syms[j], d))
    return pairs


def _system_pair_cutoffs(
    atoms: Atoms,
    gas_templates: Dict[str, Atoms],
    bond_tol: float = 0.25,
) -> Dict[Tuple[str, str], float]:
    """Per-element-pair bond cutoffs for fragment detection on the full system.

    Baseline is the scaled covalent-radii sum for every element pair present.
    Pairs that are bonded within a gas template are overridden with the actual
    template bond length * (1 + bond_tol), which is robust to thermal bond
    stretching (covalent radii alone leave essentially no margin for H2).
    """
    symbols = sorted(set(atoms.get_chemical_symbols()))
    cutoffs: Dict[Tuple[str, str], float] = {}
    for a in symbols:
        for b in symbols:
            r_sum = covalent_radii[atomic_numbers[a]] + covalent_radii[atomic_numbers[b]]
            cutoffs[(a, b)] = r_sum * _RADII_MULT

    for name, templ in gas_templates.items():
        for sym_a, sym_b, bond in _bonded_template_pairs(templ):
            cut = bond * (1.0 + float(bond_tol))
            for key in ((sym_a, sym_b), (sym_b, sym_a)):
                cutoffs[key] = max(cutoffs.get(key, 0.0), cut)

    return cutoffs


def find_fragments(
    atoms: Atoms,
    gas_templates: Dict[str, Atoms],
    bond_tol: float = 0.25,
) -> List[np.ndarray]:
    """Connected components of the bond graph (PBC-aware in x/y).

    Returns a list of index arrays, one per fragment, in ascending-index order.
    """
    n = len(atoms)
    if n == 0:
        return []
    cutoffs = _system_pair_cutoffs(atoms, gas_templates, bond_tol=bond_tol)
    i_idx, j_idx = neighbor_list("ij", atoms, cutoff=cutoffs)
    adjacency = coo_matrix(
        (np.ones(len(i_idx), dtype=np.int8), (i_idx, j_idx)), shape=(n, n)
    )
    n_frag, labels = connected_components(adjacency, directed=False)
    return [np.flatnonzero(labels == f) for f in range(n_frag)]


def partition_by_molecule(
    atoms: Atoms,
    z_cutoff: float,
    gas_templates: Dict[str, Atoms],
    bond_tol: float = 0.25,
) -> Tuple[Atoms, Atoms, Dict[str, int], Dict[int, Optional[str]], int]:
    """Split a structure into gas box and slab by molecule center of mass.

    Each bonded fragment is assigned wholly to the gas box (fragment COM z
    above ``z_cutoff``) or to the slab (COM at or below) — molecules are never
    split at the boundary, so per-species counts are integral by construction.
    Gas fragments are matched to templates by composition and tagged with a
    persistent molecule id (ASE tags, 1-based). Unmatched gas fragments are
    kept as frozen spectators (species ``None``) with a warning.

    Args:
        atoms: Full structure from the restart file.
        z_cutoff: Gas-region boundary along z (non-periodic direction).
        gas_templates: Mapping from species name to ASE template.
        bond_tol: Fractional tolerance on template bond lengths.

    Returns:
        tuple: ``(box_gas, slab_ads, gas_counts, mol_species, next_mol_id)``.
        ``box_gas`` carries molecule ids as tags; ``slab_ads`` tags are 0;
        ``gas_counts`` counts template species only; ``mol_species`` maps
        molecule id to species name (or ``None`` for spectators);
        ``next_mol_id`` is the first unused molecule id.
    """
    atoms = atoms.copy()
    atoms.set_pbc([True, True, False])

    comp_to_species = template_compositions(gas_templates)
    fragments = find_fragments(atoms, gas_templates, bond_tol=bond_tol)

    symbols = np.array(atoms.get_chemical_symbols())
    masses = atoms.get_masses()
    z = atoms.get_positions()[:, 2]

    gas_idx: List[int] = []
    gas_tags: List[int] = []
    slab_idx: List[int] = []
    mol_species: Dict[int, Optional[str]] = {}
    next_mol_id = 1

    for frag in fragments:
        m = masses[frag]
        com_z = float(np.dot(z[frag], m) / m.sum())
        if com_z <= z_cutoff:
            slab_idx.extend(int(i) for i in frag)
            continue

        species = comp_to_species.get(composition_key(symbols[frag]))
        if species is None:
            logger.warning(
                "Unrecognized gas-region fragment %s (composition %s, COM z=%.2f Å): "
                "kept as a frozen spectator (never inserted/deleted, excluded from "
                "counts), but it still contributes to the energy and overlap checks.",
                frag.tolist(),
                dict(zip(*np.unique(symbols[frag], return_counts=True))),
                com_z,
            )
        mol_species[next_mol_id] = species
        gas_idx.extend(int(i) for i in frag)
        gas_tags.extend([next_mol_id] * len(frag))
        next_mol_id += 1

    box_gas = atoms[gas_idx]
    box_gas.set_tags(gas_tags)
    slab_ads = atoms[sorted(slab_idx)]
    slab_ads.set_tags(0)

    gas_counts = {name: 0 for name in gas_templates}
    for species in mol_species.values():
        if species is not None:
            gas_counts[species] += 1

    n_spectators = sum(1 for s in mol_species.values() if s is None)
    # Summary lines live with the callers (stages/md); keep DEBUG detail here.
    logger.debug("Gas counts by species: %s", gas_counts)
    logger.debug(
        "Total gas molecule count: %d (+ %d frozen spectator fragment(s))",
        sum(gas_counts.values()),
        n_spectators,
    )

    return box_gas, slab_ads, gas_counts, mol_species, next_mol_id
