"""Partition helpers for separating vacuum regions."""

from __future__ import annotations

import logging
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np
from ase import Atoms
from ase.neighborlist import neighbor_list


logger = logging.getLogger(__name__)


def extract_box_gas(
    atoms: Atoms,
    z_cutoff: float,
    gas_templates: Dict[str, Atoms],   # e.g. {"H2": Atoms(...), "N2": Atoms(...), "CO": Atoms(...)}
    bond_tol: float = 0.25,            # cutoff = bond_length * (1 + bond_tol)
) -> Tuple[Atoms, Atoms, int, Dict[str, int], List[int], List[int], dict]:
    """
    Split system into gas box (above z_cutoff) and slab (below),
    and count diatomic molecules by gas name for all gases in gas_templates.

    Supports homo- and heteronuclear diatomics. Counting is PBC-aware in x/y.

    Returns:
        box_gas
        slab_ads
        gas_count_total
        gas_counts (dict: {gas_name: count})
        idx_in
        idx_out
        xyz_dict
    """
    positions = atoms.get_positions()
    symbols = atoms.get_chemical_symbols()

    xyz_dict = {i: pos for i, pos in enumerate(positions)}

    idx_in, idx_out, coords_in, coords_out = [], [], [], []
    for i, pos in enumerate(positions):
        if pos[2] > z_cutoff:
            idx_in.append(i)
            coords_in.append(pos)
        else:
            idx_out.append(i)
            coords_out.append(pos)

    gas_symbols = [symbols[i] for i in idx_in]

    box_gas = Atoms(gas_symbols, coords_in, cell=atoms.cell, pbc=[True, True, False])
    slab_ads = Atoms([symbols[i] for i in idx_out], coords_out, cell=atoms.cell, pbc=[True, True, False])

    # --- Build pair cutoffs from templates (order-independent element pair key) ---
    def pair_key(a: str, b: str) -> Tuple[str, str]:
        return (a, b) if a <= b else (b, a)

    # map element-pair -> gas_name (if multiple names share same pair, last wins)
    pair_to_gasname: Dict[Tuple[str, str], str] = {}
    cutoff_by_pair: Dict[Tuple[str, str], float] = {}

    for gas_name, templ in gas_templates.items():
        if len(templ) != 2:
            raise ValueError(f"Template for {gas_name} must have exactly 2 atoms.")
        a, b = templ.get_chemical_symbols()
        bond = float(np.linalg.norm(templ.positions[0] - templ.positions[1]))
        key = pair_key(a, b)

        pair_to_gasname[key] = gas_name
        cutoff_by_pair[key] = bond * (1.0 + float(bond_tol))

    if not cutoff_by_pair:
        raise ValueError("No gas templates provided.")

    # Initialize counts for all gases listed
    gas_counts: Dict[str, int] = {g: 0 for g in gas_templates.keys()}

    # --- Identify diatomic molecules in box_gas using MIC distances ---
    r_max = max(cutoff_by_pair.values())
    if len(box_gas) >= 2:
        i_idx, j_idx, d_ij = neighbor_list("ijd", box_gas, cutoff=r_max)

        # Deduplicate (neighbor_list yields directed pairs)
        mask = i_idx < j_idx
        i_idx, j_idx, d_ij = i_idx[mask], j_idx[mask], d_ij[mask]

        # We must ensure each atom is assigned to at most one diatomic molecule
        used = np.zeros(len(box_gas), dtype=bool)

        # Prefer shortest bonds first (robust in dense gas)
        order = np.argsort(d_ij)
        box_syms = np.array(box_gas.get_chemical_symbols())

        for k in order:
            i, j = int(i_idx[k]), int(j_idx[k])
            if used[i] or used[j]:
                continue

            a, b = box_syms[i], box_syms[j]
            key = pair_key(a, b)
            cut = cutoff_by_pair.get(key)
            if cut is None:
                continue
            if float(d_ij[k]) <= cut:
                gas_name = pair_to_gasname[key]
                gas_counts[gas_name] += 1
                used[i] = True
                used[j] = True

        # If you want a warning when unpaired atoms remain:
        unpaired = int((~used).sum())
        if unpaired != 0:
            logger.warning("Unpaired gas atoms in region: %d", unpaired)

    gas_count_total = int(sum(gas_counts.values()))
    logger.info("Gas counts by species: %s", gas_counts)
    logger.info("Total gas molecule count: %d", gas_count_total)

    return box_gas, slab_ads, gas_count_total, gas_counts, idx_in, idx_out, xyz_dict


def filter_active_gas_templates(gas_templates: dict, mu_target: dict) -> dict:
    """
    Return a new gas_templates dict containing only active species
    (mu_target finite).
    """
    active_templates = {
        gas: templ
        for gas, templ in gas_templates.items()
        if np.isfinite(mu_target.get(gas, np.nan))
    }

    removed = set(gas_templates) - set(active_templates)
    if removed:
        logger.info("Removed inactive gas templates: %s", sorted(removed))

    return active_templates


def per_species_atom_counts(box: Atoms, gas_dict: dict) -> dict:
    """Count atoms in `box` for each homonuclear diatomic species label.

    Args:
        box: Gas-region structure.
        gas_dict: Mapping from gas label to molecular mass.

    Returns:
        dict: Atom counts keyed by gas-species label.
    """
    atom_counts = Counter(box.get_chemical_symbols())
    out = {}
    for gas in gas_dict:          # e.g. "H2", "N2"
        elem = gas[:-1]           # "H", "N" (homonuclear assumption)
        out[gas] = atom_counts.get(elem, 0)
    return out


def all_integral_diatomics(counts: dict) -> bool:
    """Return whether every homonuclear diatomic count is an even integer.

    Args:
        counts: Atom counts keyed by gas-species label.

    Returns:
        bool: `True` when each count is divisible by two.
    """
    # homonuclear diatomics: need even number of atoms for each gas
    return all((n % 2 == 0) for n in counts.values())
