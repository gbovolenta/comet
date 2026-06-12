"""Monte Carlo move proposals."""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from ase import Atoms
from ase.io import read
from ase.neighborlist import neighbor_list
from scipy.spatial import cKDTree

logger = logging.getLogger(__name__)


def insertion_mc(
    gas_templates: Dict[str, Atoms],
    bounds: Tuple[float, float, float, float, float, float],
    box_gas: Atoms,
    min_dist: float = 2.5,
    max_attempts: int = 50,
) -> Tuple[Atoms, Atoms, str]:
    """
    Attempt to insert a gas molecule into the GCMC box without overlap (PBC-aware in x,y).

    Args:
        gas_templates: Dict of templates, e.g. {"H2": Atoms(...), "N2": Atoms(...)} (each should be diatomic).
        bounds: Region bounds as (x0, x1, y0, y1, z0, z1).
        box_gas: Current gas box.
        min_dist: Minimum allowed distance in Å (MIC in x/y).
        max_attempts: Maximum insertion attempts.

    Returns:
        (new_mol, updated_box_gas, name of inserted gas mol)

    Raises:
        RuntimeError: If no valid insertion found.
    """
    if not gas_templates:
        raise ValueError("gas_templates is empty")

    x0, x1, y0, y1, z0, z1 = bounds
    names = list(gas_templates.keys())

    # Ensure box has correct PBC/cell context for MIC distances
    # (cell must already be meaningful on box_gas)
    box_gas.set_pbc([True, True, False])

    for attempt in range(1, max_attempts + 1):
        # Pick which molecule to insert (uniformly)
        name = random.choice(names)
        templ = gas_templates[name]

        # Center template so offset defines COM position
        coords = templ.get_positions() - templ.get_center_of_mass()
        symbols = templ.get_chemical_symbols()

        offset = np.array([
            random.uniform(x0, x1),
            random.uniform(y0, y1),
            random.uniform(z0, z1),
        ])

        new = Atoms(symbols, coords + offset, cell=box_gas.cell, pbc=[True, True, False])
        new.rotate(random.uniform(0, 360), 'x', center='COM')
        new.rotate(random.uniform(0, 360), 'y', center='COM')
        new.rotate(random.uniform(0, 360), 'z', center='COM')

        # If box is empty, accept immediately
        if len(box_gas) == 0:
            logger.debug("Insertion successful (empty box, %s) on attempt %d", name, attempt)
            return new, box_gas + new, name

        # PBC-aware overlap check: any pair closer than min_dist?
        combo = box_gas + new
        # Indices: old atoms [0..n_old-1], new atoms [n_old..n_old+n_new-1]
        n_old = len(box_gas)
        n_new = len(new)
        i, j, d = neighbor_list("ijd", combo, cutoff=min_dist)

        # Look for *cross* pairs: one in old, one in new
        cross = ((i < n_old) & (j >= n_old)) | ((j < n_old) & (i >= n_old))
        if not np.any(cross):
            logger.debug("Insertion successful (%s) on attempt %d", name, attempt)
            return new, combo, name

    logger.warning("Insertion failed after %d attempts", max_attempts)
    raise RuntimeError("Insertion failed")


def load_gas_templates(gas_list, template_dir):
    """
    Load diatomic gas templates from files named H2.xyz, N2.xyz, etc.

    Args:
        gas_list: Iterable of gas-species labels to load.
        template_dir: Directory containing one `.xyz` file per species.

    Returns:
        dict: Mapping from gas label to centered two-atom ASE template.
    """
    template_dir = Path(template_dir)
    gas_templates = {}

    for gas in gas_list:
        path = template_dir / f"{gas}.xyz"
        if not path.exists():
            raise FileNotFoundError(path)

        templ = read(path)
        if len(templ) != 2:
            raise ValueError(f"Template {path} must contain exactly 2 atoms")

        # Ensure centered (important!)
        templ.positions -= templ.get_center_of_mass()
        gas_templates[gas] = templ

    return gas_templates


Pair = Tuple[str, str]


def _bond_length_from_template(template: Atoms) -> float:
    """Return the bond length of a two-atom gas template."""
    if len(template) != 2:
        raise ValueError("Each gas template must contain exactly 2 atoms (a diatomic).")
    p = template.get_positions()
    return float(np.linalg.norm(p[0] - p[1]))


def _pair_key(a: str, b: str) -> Pair:
    """Canonical (order-independent) key for an element pair."""
    return (a, b) if a <= b else (b, a)


def deletion_mc(
    box_gas: Atoms,
    gas_templates: Dict[str, Atoms],
    bond_tol: float = 0.25,
    prefer_closest: bool = True,
) -> Tuple[Atoms, Atoms]:
    """
    Delete a single diatomic molecule from box_gas, supporting homo- and heteronuclear diatomics.
    PBC-aware via ASE neighbor_list (minimum-image distances).

    Args:
        box_gas: Current gas box (mixture of diatomic molecules).
        gas_templates: Dict mapping molecule label -> 2-atom ASE Atoms template.
                       Labels are informational; element identity is taken from the template itself.
        bond_tol: Fractional tolerance added to template bond length to define cutoff.
        prefer_closest: If True, delete the closest valid diatomic pair; otherwise deletes first found.

    Returns:
        (gas_mol, updated_box_gas)

    Raises:
        RuntimeError if no deletable molecule is found.
    """
    if len(box_gas) < 2:
        raise RuntimeError("Deletion failed: box_gas has fewer than 2 atoms")

    symbols = np.array(box_gas.get_chemical_symbols())

    # Build cutoff map for allowed element pairs from templates
    cutoff_by_pair: Dict[Pair, float] = {}
    for name, templ in gas_templates.items():
        syms = templ.get_chemical_symbols()
        if len(syms) != 2:
            raise ValueError(f"Template '{name}' must contain exactly 2 atoms.")
        a, b = syms[0], syms[1]
        bond = _bond_length_from_template(templ)
        cutoff_by_pair[_pair_key(a, b)] = bond * (1.0 + float(bond_tol))

    if not cutoff_by_pair:
        raise ValueError("No gas templates provided.")

    r_max = max(cutoff_by_pair.values())

    # PBC-aware neighbor search (MIC distances)
    i_idx, j_idx, d_ij = neighbor_list("ijd", box_gas, cutoff=r_max)

    # neighbor_list returns directed pairs; keep i<j to avoid duplicates
    mask = i_idx < j_idx
    i_idx = i_idx[mask]
    j_idx = j_idx[mask]
    d_ij = d_ij[mask]

    best = None  # (dist, i, j, pair_key)
    for i, j, d in zip(i_idx, j_idx, d_ij):
        si, sj = symbols[i], symbols[j]
        key = _pair_key(si, sj)
        cut = cutoff_by_pair.get(key)
        if cut is None:
            continue
        if d <= cut:
            if not prefer_closest:
                best = (float(d), int(i), int(j), key)
                break
            if best is None or d < best[0]:
                best = (float(d), int(i), int(j), key)

    if best is None:
        logger.warning(
            "No valid diatomic molecule found for deletion using templates %s (r_max=%.3f Å)",
            list(gas_templates.keys()),
            r_max,
        )
        raise RuntimeError("Deletion failed")

    dist, i, j, key = best
    pos = box_gas.get_positions()

    # Preserve the actual element order as it appears in the box
    gas_mol = Atoms([symbols[i], symbols[j]], [pos[i], pos[j]])

    # Remove atoms i and j from the box
    keep = [k not in (i, j) for k in range(len(box_gas))]

    logger.debug(
        "Deleted diatomic %s-%s indices (%d, %d) at MIC distance %.3f Å (cutoff %.3f Å)",
        symbols[i], symbols[j], i, j, dist, cutoff_by_pair[key]
    )
    return gas_mol, box_gas[keep]


def choose_unbiased_move(
    unconverged: set[str],
    gas_counts: dict[str, int],
) -> tuple[str | None, bool | None]:
    """Choose a species and insertion/deletion direction without μ-based bias.

    Args:
        unconverged: Iterable of species names still outside the μ tolerance.
        gas_counts: Mapping from species name to current molecule count.

    Returns:
        tuple: `(species_name, insert_flag)` or `(None, None)` when there are
        no unconverged species.
    """
    gases_all = list(unconverged)
    if not gases_all:
        return None, None

    g = random.choice(gases_all)
    if gas_counts.get(g, 0) == 0:
        return g, True
    return g, (random.random() < 0.5)


def choose_biased_move(
    mu_target: dict[str, float],
    mu_current: dict[str, float],
    unconverged: set[str],
    gas_counts: dict[str, int],
    force_dmu_threshold: float = 0.0,
    force_single_species: bool = True,
) -> tuple[str | None, bool | None]:
    """Choose a species and move direction using chemical-potential mismatch.

    Args:
        mu_target: Mapping from species name to target chemical potential.
        mu_current: Mapping from species name to current chemical potential.
        unconverged: Iterable of species names still outside the μ tolerance.
        gas_counts: Mapping from species name to current molecule count.
        force_dmu_threshold: Threshold above which the move direction is forced
            by the sign of `Δμ`.
        force_single_species: When `True`, a single unconverged species is
            always chosen directly.

    Returns:
        tuple: `(species_name, insert_flag)` or `(None, None)` when there are
        no unconverged species.
    """
    gases_all = list(unconverged)
    if not gases_all:
        return None, None

    # If only one species remains and you want to push it, always pick it.
    if force_single_species and len(gases_all) == 1:
        g = gases_all[0]
        if gas_counts.get(g, 0) == 0:
            return g, True
        current = mu_current[g]
        target = mu_target[g]
        if not np.isfinite(current):
            return g, True
        dmu = float(target - current)
        if np.isfinite(dmu) and abs(dmu) >= float(force_dmu_threshold):
            return g, (dmu > 0.0)
        return g, (random.random() < 0.5)

    # Species absent from the current gas box should be inserted first.
    force_insert = [
        g for g in gases_all
        if np.isfinite(mu_target[g]) and not np.isfinite(mu_current[g])
    ]
    if force_insert:
        g = random.choice(force_insert)
        return g, True

    # Choose species stochastically, weighted by |Δμ|.
    dmu_map = {}
    for g in gases_all:
        dmu_map[g] = float(mu_target[g] - mu_current[g])

    finite_gases = [g for g in gases_all if np.isfinite(dmu_map[g])]
    if not finite_gases:
        g = random.choice(gases_all)
        if gas_counts.get(g, 0) == 0:
            return g, True
        return g, (random.random() < 0.5)

    w = np.array([abs(dmu_map[g]) for g in finite_gases], dtype=float)
    wsum = float(w.sum())
    if (not np.isfinite(wsum)) or wsum <= 0.0:
        g = random.choice(finite_gases)
    else:
        w = w / wsum
        g = random.choices(finite_gases, weights=w.tolist(), k=1)[0]

    dmu = dmu_map[g]

    # If the chosen species is absent, insertion is the only valid move.
    if gas_counts.get(g, 0) == 0:
        return g, True

    # Deterministic direction when the chemical-potential mismatch is large enough.
    if abs(dmu) >= float(force_dmu_threshold):
        return g, (dmu > 0.0)

    # Otherwise fall back to an unbiased direction for the selected species.
    return g, (random.random() < 0.5)
