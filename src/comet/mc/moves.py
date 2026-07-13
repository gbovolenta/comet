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
from scipy.spatial.transform import Rotation

logger = logging.getLogger(__name__)


def _random_rotation_matrix() -> np.ndarray:
    """Rotation matrix drawn uniformly from SO(3) (Haar measure).

    A quaternion with i.i.d. normal components, normalized to the unit
    3-sphere, gives a uniformly distributed rotation. Sequential uniform
    Euler angles do NOT (orientations cluster near the poles), and the GCMC
    acceptance rule assumes trial orientations are proposed uniformly.

    Draws from the global numpy RNG so the `seed` config option keeps runs
    reproducible.
    """
    q = np.random.normal(size=4)
    q /= np.linalg.norm(q)
    return Rotation.from_quat(q).as_matrix()


def insertion_mc(
    gas_templates: Dict[str, Atoms],
    bounds: Tuple[float, float, float, float, float, float],
    box_gas: Atoms,
    mol_id: int,
    min_dist: float = 2.5,
    max_attempts: int = 50,
) -> Tuple[Atoms, Atoms, str]:
    """
    Attempt to insert a gas molecule into the GCMC box without overlap (PBC-aware in x,y).

    The template may contain any number of atoms; it is inserted rigid, with a
    uniformly random orientation about its center of mass. All atoms of the new
    molecule are tagged with ``mol_id`` so it can later be deleted (or counted)
    as a unit without re-deriving connectivity.

    Args:
        gas_templates: Dict of templates, e.g. {"H2": Atoms(...), "CH3OH": Atoms(...)}.
        bounds: Region bounds as (x0, x1, y0, y1, z0, z1).
        box_gas: Current gas box (atoms tagged with molecule ids).
        mol_id: Molecule id stamped on the inserted atoms (caller-owned counter).
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

        # Rotate the COM-centered template uniformly, then place the COM.
        rotated = coords @ _random_rotation_matrix().T
        new = Atoms(symbols, rotated + offset, cell=box_gas.cell, pbc=[True, True, False])
        new.set_tags(mol_id)

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
    Load gas templates from files named H2.xyz, CH3OH.xyz, etc.

    Args:
        gas_list: Iterable of gas-species labels to load.
        template_dir: Directory containing one `.xyz` file per species.

    Returns:
        dict: Mapping from gas label to a COM-centered ASE template.
    """
    template_dir = Path(template_dir)
    gas_templates = {}

    for gas in gas_list:
        path = template_dir / f"{gas}.xyz"
        if not path.exists():
            raise FileNotFoundError(path)

        templ = read(path)
        if len(templ) < 1:
            raise ValueError(f"Template {path} contains no atoms")

        # Ensure centered (important!)
        templ.positions -= templ.get_center_of_mass()
        gas_templates[gas] = templ

    return gas_templates


def deletion_mc(
    box_gas: Atoms,
    species: str,
    mol_species: Dict[int, Optional[str]],
) -> Tuple[Atoms, Atoms, str, int]:
    """
    Delete one molecule of ``species``, chosen uniformly among those present.

    Molecules are identified by their per-atom tags (molecule ids) — no bond
    re-matching — so any molecular formula works. Uniform choice is required
    for detailed balance: the deletion acceptance prefactor n/V assumes the
    deleted molecule was selected with probability 1/n.

    Args:
        box_gas: Current gas box, atoms tagged with molecule ids.
        species: Species name to delete.
        mol_species: Mapping from molecule id to species name (``None`` entries
                     are frozen spectators and never chosen). The caller pops
                     the returned id from this mapping once the move is accepted.

    Returns:
        (gas_mol, updated_box_gas, species, deleted_mol_id)

    Raises:
        RuntimeError if no molecule of ``species`` is present.
    """
    candidates = [m for m, s in mol_species.items() if s == species]
    if not candidates:
        raise RuntimeError(f"Deletion failed: no '{species}' molecules present")

    chosen = random.choice(candidates)
    mask = box_gas.get_tags() == chosen
    if not mask.any():
        raise RuntimeError(
            f"Inconsistent state: molecule id {chosen} ('{species}') has no atoms in the gas box"
        )

    gas_mol = box_gas[mask]
    remaining = box_gas[~mask]

    logger.debug(
        "Deleted %s (mol_id %d, %d atoms); %d atoms remain",
        species, chosen, int(mask.sum()), len(remaining),
    )
    return gas_mol, remaining, species, int(chosen)


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
