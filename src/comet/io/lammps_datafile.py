from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from ase import Atoms
from ase.data import atomic_masses, atomic_numbers
from scipy.spatial import cKDTree

from comet.physics.velocities import boltzmann_velocity_distribution


def parse_lammps_data(
    file_path: Path | str,
) -> tuple[
    dict[int, tuple[int, np.ndarray, tuple[int, int, int]]],
    dict[int, np.ndarray],
]:
    """Parse the `Atoms` and `Velocities` sections of a LAMMPS data file.

    Args:
        file_path: Path to the input LAMMPS data file.

    Returns:
        tuple: Two dictionaries keyed by atom ID. The first maps IDs to
        `(atom_type, position, image_flags)` records, and the second maps IDs
        to velocity vectors.
    """
    in_atoms = False
    in_vels = False

    atoms_by_id = {}  # id -> (type, pos, image_flags)
    vels_by_id = {}   # id -> vel

    with open(file_path, "r") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue

            # Section headers
            if s.startswith("Atoms"):
                in_atoms = True
                in_vels = False
                continue
            if s.startswith("Velocities"):
                in_atoms = False
                in_vels = True
                continue

            # Stop when another section begins (common LAMMPS style: a single word header)
            is_section_header = re.match(r"^[A-Za-z][A-Za-z0-9_/#-]*\s*$", s)
            if is_section_header and not s.startswith(("Atoms", "Velocities")):
                in_atoms = False
                in_vels = False
                continue

            if in_atoms:
                # Expected: id type x y z ix iy iz  (image flags may be absent)
                parts = re.split(r"\s+", s)
                if len(parts) < 5:
                    continue
                aid = int(parts[0])
                atype = int(parts[1])
                x, y, z = map(float, parts[2:5])

                img = (0, 0, 0)
                if len(parts) >= 8:
                    img = tuple(map(int, parts[5:8]))

                atoms_by_id[aid] = (atype, np.array([x, y, z], dtype=float), img)

            elif in_vels:
                # Expected: id vx vy vz
                parts = re.split(r"\s+", s)
                if len(parts) < 4:
                    continue
                aid = int(parts[0])
                vx, vy, vz = map(float, parts[1:4])
                vels_by_id[aid] = np.array([vx, vy, vz], dtype=float)

    return atoms_by_id, vels_by_id


def format_masses(elements: list[str]) -> str:
    """
    Generate the LAMMPS 'Masses' section from a list of element symbols.
    """
    lines = []
    for i, element in enumerate(elements, start=1):
        Z = atomic_numbers[element]
        mass = atomic_masses[Z]
        lines.append(f"{i} {mass:.6f}")
    return "\n".join(lines)


def element_masses(elements: list) -> list:
    """
    Return per-type atomic masses [amu] for a list of element symbols.

    The result is ordered to match LAMMPS atom types: ``elements[i - 1]`` is the
    mass of type ``i``. Unlike the LAMMPS ``Masses`` section, no type index is
    embedded — callers that write the section add the index themselves.
    """
    return [atomic_masses[atomic_numbers[element]] for element in elements]


def write_lammpsdata(
    num_atoms,
    positions,
    velocities,
    x_cell,
    y_cell,
    z_cell,
    elements: list[str],
) -> str:
    """Assemble a minimal atomic-style LAMMPS data file as a single string.

    Args:
        num_atoms: Number of atoms to report in the header.
        positions: Preformatted `Atoms` section payload.
        velocities: Preformatted `Velocities` section payload.
        x_cell: Upper x cell bound.
        y_cell: Upper y cell bound.
        z_cell: Upper z cell bound.
        elements: Element symbols defining the atom-type order.

    Returns:
        str: Complete LAMMPS data file contents.
    """
    masses_string = format_masses(elements)

    return f"""LAMMPS data file

{num_atoms} atoms
{len(elements)} atom types

0 {x_cell} xlo xhi
0 {y_cell} ylo yhi
0 {z_cell} zlo zhi

Masses

{masses_string}

Atoms # atomic

{positions}

Velocities

{velocities}
"""


def write_string(xyz) -> str:
    """
    Convert a 2D array-like object into a formatted multiline string.
    """
    return "\n".join(
        "     ".join(str(value) for value in row)
        for row in xyz
    )


def build_new_positions_and_velocities(
    old_lammps_path: Path | str,
    new_struct: Atoms,
    elements: List[str],
    T: float,
    tol: float = 1e-4,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build (new_pos, new_vel) for new_struct:
      - new_pos from new_struct positions
      - new_vel copied from old LAMMPS data (by matching type+position within tol),
        otherwise assigned via Maxwell-Boltzmann using atomic masses.

    Args:
      old_lammps_path: Path to the reference LAMMPS data file.
      new_struct: Structure whose positions and velocities are being built.
      elements: Element symbols defining the LAMMPS type order.
      T: Temperature used to generate velocities for unmatched atoms.
      tol: Position-matching tolerance in Angstrom.

    Returns:
      new_pos: (N,3) float
      new_vel: (N,3) float
    """
    atoms_by_id, vels_by_id = parse_lammps_data(old_lammps_path)  # id -> (type, pos, img), id -> vel

    # Build mapping from symbol -> lammps type (1-based)
    sym_to_type = {sym: i + 1 for i, sym in enumerate(elements)}

    new_pos = np.asarray(new_struct.get_positions(), dtype=float)
    new_syms = np.asarray(new_struct.get_chemical_symbols(), dtype=str)
    new_types = np.array([sym_to_type[s] for s in new_syms], dtype=int)

    new_vel = np.zeros((len(new_struct), 3), dtype=float)

    # Build KDTree per LAMMPS type from old positions, storing corresponding old IDs
    trees: Dict[int, Tuple[cKDTree, np.ndarray]] = {}
    old_used = set()

    # collect by type
    pos_by_type: Dict[int, List[np.ndarray]] = {}
    id_by_type: Dict[int, List[int]] = {}
    for aid, (atype, pos, img) in atoms_by_id.items():
        pos_by_type.setdefault(atype, []).append(pos)
        id_by_type.setdefault(atype, []).append(aid)

    for atype, plist in pos_by_type.items():
        P = np.vstack(plist)
        I = np.array(id_by_type[atype], dtype=int)
        trees[atype] = (cKDTree(P), I)

    # Assign velocities
    for i, (atype, p, sym) in enumerate(zip(new_types, new_pos, new_syms)):
        matched = False

        if atype in trees:
            tree, old_ids = trees[atype]
            dist, j = tree.query(p, k=1)
            if np.isfinite(dist) and dist <= tol:
                old_id = int(old_ids[int(j)])
                if old_id not in old_used and old_id in vels_by_id:
                    new_vel[i] = vels_by_id[old_id]   # raw LAMMPS velocity (preserved)
                    old_used.add(old_id)
                    matched = True

        if not matched:
            # New atom: assign velocity based on atomic mass
            m_amu = atomic_masses[atomic_numbers[sym]]
            v = boltzmann_velocity_distribution(T, m_amu)  # returns list-like length 3
            new_vel[i] = np.asarray(v, dtype=float) / 100.0  # keep your convention

    return new_pos, new_vel


def assign_ids_preserve_slab(
    old_atoms_by_id: Dict[int, Tuple[int, np.ndarray, Tuple[int, int, int]]],
    new_struct: Atoms,
    elements: List[str],
    n_slab: int,
    tol: float = 1e-4,
    reuse_old_ids_for_gas: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Assign LAMMPS IDs to `new_struct` while preserving slab IDs exactly.

    The slab is identified positionally: the first ``n_slab`` atoms of
    `new_struct` (the frozen slab/adsorbate block, unchanged by GCMC) keep
    their matched old IDs — the slab may contain any mix of elements,
    including elements shared with the gas. Remaining atoms are renumbered
    into the free IDs in ascending order so the final ID list stays dense
    whenever the preserved slab IDs already fit within `1..N`.

    Args:
      old_atoms_by_id: Parsed old atom records keyed by LAMMPS atom ID.
      new_struct: Structure that needs new LAMMPS IDs (slab block first).
      elements: Element symbols defining the LAMMPS type order.
      n_slab: Number of leading atoms in `new_struct` that form the frozen slab.
      tol: Position-matching tolerance in Angstrom.
      reuse_old_ids_for_gas: Whether to preserve matched gas IDs when possible.

    Returns:
      new_ids: (N,) array of 1-based LAMMPS IDs
      matched_old_id: (N,) array of matched old IDs, or -1 if unmatched
    """
    sym_to_type = {sym: i + 1 for i, sym in enumerate(elements)}

    new_pos = np.asarray(new_struct.get_positions(), dtype=float)
    new_syms = np.asarray(new_struct.get_chemical_symbols(), dtype=str)
    new_types = np.array([sym_to_type[s] for s in new_syms], dtype=int)

    # Build KDTree per type from old positions; store corresponding old IDs
    pos_by_type: Dict[int, List[np.ndarray]] = {}
    id_by_type: Dict[int, List[int]] = {}
    for aid, (atype, pos, img) in old_atoms_by_id.items():
        pos_by_type.setdefault(atype, []).append(pos)
        id_by_type.setdefault(atype, []).append(aid)

    trees: Dict[int, Tuple[cKDTree, np.ndarray]] = {}
    for atype, plist in pos_by_type.items():
        P = np.vstack(plist)
        I = np.array(id_by_type[atype], dtype=int)
        trees[atype] = (cKDTree(P), I)

    matched_old_id = -np.ones(len(new_struct), dtype=int)
    used_old = set()

    # Pass 1: nearest-neighbor matching by (type + position)
    for i, (atype, p) in enumerate(zip(new_types, new_pos)):
        if atype not in trees:
            continue
        tree, old_ids = trees[atype]
        dist, j = tree.query(p, k=1)
        if np.isfinite(dist) and dist <= tol:
            oid = int(old_ids[int(j)])
            if oid not in used_old:
                matched_old_id[i] = oid
                used_old.add(oid)

    # Pass 2: assign IDs with slab preservation
    new_ids = -np.ones(len(new_struct), dtype=int)
    used_new_ids = set()

    # Preserve slab IDs exactly (slab = leading n_slab atoms, element-agnostic)
    if not (0 <= n_slab <= len(new_struct)):
        raise ValueError(f"n_slab={n_slab} outside 0..{len(new_struct)}")
    slab_mask = np.zeros(len(new_struct), dtype=bool)
    slab_mask[:n_slab] = True
    slab_indices = np.where(slab_mask)[0]
    for i in slab_indices:
        oid = int(matched_old_id[i])
        if oid <= 0:
            raise RuntimeError(
                f"Slab atom (symbol={new_syms[i]}) at index {i} could not be matched to an old atom within tol={tol}."
            )
        if oid in used_new_ids:
            raise RuntimeError(f"Duplicate preserved slab ID encountered: {oid}")
        new_ids[i] = oid
        used_new_ids.add(oid)

    # Optionally reuse old IDs for non-slab matched atoms.
    # This preserves historical gas IDs, but can leave gaps after deletions.
    if reuse_old_ids_for_gas:
        for i in np.where(~slab_mask)[0]:
            oid = int(matched_old_id[i])
            if oid > 0 and oid not in used_new_ids:
                new_ids[i] = oid
                used_new_ids.add(oid)

    # Fill remaining gas IDs from the smallest free positive integers.
    # This reuses IDs released by deleted gas atoms and keeps the final list
    # consecutive as long as the preserved slab IDs lie within 1..N.
    next_id = 1
    for i in range(len(new_struct)):
        if new_ids[i] != -1:
            continue
        while next_id in used_new_ids:
            next_id += 1
        new_ids[i] = next_id
        used_new_ids.add(next_id)
        next_id += 1

    return new_ids, matched_old_id


def build_new_velocities_from_matched_ids(
    matched_old_id: np.ndarray,
    old_vels_by_id: Dict[int, np.ndarray],
    new_struct: Atoms,
    T: float,
) -> np.ndarray:
    """
    For each atom in new_struct:
      - if matched_old_id exists in old_vels_by_id: copy velocity
      - else: assign a new MB velocity based on atomic mass
    """
    new_syms = np.asarray(new_struct.get_chemical_symbols(), dtype=str)
    new_vel = np.zeros((len(new_struct), 3), dtype=float)

    for i, oid in enumerate(matched_old_id):
        oid = int(oid)
        if oid > 0 and oid in old_vels_by_id:
            new_vel[i] = old_vels_by_id[oid]  # raw LAMMPS velocity preserved
        else:
            sym = new_syms[i]
            m_amu = atomic_masses[atomic_numbers[sym]]
            v = boltzmann_velocity_distribution(T, m_amu)  # returns list length 3
            new_vel[i] = np.asarray(v, dtype=float) / 100.0  # keep your convention

    return new_vel


def write_lammps_data_atomic_with_ids(
    out_path: Path,
    cell,
    elements: List[str],
    masses_by_type: List[float],
    new_ids: np.ndarray,
    new_struct: Atoms,
    new_vel: np.ndarray,
) -> None:
    """Write an atomic-style LAMMPS data file using caller-supplied IDs.

    Args:
        out_path: Destination path for the LAMMPS data file.
        cell: Simulation cell used to write the box bounds.
        elements: Element symbols defining the atom-type order.
        masses_by_type: Mass records aligned with `elements`.
        new_ids: One-based LAMMPS atom IDs aligned with `new_struct`.
        new_struct: Structure to write in the `Atoms` section.
        new_vel: Velocity array aligned with `new_ids` and `new_struct`.
    """
    out_path = Path(out_path)
    N = len(new_struct)
    if new_vel.shape != (N, 3):
        raise ValueError("new_vel must be shape (N,3)")
    if new_ids.shape != (N,):
        raise ValueError("new_ids must be shape (N,)")

    # type mapping
    sym_to_type = {sym: i + 1 for i, sym in enumerate(elements)}
    types = np.array([sym_to_type[s] for s in new_struct.get_chemical_symbols()], dtype=int)
    pos = np.asarray(new_struct.get_positions(), dtype=float)

    # Box bounds (orthorhombic)
    cell = np.asarray(cell)
    xhi, yhi, zhi = cell[0, 0], cell[1, 1], cell[2, 2]

    # Sort by ID for neatness (LAMMPS doesn't require it, but it's nice)
    order = np.argsort(new_ids)
    ids = new_ids[order].astype(int)
    types = types[order]
    pos = pos[order]
    vel = new_vel[order]

    with open(out_path, "w") as f:
        f.write("LAMMPS data file\n\n")
        f.write(f"{N} atoms\n")
        f.write(f"{len(elements)} atom types\n\n")

        f.write(f"0.0 {xhi:.15g} xlo xhi\n")
        f.write(f"0.0 {yhi:.15g} ylo yhi\n")
        f.write(f"0.0 {zhi:.15g} zlo zhi\n\n")

        f.write("Masses\n\n")
        for t, m in enumerate(masses_by_type, start=1):
            f.write(f"{t} {float(m):.6f}\n")
        f.write("\n")

        f.write("Atoms # atomic\n\n")
        # id type x y z ix iy iz
        for aid, atype, (x, y, z) in zip(ids, types, pos):
            f.write(f"{aid:d} {atype:d} {x:.15g} {y:.15g} {z:.15g} 0 0 0\n")
        f.write("\n")

        f.write("Velocities\n\n")
        # id vx vy vz
        for aid, (vx, vy, vz) in zip(ids, vel):
            f.write(f"{aid:d} {vx:.15g} {vy:.15g} {vz:.15g}\n")
        f.write("\n")
