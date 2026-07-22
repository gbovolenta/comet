"""Generate the binary H2/N2 seed from the H2-only seed.

Adds N2 molecules on a deterministic grid in the gas region of
fe_slab_h2.lammps and writes fe_slab_h2_n2.lammps with atom types
ordered [Fe, N, H] (must match the config's `elements`).

Run from this directory with an ase-equipped python:
    python gen_mixture_seed.py
"""

import numpy as np
from ase import Atoms
from ase.io.lammpsdata import read_lammps_data, write_lammps_data

N2_BOND = 1.098
SPECORDER = ["Fe", "N", "H"]

base = read_lammps_data("fe_slab_h2.lammps", sort_by_id=False, read_image_flags=False)

# Deterministic grid of N2 centers, well above the slab and clear of cell edges.
centers = [
    (4.0, 4.0, 28.0), (12.0, 4.0, 30.0), (19.0, 4.0, 32.0),
    (4.0, 12.0, 34.0), (12.0, 12.0, 36.0), (19.0, 12.0, 28.5),
]
for cx, cy, cz in centers:
    n2 = Atoms("NN", [[cx, cy, cz - N2_BOND / 2], [cx, cy, cz + N2_BOND / 2]])
    d = np.linalg.norm(base.positions - n2.get_center_of_mass(), axis=1)
    assert d.min() > 2.5, f"N2 at {(cx, cy, cz)} overlaps an existing atom ({d.min():.2f} A)"
    base += n2

# Rebuild a clean Atoms object: reading a LAMMPS file leaves a per-atom
# `type` array on the object, which write_lammps_data prefers over
# `specorder` — appended atoms would silently get type 0.
clean = Atoms(
    symbols=base.get_chemical_symbols(),
    positions=base.get_positions(),
    cell=base.cell,
    pbc=base.pbc,
    momenta=base.get_momenta(),
)

write_lammps_data(
    "fe_slab_h2_n2.lammps", clean,
    specorder=SPECORDER, atom_style="atomic", masses=True, velocities=True,
)
syms = np.array(clean.get_chemical_symbols())
print("wrote fe_slab_h2_n2.lammps:",
      {s: int((syms == s).sum()) for s in SPECORDER})
