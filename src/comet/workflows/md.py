"""ASE-MD stage for `comet cycle`: GCMC <-> MD alternation without LAMMPS.

Mirrors the production LAMMPS recipe (NVE + CSVR thermostat on the mobile
atoms, frozen bottom slab layers, reflective lid at the cell top) using ASE's
Bussi dynamics and the same calculator instance the GCMC energies come from —
so both halves of the cycle run on one PES.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from ase import units
from ase.constraints import FixAtoms
from ase.md.bussi import Bussi
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary

from comet.config.schema import RunConfig
from comet.io.trajectory import write_extxyz_sequence
from comet.physics.thermo import chemical_potentials_from_particles
from comet.potentials.backends import EnergyBackend
from comet.system.molecules import partition_by_molecule
from comet.workflows.logging_utils import _fmt_energy, logger
from comet.workflows.stages import GcmcState


def _reflect_top(atoms, z_top: float) -> None:
    """Reflect atoms crossing the cell top (LAMMPS `wall/reflect zhi`).

    z is the non-periodic direction; without a lid, gas would escape upward.
    Called as a dynamics observer every step, so crossings are shallow.
    """
    pos = atoms.get_positions()
    over = pos[:, 2] > z_top
    if over.any():
        mom = atoms.get_momenta()
        pos[over, 2] = 2.0 * z_top - pos[over, 2]
        mom[over, 2] = -np.abs(mom[over, 2])
        atoms.set_positions(pos)
        atoms.set_momenta(mom)


def write_cycle_checkpoint(state: GcmcState, config: RunConfig, path: Path) -> None:
    """Write the current slab+gas state as a LAMMPS data file.

    Unlike :func:`comet.workflows.stages.write_restart`, this does not match
    against the original restart file (positions have moved during MD): atoms
    are written slab-block-first with fresh sequential ids and their CURRENT
    velocities (converted from ASE units to LAMMPS ``metal`` Å/ps).
    """
    from comet.io.lammps_datafile import element_masses, write_lammps_data_atomic_with_ids

    new_struct = state.slab_ads + state.box_gas
    ids = np.arange(1, len(new_struct) + 1)
    vel_metal = new_struct.get_velocities() * units.fs * 1000.0  # ASE -> Å/ps
    write_lammps_data_atomic_with_ids(
        out_path=path,
        cell=new_struct.cell,
        elements=config.elements,
        masses_by_type=element_masses(config.elements),
        new_ids=ids,
        new_struct=new_struct,
        new_vel=vel_metal,
    )
    logger.info("Wrote cycle checkpoint %s", path)


def run_md(
    state: GcmcState,
    config: RunConfig,
    calculator,
    backend: EnergyBackend,
    cycle_index: int = 0,
) -> GcmcState:
    """Run one MD segment on slab+gas, then re-partition into a fresh state.

    Velocities are re-initialized from a Maxwell-Boltzmann distribution each
    segment (the thermostat re-equilibrates within a few tau_t anyway; carried
    velocities would be undefined for freshly inserted molecules).
    """
    md = config.md
    T = state.T

    full = state.slab_ads + state.box_gas
    full.calc = calculator

    z = full.get_positions()[:, 2]
    z_floor = z.min() + float(md.freeze_bottom)
    frozen = np.flatnonzero(z < z_floor)
    if len(frozen) == 0:
        logger.warning(
            "freeze_bottom=%.2f Å freezes no atoms; the slab is free to drift",
            md.freeze_bottom,
        )
    else:
        full.set_constraint(FixAtoms(indices=frozen))

    MaxwellBoltzmannDistribution(full, temperature_K=T, rng=np.random)
    Stationary(full)
    momenta = full.get_momenta()
    momenta[frozen] = 0.0
    full.set_momenta(momenta)

    z_top = float(full.cell[2][2])
    dyn = Bussi(
        full,
        timestep=float(md.timestep_fs) * units.fs,
        temperature_K=T,
        taut=float(md.tau_t_ps) * 1000.0 * units.fs,
        rng=np.random,
    )
    dyn.attach(lambda: _reflect_top(full, z_top), interval=1)
    if md.traj_every > 0:
        traj_path = Path(config.bdir) / "md_cycle.extxyz"
        dyn.attach(lambda: write_extxyz_sequence(traj_path, full), interval=md.traj_every)

    logger.info(
        "MD segment (cycle %d): %d steps x %.2f fs, Bussi tau=%.3f ps, "
        "%d frozen / %d mobile atoms, T=%.1f K",
        cycle_index, md.md_steps, md.timestep_fs, md.tau_t_ps,
        len(frozen), len(full) - len(frozen), T,
    )
    dyn.run(md.md_steps)
    logger.info(
        "MD segment done: T=%.1f K, E_pot=%s eV",
        full.get_temperature(),
        _fmt_energy(full.get_potential_energy()),
    )

    # Re-partition: molecules may have adsorbed (COM below z_cutoff -> frozen
    # slab side) or desorbed; recognition also quarantines any MD-induced
    # fragments as spectators. Fresh molecule ids are assigned.
    full.calc = None
    box_gas, slab_ads, gas_counts, mol_species, next_mol_id = partition_by_molecule(
        full, float(config.z_cutoff), state.gas_templates_all,
    )

    state.box_gas = box_gas
    state.slab_ads = slab_ads
    state.gas_counts = gas_counts
    state.gas_count = sum(gas_counts.values())
    state.mol_species = mol_species
    state.next_mol_id = next_mol_id
    state.E_current = backend.energy(box_gas, f"cycle_{cycle_index}_post_md")
    state.mu_current = chemical_potentials_from_particles(
        state.T, state.V, state.gas_counts, state.gas_dict,
    )
    return state
