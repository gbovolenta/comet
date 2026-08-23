"""GCMC workflow stages: build the initial system, run the MC loop, write output.

These are the units `run()` orchestrates. State is carried between them in a
single :class:`GcmcState`, and the energy backend is injected (see
:mod:`comet.potentials.backends`) so the whole pipeline can be driven with a
stub in tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from ase import Atoms

from comet.config.schema import RunConfig
from comet.io.lammps_datafile import (
    assign_ids_preserve_slab,
    build_new_velocities_from_matched_ids,
    element_masses,
    parse_lammps_data,
    write_lammps_data_atomic_with_ids,
)
from comet.io.region import bounds_from_restart
from comet.io.trajectory import get_last_frame, write_extxyz_sequence
from comet.mc.accept import metropolis_criteria, metropolis_probability
from comet.mc.moves import (
    choose_biased_move,
    choose_unbiased_move,
    deletion_mc,
    insertion_mc,
    load_gas_templates,
)
from comet.physics.thermo import (
    chemical_potentials_from_particles,
    compute_chemical_potentials,
    compute_pressure_atm,
    count_convergence_status,
    quantized_target_counts,
    target_counts_from_mu,
)
from comet.potentials.backends import EnergyBackend
from comet.potentials.templates import load_centered_gas_templates
from comet.system.molecules import partition_by_molecule
from comet.workflows.logging_utils import (
    _fmt_energy,
    _fmt_int,
    _fmt_mu_scalar,
    _fmt_prob,
    _fmt_pressure,
    _format_mu_dict,
    _format_mu_status,
    logger,
)

@dataclass
class GcmcState:
    """Mutable physical state threaded through the workflow stages."""

    box_gas: Atoms                       # gas-region structure (updated on accept)
    slab_ads: Atoms                      # frozen slab + adsorbates (below z_cutoff)
    gas_count: int                       # total gas-molecule count
    gas_counts: Dict[str, int]           # per-species molecule counts
    E_current: float                     # current energy [eV]
    V: float                             # gas-region volume [Å³]
    T: float                             # temperature [K]
    mu_dict: Dict[str, float]            # target chemical potentials [eV]
    mu_current: Dict[str, float]         # current chemical potentials [eV]
    gas_en_dict: Dict[str, float]        # per-species reference energies [eV]
    region_bounds: Tuple[float, ...]     # GCMC insertion bounds
    gas_templates_all: Dict[str, Atoms]  # master per-species templates
    gas_dict: Dict[str, float]           # per-species masses [amu]
    mol_species: Dict[int, Optional[str]]  # molecule id -> species (None = spectator)
    next_mol_id: int                     # first unused molecule id
    n_targets: Dict[str, int]            # integer convergence targets per species


def _gas_masses(config: RunConfig, gas_templates: Dict[str, Atoms]) -> Dict[str, float]:
    """Per-species molecular masses [amu], derived from template composition.

    A `gas_masses` list in the config is only cross-checked (it is redundant
    with the template): a mismatch beyond 0.05 amu logs a warning and the
    template-derived value wins.
    """
    gas_dict = {name: float(t.get_masses().sum()) for name, t in gas_templates.items()}
    if config.gas_masses is not None:
        for name, declared in zip(config.gas_list, config.gas_masses):
            if abs(declared - gas_dict[name]) > 0.05:
                logger.warning(
                    "gas_masses[%s] = %.4f amu differs from the template-derived mass "
                    "%.4f amu; using the template value.",
                    name, declared, gas_dict[name],
                )
    return gas_dict


def build_initial_system(config: RunConfig, backend: EnergyBackend) -> Optional[GcmcState]:
    """Load the restart, extract the gas box, and compute initial energetics.

    Returns ``None`` (after logging a critical message) if the trajectory or the
    initial energy cannot be obtained; otherwise returns the populated state.
    """
    region_bounds = bounds_from_restart(config.restart_path, config.z_cutoff)
    gas_templates = load_gas_templates(config.gas_list, config.gas_template_dir)
    gas_dict = _gas_masses(config, gas_templates)

    try:
        atoms = get_last_frame(config.restart_path)
    except Exception as e:  # pragma: no cover - explicit runtime guard
        logger.critical(f"Cannot load trajectory: {e}")
        return None

    # Split into gas box and slab by molecule COM: whole molecules are assigned
    # to one side, so per-species counts are integral by construction and no
    # cutoff adjustment is needed. Gas molecules are tagged with persistent ids.
    z_cut = float(config.z_cutoff)
    box_gas, slab_ads, gas_counts, mol_species, next_mol_id = partition_by_molecule(
        atoms, z_cut, gas_templates,
    )
    gas_count = sum(gas_counts.values())
    logger.info(
        "Molecule partition at z_cutoff = %.2f Å: %s (gas box: %d atoms, slab: %d atoms)",
        z_cut, gas_counts, len(box_gas), len(slab_ads),
    )

    # Initial energy of the gas box.
    try:
        E_current = backend.energy(box_gas, "initial_box")
    except Exception as e:  # pragma: no cover - explicit runtime guard
        logger.critical(f"Energy calculation failed: {e}")
        return None

    V = atoms.cell[0][0] * atoms.cell[1][1] * (atoms.cell[2][2] - z_cut)
    T = config.temperature

    mu_dict = compute_chemical_potentials(
        T,
        gas_dict,
        pressure=config.pressure,
        pressure_unit=config.pressure_unit,
        y1=config.y1,
        partial_pressures=config.partial_pressures,
    )
    logger.info("Target chemical potentials: %s", _format_mu_dict(mu_dict))

    # Integer convergence targets nearest the ideal-gas expectations; with
    # `ratios` the composition is preserved exactly. Fails (no fallback) when
    # the gas volume cannot accommodate the requested state.
    n_star = target_counts_from_mu(T, V, mu_dict, gas_dict)
    try:
        n_targets = quantized_target_counts(T, V, mu_dict, gas_dict, ratios=config.ratios)
    except ValueError as e:
        logger.critical(str(e))
        return None
    logger.info(
        "Expected counts N* = pV/kBT (V=%.0f Å³): %s",
        V,
        {g: round(n, 2) for g, n in n_star.items()},
    )
    logger.info(
        "Integer convergence targets: %s (implied pressures [atm]: %s)",
        {g: n for g, n in n_targets.items() if np.isfinite(mu_dict[g])},
        {g: round(compute_pressure_atm(T, V, n), 2)
         for g, n in n_targets.items() if np.isfinite(mu_dict[g])},
    )

    inactive = [g for g, mu in mu_dict.items() if not np.isfinite(mu)]
    if inactive:
        logger.info(
            "Inactive species (μ_target = -inf, no insertion expected): %s",
            inactive,
        )

    # NOTE: gas templates are intentionally NOT filtered to active species.
    # Move selection already draws only from the unconverged (active) set via
    # mu_convergence_status, so inactive species are never inserted/deleted; the
    # frozen molecules must remain available for counting, logging and overlap.
    gas_templates_all = load_centered_gas_templates(
        gas_template_dir=config.gas_template_dir,
        gas_list=config.gas_list,
    )
    gas_en_dict = backend.gas_energies(gas_templates_all)
    logger.debug(gas_en_dict)

    logger.info(
        "Initial gas-molecule count: %s, Initial energy: %s eV",
        _fmt_int(gas_count),
        _fmt_energy(E_current),
    )
    logger.info("Initial gas counts by species: %s", gas_counts)

    mu_current = chemical_potentials_from_particles(T, V, gas_counts, gas_dict)

    return GcmcState(
        box_gas=box_gas,
        slab_ads=slab_ads,
        gas_count=gas_count,
        gas_counts=gas_counts,
        E_current=E_current,
        V=V,
        T=T,
        mu_dict=mu_dict,
        mu_current=mu_current,
        gas_en_dict=gas_en_dict,
        region_bounds=region_bounds,
        gas_templates_all=gas_templates_all,
        gas_dict=gas_dict,
        mol_species=mol_species,
        next_mol_id=next_mol_id,
        n_targets=n_targets,
    )


def run_mc_loop(state: GcmcState, config: RunConfig, backend: EnergyBackend) -> GcmcState:
    """Run the GCMC move loop, mutating and returning the state."""
    logger.info("Starting GCMC loop with integer-count termination")
    output_extxyz = Path(config.bdir) / "mc_cycle.extxyz"

    # Local working copies of the mutable state.
    box_gas = state.box_gas
    gas_count = state.gas_count
    gas_counts = state.gas_counts
    E_current = state.E_current
    mu_current = state.mu_current
    V, T = state.V, state.T
    mu_dict = state.mu_dict
    gas_en_dict = state.gas_en_dict
    gas_dict = state.gas_dict
    region_bounds = state.region_bounds
    gas_templates_all = state.gas_templates_all
    mol_species = state.mol_species
    next_mol_id = state.next_mol_id
    n_targets = state.n_targets

    inactive, converged, unconverged = count_convergence_status(n_targets, gas_counts, mu_dict)
    logger.info("Initial μ_current: %s", _format_mu_dict(mu_current))
    if config.log_mu_diagnostics:
        logger.info("μ diagnostics: %s", _format_mu_status(mu_dict, mu_current))
    logger.info("Converged: %s | Unconverged: %s | Inactive: %s",
                sorted(converged), sorted(unconverged), sorted(inactive))

    run_until_converged = config.run_until_converged
    max_steps = int(config.max_steps)
    n_steps = int(config.steps)
    biased_moves = config.biased_moves

    step_iter = range(1, (max_steps if run_until_converged else n_steps) + 1)
    stop_reason = "loop not entered"
    step = 0  # guard: range may be empty (steps/max_steps == 0)

    for step in step_iter:

        if not unconverged:
            logger.info("All active species converged at step %d", step)
            stop_reason = "all active species converged"
            break

        try:
            # Decide move
            if biased_moves:
                move_name, ins = choose_biased_move(
                    n_targets=n_targets,
                    gas_counts=gas_counts,
                    unconverged=unconverged,
                    force_single_species=config.force_single_species,
                )
                if move_name is None:
                    logger.info("No unconverged species left at step %d", step)
                    stop_reason = "no unconverged species left"
                    break
                gas_templates_active = {move_name: gas_templates_all[move_name]}
            else:
                move_name, ins = choose_unbiased_move(unconverged, gas_counts)
                if move_name is None:
                    logger.info("No unconverged species left at step %d", step)
                    stop_reason = "no unconverged species left"
                    break
                gas_templates_active = {move_name: gas_templates_all[move_name]}

            # Propose move
            del_id = None
            if ins:
                new_mol, new_box, ins_name = insertion_mc(
                    gas_templates_active, region_bounds, box_gas, mol_id=next_mol_id,
                )
                move_name = ins_name
            else:
                if gas_counts.get(move_name, 0) == 0:
                    logger.info("Step %d proposal skipped: no %s molecules available for deletion",
                                step, move_name)
                    mu_current = chemical_potentials_from_particles(T, V, gas_counts, gas_dict)
                    inactive, converged, unconverged = count_convergence_status(n_targets, gas_counts, mu_dict)
                    if config.log_mu_diagnostics:
                        logger.info("μ diagnostics: %s", _format_mu_status(mu_dict, mu_current))
                    logger.info("Converged: %s | Unconverged: %s | Inactive: %s",
                                sorted(converged), sorted(unconverged), sorted(inactive))
                    continue
                try:
                    del_mol, new_box, move_name, del_id = deletion_mc(box_gas, move_name, mol_species)
                except RuntimeError:
                    # fallback to insertion (same active templates)
                    logger.info("Step %d deletion proposal failed; falling back to insertion.", step)
                    new_mol, new_box, ins_name = insertion_mc(
                        gas_templates_active, region_bounds, box_gas, mol_id=next_mol_id,
                    )
                    move_name = ins_name
                    ins = True

            # Energy of proposal
            E_new = backend.energy(new_box, f"step_{step}_{move_name}_{'ins' if ins else 'del'}")

            # Metropolis acceptance using moved species
            mass_X = gas_dict[move_name]
            mu_X = mu_dict[move_name]
            gas_en_X = gas_en_dict[move_name]
            prob, delta_E = metropolis_probability(E_current, E_new, T, mass_X, mu_X, ins, V, gas_count, gas_en_X)

            logger.info(
                "Step %d proposal (%s %s): E_current=%s eV, E_new=%s eV, "
                "delta_E=%s eV, mu_target=%s eV, gas_ref=%s eV, prob=%s",
                step,
                "ins" if ins else "del",
                move_name,
                _fmt_energy(E_current),
                _fmt_energy(E_new),
                _fmt_energy(delta_E),
                _fmt_mu_scalar(mu_X),
                _fmt_energy(gas_en_X),
                _fmt_prob(prob),
            )

            accepted = metropolis_criteria(E_current, E_new, T, mass_X, mu_X, ins, V, gas_count, gas_en_X)

            if accepted:
                box_gas = new_box
                E_current = E_new

                # Update molecule bookkeeping incrementally: the moves return
                # the species, and molecule identity lives in tags/mol_species.
                if ins:
                    mol_species[next_mol_id] = move_name
                    next_mol_id += 1
                    gas_counts[move_name] = gas_counts.get(move_name, 0) + 1
                else:
                    mol_species.pop(del_id)
                    gas_counts[move_name] -= 1

                gas_count = sum(gas_counts.values())

                p_step = compute_pressure_atm(T, V, gas_count)
                logger.info("Step %d accepted (%s %s): n_gas=%s, E=%s eV, P=%s atm",
                            step, "ins" if ins else "del", move_name, _fmt_int(gas_count), _fmt_energy(E_current), _fmt_pressure(p_step))
                logger.info("Step %d accepted gas counts by species: %s", step, gas_counts)
            else:
                p_step = compute_pressure_atm(T, V, gas_count)
                logger.info(
                    "Step %d rejected (%s %s): prob=%s, n_gas=%s, E=%s eV, P=%s atm",
                    step,
                    "ins" if ins else "del",
                    move_name,
                    _fmt_prob(prob),
                    _fmt_int(gas_count),
                    _fmt_energy(E_current),
                    _fmt_pressure(p_step),
                )

            # Update classification and μ diagnostics every step (accepted or not)
            mu_current = chemical_potentials_from_particles(T, V, gas_counts, gas_dict)
            inactive, converged, unconverged = count_convergence_status(n_targets, gas_counts, mu_dict)
            if config.log_mu_diagnostics:
                logger.info("μ diagnostics: %s", _format_mu_status(mu_dict, mu_current))

            if step == 1 or step % int(config.log_every) == 0:
                logger.info("Converged: %s | Unconverged: %s | Inactive: %s",
                            sorted(converged), sorted(unconverged), sorted(inactive))

            write_extxyz_sequence(output_extxyz, box_gas)

        except Exception:
            # Log the full traceback (not just the message) so genuine bugs are
            # visible rather than disguised as "a step failed", then stop
            # gracefully so write_restart still persists the last good state.
            logger.exception("MC step %d failed; stopping loop", step)
            stop_reason = f"exception during step {step}"
            break

    else:
        # Only runs if loop exhausted without break
        if run_until_converged and unconverged:
            logger.warning("Reached max_steps=%d without full convergence. Unconverged: %s",
                           max_steps, sorted(unconverged))
            stop_reason = f"reached max_steps={max_steps} without convergence"
        else:
            stop_reason = f"reached fixed step limit {n_steps}"

    logger.info("GCMC loop finished at step %d: %s", step, stop_reason)

    # Persist the mutated working copies back into the state.
    state.box_gas = box_gas
    state.gas_count = gas_count
    state.gas_counts = gas_counts
    state.E_current = E_current
    state.mu_current = mu_current
    state.mol_species = mol_species
    state.next_mol_id = next_mol_id
    return state


def write_restart(state: GcmcState, config: RunConfig) -> None:
    """Merge slab + gas and write the next-cycle LAMMPS restart, with final logs."""
    new_struct = state.slab_ads + state.box_gas

    old_atoms_by_id, old_vels_by_id = parse_lammps_data(config.restart_path)

    new_ids, matched_old_id = assign_ids_preserve_slab(
        old_atoms_by_id=old_atoms_by_id,
        new_struct=new_struct,
        elements=config.elements,
        n_slab=len(state.slab_ads),
        tol=1e-4,
        reuse_old_ids_for_gas=False,
    )

    new_vel = build_new_velocities_from_matched_ids(
        matched_old_id=matched_old_id,
        old_vels_by_id=old_vels_by_id,
        new_struct=new_struct,
        T=state.T,
    )

    bdir = Path(config.bdir)
    write_lammps_data_atomic_with_ids(
        out_path=bdir / "initial.lammpsdata",
        cell=new_struct.cell,
        elements=config.elements,
        masses_by_type=element_masses(config.elements),
        new_ids=new_ids,
        new_struct=new_struct,
        new_vel=new_vel,
    )

    p_fin = compute_pressure_atm(state.T, state.V, state.gas_count)
    logger.info(
        "Finished GCMC. Final gas molecules count: %s, Final energy: %s eV",
        _fmt_int(state.gas_count),
        _fmt_energy(state.E_current),
    )
    logger.info("Final gas counts by species: %s", state.gas_counts)
    if config.log_mu_diagnostics:
        logger.info("Final μ diagnostics: %s", _format_mu_status(state.mu_dict, state.mu_current))

    logger.info("Finished GCMC. Final pressure: %s atm.", _fmt_pressure(p_fin))
    logger.info("Wrote %s", bdir / "initial.lammpsdata")
    logger.info("Wrote %s", bdir / "mc_cycle.extxyz")
