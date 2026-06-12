"""Workflow runner ported from the legacy :mod:`comet.py` script."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from comet.config.io import load_config


def setup_logging() -> logging.Logger:
    """
    Configure and return a logger that writes DEBUG-level messages to the file 'gcmc_run.log'
    and INFO-level messages to the console. All log entries exclude timestamps.

    Safe to call more than once per process: handlers (and therefore the
    'gcmc_run.log' file) are created only on the first call. This is invoked from
    `run()` rather than at import time so merely importing the package does not
    create a log file.

    Returns:
        logging.Logger: The configured logger instance.
    """
    log = logging.getLogger("gcmc")
    log.setLevel(logging.DEBUG)

    # Already configured (e.g. a previous run() in this process) — don't add
    # duplicate handlers or truncate the existing log.
    if log.handlers:
        return log

    # Formatter without timestamps
    formatter = logging.Formatter("%(levelname)s: %(message)s")

    # File handler (DEBUG+)
    fh = logging.FileHandler("gcmc_run.log", mode="w")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    log.addHandler(fh)

    # Console handler (INFO+)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    log.addHandler(ch)

    return log


# Module-level logger handle. Handlers (and the log file) are attached lazily by
# setup_logging(), which run() calls — importing this module must not write files.
logger = logging.getLogger("gcmc")


def _prepare_paths(config: dict) -> dict:
    """Convert configured filesystem paths to `Path` objects and set defaults.

    Args:
        config: Raw configuration mapping loaded from YAML.

    Returns:
        dict: Configuration copy with path-like entries converted to `Path`
        objects.
    """
    config = dict(config)
    for key in (
        "bdir",
        "model_dir",
        "gas_path",
        "restart_path",
        "box_path",
        "gas_template_dir",
        "orca_work_root",
    ):
        if key in config and config[key] is not None:
            config[key] = Path(config[key])
    if config.get("bdir") is None:
        config["bdir"] = Path.cwd()
    return config


def _log_run_header(log: logging.Logger, args: dict, energy_backend: str) -> None:
    """Write a compact summary of the current run settings to the log.

    Args:
        log: Logger receiving the run summary.
        args: Prepared workflow configuration.
        energy_backend: Name of the active energy backend.

    Returns:
        None
    """
    unit = args.get("pressure_unit", "bar")
    if args.get("partial_pressures") is not None:
        pressure_desc = f"partial_pressures={args['partial_pressures']} {unit}"
    else:
        pressure_desc = f"{args.get('pressure')} {unit}"
    log.info(
        "Run header: backend=%s, bdir=%s, temperature=%.3f K, pressure=%s, "
        "run_until_converged=%s, steps=%s, max_steps=%s, biased_moves=%s",
        energy_backend,
        args["bdir"],
        args["temperature"],
        pressure_desc,
        bool(args.get("run_until_converged", False)),
        args.get("steps"),
        args.get("max_steps"),
        bool(args.get("biased_moves", False)),
    )
    log.info("Log settings: log_mu_diagnostics=%s", bool(args.get("log_mu_diagnostics", False)))
    if energy_backend == "orca":
        log.info(
            "ORCA settings: method=%s, basis=%s, nprocs=%s, charge=%s, mult=%s",
            args.get("orca_method", "PBE"),
            args.get("orca_basis", "def2-SVP"),
            args.get("orca_nprocs", 1),
            args.get("orca_charge", 0),
            args.get("orca_mult", 1),
        )


def _fmt_int(value) -> str:
    """Format an integer-like value for human-readable log output."""
    return f"{int(value)}"


def _fmt_energy(value: float) -> str:
    """Format an energy value in eV for the log."""
    return f"{float(value):.3f}"


def _fmt_prob(value: float) -> str:
    """Format an acceptance probability for the log."""
    return f"{float(value):.2f}"


def _fmt_pressure(value: float) -> str:
    """Format a pressure value for the log."""
    return f"{float(value):.2f}"


def _fmt_mu_scalar(value: float) -> str:
    """Format a chemical potential value or an inactive sentinel for the log."""
    if np.isfinite(value):
        return f"{float(value):.3f}"
    return "inactive"


def _format_mu_dict(mu_dict: dict) -> str:
    """Render a species-to-μ mapping as a compact single-line string.

    Args:
        mu_dict: Mapping from species name to chemical potential.

    Returns:
        str: Formatted dictionary-like string for logging.
    """
    return "{" + ", ".join(f"{gas}: {_fmt_mu_scalar(mu)}" for gas, mu in mu_dict.items()) + "}"


def _format_mu_status(mu_target: dict, mu_current: dict) -> str:
    """Render detailed per-species target/current μ diagnostics for logging.

    Args:
        mu_target: Mapping from species name to target chemical potential.
        mu_current: Mapping from species name to current chemical potential.

    Returns:
        str: Semicolon-separated diagnostic string.
    """
    fields = []
    for gas in mu_target:
        target = mu_target[gas]
        current = mu_current.get(gas, float("nan"))
        if np.isfinite(target) and np.isfinite(current):
            delta = target - current
            fields.append(
                f"{gas}: target={_fmt_mu_scalar(target)} current={_fmt_mu_scalar(current)} dmu={_fmt_mu_scalar(delta)}"
            )
        elif np.isfinite(target) and not np.isfinite(current):
            fields.append(
                f"{gas}: target={_fmt_mu_scalar(target)} current=no molecules dmu=n/a"
            )
        elif not np.isfinite(target):
            fields.append(
                f"{gas}: target=inactive current={_fmt_mu_scalar(current)} dmu=n/a"
            )
        else:
            fields.append(f"{gas}: target=unknown current=unknown dmu=n/a")
    return "; ".join(fields)


def run(config_path: str) -> int:
    """Execute the workflow defined by the configuration file."""

    from collections import Counter
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
        load_gas_masses,
        mu_convergence_status,
    )
    from comet.potentials.templates import load_centered_gas_templates
    from comet.system.partition import (
        all_integral_diatomics,
        extract_box_gas,
        per_species_atom_counts,
    )

    setup_logging()  # attach handlers / create gcmc_run.log now, not at import
    args = _prepare_paths(load_config(config_path))
    Path(args["bdir"]).mkdir(parents=True, exist_ok=True)
    energy_backend = str(args.get("energy_backend", "mace")).lower()
    _log_run_header(logger, args, energy_backend)

    if energy_backend == "mace":
        from comet.potentials.mace import get_energy_mace, compute_gas_energies

        def energy_fn(atoms, label):
            return get_energy_mace(atoms, args["model_dir"])

        def gas_energy_fn(templates):
            return compute_gas_energies(templates, args["model_dir"])
    elif energy_backend == "orca":
        from comet.potentials.orca import get_energy_orca, compute_gas_energies_orca

        def energy_fn(atoms, label):
            return get_energy_orca(atoms, args, label=label)

        def gas_energy_fn(templates):
            return compute_gas_energies_orca(templates, args)
    else:
        raise ValueError(f"Unsupported energy backend: {energy_backend}")

    # 2. Preload region bounds and gas template
    #region_bounds = read_region(args["box_path"])
    #box_path = get_last_frame(args["box_path"])
    region_bounds = bounds_from_restart(args["restart_path"], args["z_cutoff"])
    #gas_template = read(str(args["gas_path"]))
    gas_dict = load_gas_masses(args["gas_list"], args["gas_masses"])
    gas_templates = load_gas_templates(args["gas_list"], args["gas_template_dir"])
    #logger.debug(gas_templates)


    # 3. Load last trajectory frame
    try:
        atoms = get_last_frame(args["restart_path"])
    except Exception as e:  # pragma: no cover - explicit runtime guard
        logger.critical(f"Cannot load trajectory: {e}")
        return 1


    restart_dict = parse_lammps_data(args["restart_path"])
    atoms_by_id, vels_by_id = parse_lammps_data(args["restart_path"])
    # 4. Extract gas box and surface
    #box_gas, slab_ads, gas_count, idx_in, idx_out, xyz_dict, n_slab = extract_box_gas(
    #    atoms, args["z_cutoff"]
    #)
    logger.info("Extracted Atoms and Velocities blocks from old restart file.")

    elements = args["elements"]   # ["Fe", "N", "H"]
    # Build mapping: symbol -> index (1-based)
    elem_to_id = {sym: i + 1 for i, sym in enumerate(elements)}

    # initial extract
    z_cut = float(args["z_cutoff"])
    box_gas, slab_ads, gas_count, gas_counts, idx_in, idx_out, xyz_dict = extract_box_gas(
        atoms,
        z_cut,
        gas_templates=gas_templates,
    )
    
    z_step = 0.1
    max_adjust = 5.0   # safety: max 5 Å shift (adjust if you like)
    
    counts = per_species_atom_counts(box_gas, gas_dict)
    if all_integral_diatomics(counts):
        logger.info(
            "All gas species have integral molecule counts at z_cutoff = %.2f Å. Counts: %s",
            z_cut,
            {g: n // 2 for g, n in counts.items()},
        )
    else:
        logger.warning(
            "Non-integral molecule count detected at z_cutoff = %.2f Å. Atom counts: %s. Adjusting cutoff.",
            z_cut,
            counts,
        )

        moved = 0.0
        while (not all_integral_diatomics(counts)) and (moved <= max_adjust):
            # move cutoff downward (includes a bit more atoms into gas region)
            z_cut -= z_step
            moved += z_step

            box_gas, slab_ads, gas_count, gas_counts, idx_in, idx_out, xyz_dict = extract_box_gas(
                atoms,
                z_cut,
                gas_templates=gas_templates,
            )
            counts = per_species_atom_counts(box_gas, gas_dict)

            logger.info(
                "Adjusted z_cutoff = %.2f Å → atom counts: %s → molecule counts: %s",
                z_cut,
                counts,
                {g: n // 2 for g, n in counts.items()},
            )

        if not all_integral_diatomics(counts):
            logger.warning(
                "Failed to achieve integral per-species counts after shifting z_cutoff by %.2f Å "
                "(final z_cutoff = %.2f Å). Final atom counts: %s",
                moved,
                z_cut,
                counts,
            )
        else:
            logger.info(
                "Integral per-species molecule counts achieved at z_cutoff = %.2f Å. Molecules: %s",
                z_cut,
                {g: n // 2 for g, n in counts.items()},
            )

    # If you want the rest of the script to use the adjusted cutoff:
    args["z_cutoff"] = z_cut
    old_gas_pos = atoms.get_positions()[idx_in]   # shape (N_old, 3)
    
    
    #    # update counts after accept
    #symbols = box_gas.get_chemical_symbols()
    #atom_counts = Counter(symbols)
    #
    ##gas_counts = {}
    #for gas in gas_dict:
    #    elem = gas[:-1]                  # works for homonuclear diatomics like H2/N2
    #    n_atoms = atom_counts.get(elem, 0)
    #    if n_atoms % 2 == 0:


    ##if gas_count % 1 == 0:
    #        logger.info(
    #        "gas-molecules count is integral (%.1f) at z_cutoff = %.2f Å. No cutoff adjustment needed.",
    #        gas_count,
    #        args["z_cutoff"],
    #     )
    #    else:
    #        logger.warning("Odd number of %s atoms in gas box: %d", elem, n_atoms)
    #        logger.info(
    #                "Non-integral gas-molecules count detected (%.1f) at z_cutoff = %.2f Å. "
    #            "Incrementally adjusting cutoff.",
    #            gas_count,
    #            args["z_cutoff"],
    #        )

    #        z_cutoff_incr = 0.1

    #        while n_atoms % 2 != 0:
    #        #while gas_count % 1 != 0:
    #            #box_gas, slab_ads, gas_count, idx_in, idx_out, xyz_dict, n_slab = extract_box_gas(
    #            #    atoms, args["z_cutoff"] - z_cutoff_incr
    #            #)
    #            box_gas, slab_ads, gas_count, gas_counts, idx_in, idx_out, xyz_dict = extract_box_gas(
    #            atoms,
    #            args["z_cutoff"],
    #            gas_templates=gas_templates
    #            )

    #            logger.info(
    #                "Adjusted z_cutoff = %.2f Å → gas count = %.1f",
    #                args["z_cutoff"] - z_cutoff_incr,
    #                gas_count,
    #            )

    #            z_cutoff_incr += 0.1

    #        logger.info(
    #            "Integral gas-molecules count achieved (%.1f) at z_cutoff = %.2f Å.",
    #            gas_count,
    #            args["z_cutoff"] - (z_cutoff_incr - 0.1),
    #        )
    #logger.info(vels_by_id.get(273))


    # 5. Compute initial energy and set parameters
    try:
        E_current = energy_fn(box_gas, "initial_box")
    except Exception as e:  # pragma: no cover - explicit runtime guard
        logger.critical(f"Energy calculation failed: {e}")
        return 1
    V = atoms.cell[0][0] * atoms.cell[1][1] * (atoms.cell[2][2] - args["z_cutoff"])
    T = args["temperature"]
    ##mu = args["chemical_potential"]
    #m_amu = args["mass"]
    #m_dict = load_gas_masses(args["gas_list"], args["gas_masses"])
    #mass1, mass2 = m_dict.values()
    #mu = chemical_potentials_binary_mixture(T, args["pressure"], args["pressure_unit"], 1, mass1, mass2)

    #mass1, mass2 = gas_dict.values()
    #mu = chemical_potential_pure(T, mass1, args["pressure"], args["pressure_unit"])

    mu_dict = compute_chemical_potentials(
        T,
        gas_dict,
        pressure=args.get("pressure"),
        pressure_unit=args.get("pressure_unit", "bar"),
        y1=float(args.get("y1", 0.75)),
        partial_pressures=args.get("partial_pressures"),
    )
    logger.info("Target chemical potentials: %s", _format_mu_dict(mu_dict))

    inactive = [g for g, mu in mu_dict.items() if not np.isfinite(mu)]
    if inactive:
        logger.info(
            "Inactive species (μ_target = -inf, no insertion expected): %s",
            inactive,
        )

    # NOTE: gas templates are intentionally NOT filtered to active species here.
    # Move selection already draws only from the unconverged (active) set via
    # mu_convergence_status, so inactive species are never inserted/deleted; the
    # frozen molecules must remain available for counting, logging and overlap.

    #gas_en = args["gas_energy"]
    gas_templates = load_centered_gas_templates(
    gas_template_dir=args["gas_template_dir"],
    gas_list=args["gas_list"],
    )

    gas_en_dict = gas_energy_fn(gas_templates)
    logger.debug(gas_en_dict)



    logger.info("Initial gas-molecule count: %s, Initial energy: %s eV", _fmt_int(gas_count), _fmt_energy(E_current))
    logger.info("Initial gas counts by species: %s", gas_counts)

    # 6. Monte Carlo loop
    logger.info("Starting GCMC loop with Δμ termination")
    output_extxyz = Path(args["bdir"]) / "mc_cycle.extxyz"

    tol = 1e-3
    
    gas_templates_all = gas_templates  # keep immutable master dict
    
    # initial diagnostics
    p_current = compute_pressure_atm(T, V, gas_count)
    mu_current = chemical_potentials_from_particles(T, V, gas_counts, gas_dict)
    inactive, converged, unconverged = mu_convergence_status(mu_dict, mu_current, tol)
    
    logger.info("Initial μ_current: %s", _format_mu_dict(mu_current))
    if bool(args.get("log_mu_diagnostics", False)):
        logger.info("μ diagnostics: %s", _format_mu_status(mu_dict, mu_current))
    logger.info("Converged: %s | Unconverged: %s | Inactive: %s",
                sorted(converged), sorted(unconverged), sorted(inactive))
    
    # flags
    run_until_converged = bool(args.get("run_until_converged", False))
    max_steps = int(args.get("max_steps", args["steps"]))
    n_steps = int(args["steps"])
    
    biased_moves = bool(args.get("biased_moves", False))
    
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
                    mu_target=mu_dict,
                    mu_current=mu_current,
                    unconverged=unconverged,
                    gas_counts=gas_counts,
                    force_dmu_threshold=float(args.get("force_dmu_threshold", 0.0)),
                    force_single_species=bool(args.get("force_single_species", True)),
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
            if ins:
                new_mol, new_box, ins_name = insertion_mc(gas_templates_active, region_bounds, box_gas)
                move_name = ins_name
            else:
                if gas_count == 0:
                    logger.info("Step %d proposal skipped: no gas molecules available for deletion", step)
                    mu_current = chemical_potentials_from_particles(T, V, gas_counts, gas_dict)
                    inactive, converged, unconverged = mu_convergence_status(mu_dict, mu_current, tol)
                    if bool(args.get("log_mu_diagnostics", False)):
                        logger.info("μ diagnostics: %s", _format_mu_status(mu_dict, mu_current))
                    logger.info("Converged: %s | Unconverged: %s | Inactive: %s",
                                sorted(converged), sorted(unconverged), sorted(inactive))
                    continue
                try:
                    del_mol, new_box, move_name = deletion_mc(box_gas, gas_templates_active)
                except RuntimeError:
                    # fallback to insertion (same active templates)
                    logger.info("Step %d deletion proposal failed; falling back to insertion.", step)
                    new_mol, new_box, ins_name = insertion_mc(gas_templates_active, region_bounds, box_gas)
                    move_name = ins_name
                    ins = True
    
            # Energy of proposal
            E_new = energy_fn(new_box, f"step_{step}_{move_name}_{'ins' if ins else 'del'}")
    
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
    
                # update counts after accept (homonuclear diatomics assumption)
                symbols = box_gas.get_chemical_symbols()
                atom_counts = Counter(symbols)
    
                gas_counts = {}
                for gas in gas_dict:
                    elem = gas[:-1]
                    n_atoms = atom_counts.get(elem, 0)
                    if n_atoms % 2 != 0:
                        logger.warning("Odd number of %s atoms in gas box: %d", elem, n_atoms)
                    gas_counts[gas] = n_atoms // 2
    
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

            # Update μ diagnostics every step (accepted or not)
            mu_current = chemical_potentials_from_particles(T, V, gas_counts, gas_dict)
            inactive, converged, unconverged = mu_convergence_status(mu_dict, mu_current, tol)
            if bool(args.get("log_mu_diagnostics", False)):
                logger.info("μ diagnostics: %s", _format_mu_status(mu_dict, mu_current))

            if step == 1 or step % int(args.get("log_every", 1)) == 0:
                logger.info("Converged: %s | Unconverged: %s | Inactive: %s",
                            sorted(converged), sorted(unconverged), sorted(inactive))

            write_extxyz_sequence(output_extxyz, box_gas)

        except Exception as e:
            logger.error("MC step %d failed: %s", step, e)
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

    # 7. Merge and write new initial data for next step
    new_struct = slab_ads + box_gas

    old_atoms_by_id, old_vels_by_id = parse_lammps_data(args["restart_path"])
    
    new_ids, matched_old_id = assign_ids_preserve_slab(
        old_atoms_by_id=old_atoms_by_id,
        new_struct=new_struct,
        elements=args["elements"],          # ["Fe","N","H"]
        slab_element=args["slab"],          # "Fe"
        tol=1e-4,
        reuse_old_ids_for_gas=False,
    )
    
    new_pos = np.asarray(new_struct.get_positions(), dtype=float)

    new_vel = build_new_velocities_from_matched_ids(
        matched_old_id=matched_old_id,
        old_vels_by_id=old_vels_by_id,
        new_struct=new_struct,
        T=T,
    )
   
    bdir = Path(args["bdir"])

    write_lammps_data_atomic_with_ids(
        out_path=bdir / "initial.lammpsdata",
        cell=new_struct.cell,
        elements=args["elements"],                 # ["Fe","N","H"]
        masses_by_type=element_masses(args["elements"]),
        new_ids=new_ids,
        new_struct=new_struct,
        new_vel=new_vel,
    )
    
    #new_N = new_struct.get_global_number_of_atoms()
    #new_in_list_pos = []
    #new_in_list_vel = []
    #cell_new_in_list_pos = []
    #cell_new_in_list_vel = []

    ## identify unchanged molecules in the pressure-controlled region
    ##logger.info(idx_in)
    #common_pos, new_pos, new_syms = get_matching_gas_mols_with_symbols(
    #box_gas,
    #old_gas_pos,
    #tol=1e-5
    #)

    ##common_gasmols, not_common_gasmols = get_matching_gas_mols(box_gas.get_positions(), idx_in)
    ##logger.info(not_common_gasmols)
    ##org_id_list = filter_dict(xyz_dict, common_gasmols)
    #org_id_list = filter_dict(xyz_dict, common_pos)

    #for id_num in org_id_list:
    #    rec = restart_dict[id_num]
    #    new_in_list_pos.append([*rec[:3], *map(str, rec[3:6])])
    #    new_in_list_vel.append(rec[6:9])

    ## assign positions to the newly added molecules
    #for xyz in new_pos:
    #    new_in_list_pos.extend([xyz.tolist() + ["0"] * 3])

    ## assign initial velocities to the newly added molecules
    #for sym in new_syms:
    #    m_amu = atomic_masses[atomic_numbers[sym]]
    #    v = boltzmann_velocity_distribution(T, m_amu)
    #    new_in_list_vel.append([vi / 100 for vi in v])
    #

    ##for xyz in not_common_gasmols:
    ##    new_in_list_pos.extend([xyz.tolist() + ["0"] * 3])
    ##    new_v = [i / 100 for i in boltzmann_velocity_distribution(T, mass1 / 2)]
    ##    new_in_list_vel.append(new_v)

    #for id_num in idx_out:
    #    rec = restart_dict[id_num]
    #    cell_new_in_list_pos.append([*rec[:3], *map(str, rec[3:6])])
    #    cell_new_in_list_vel.append(rec[6:9])

    ## generate final list with positions and velocities
    #cell_new_in_list_pos.extend(new_in_list_pos)
    #cell_new_in_list_vel.extend(new_in_list_vel)

    #flatten_list = reduce(operator.concat, cell_new_in_list_pos)
    #new_coord = np.asarray(flatten_list, dtype=object).reshape(new_N, 6)
    ##type_list = [1] * n_slab + [2] * (new_N - n_slab)
    ##type_arr = np.asarray(type_list, dtype=int).reshape(new_N, 1)
    #num_arr = np.asarray(range(1, new_struct.get_global_number_of_atoms()+1), dtype=int).reshape(new_struct.get_global_number_of_atoms(), 1)

    ###xyz_c = new_struct.get_chemical_symbols()
    ###pos_block = np.concatenate((xyz_c, new_coord.astype(str)), axis=1)
    ####xyz_c = np.concatenate((num_arr.astype(str), type_arr.astype(str)), axis=1)

    ### (N, 1) array of symbols
    #sym_new_struct = np.array(new_struct.get_chemical_symbols(), dtype=str).reshape(-1, 1)
    ##
    ### Convert symbols to integer IDs
    #xyz_id = np.vectorize(elem_to_id.get)(sym_new_struct)
    #xyz_c = np.concatenate((num_arr.astype(str), xyz_id.astype(str)), axis=1)
    ###xyz_c = np.array(new_struct.get_chemical_symbols(), dtype=str).reshape(-1, 1)  # (N,1)
    ##new_coord = np.asarray(new_coord)                                             # (N,3)
    ##
    #pos_block = np.concatenate((xyz_c, new_pos), axis=1)            # (N,4)
    ##
    ##vel_arr = np.asarray(cell_new_in_list_vel).reshape(new_N, 3)
    #vel_block = np.concatenate((num_arr.astype(str), new_vel), axis=1)

    ##pos_string = write_string(pos)
    ##vel_string = write_string(vel)
    #pos_string = write_string(pos_block)
    #vel_string = write_string(vel_block)

    #lammpsdata_string = write_lammpsdata(
    #    num_atoms=new_struct.get_global_number_of_atoms(),
    #    positions=pos_string,
    #    velocities=vel_string,
    #    x_cell=new_struct.cell[0, 0],
    #    y_cell=new_struct.cell[1, 1],
    #    z_cell=new_struct.cell[2, 2],
    #    elements=elements,
    #)

    #bdir = Path(args["bdir"])
    #data_file = bdir / "initial.lammpsdata"
    #data_file.write_text(lammpsdata_string)
    p_fin = compute_pressure_atm(T, V, gas_count)
    logger.info(
        "Finished GCMC. Final gas molecules count: %s, Final energy: %s eV", _fmt_int(gas_count), _fmt_energy(E_current)
    )
    logger.info("Final gas counts by species: %s", gas_counts)
    if bool(args.get("log_mu_diagnostics", False)):
        logger.info("Final μ diagnostics: %s", _format_mu_status(mu_dict, mu_current))

    logger.info(
        "Finished GCMC. Final pressure: %s atm.", _fmt_pressure(p_fin)
    )
    logger.info("Wrote %s", bdir / "initial.lammpsdata")
    logger.info("Wrote %s", output_extxyz)

    return 0
