"""GCMC workflow orchestration.

`run()` wires the pipeline together — load+validate config, build the energy
backend, build the initial system, run the MC loop, write the restart — while
the actual work lives in :mod:`comet.workflows.stages` and the energy backend in
:mod:`comet.potentials.backends`.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np

from comet.config.schema import RunConfig, load_run_config
from comet.potentials.backends import build_energy_backend
from comet.workflows.logging_utils import (
    _fmt_counts,
    log_banner,
    log_section,
    log_settings,
    logger,
    setup_logging,
)
from comet.workflows.stages import (
    build_initial_system,
    log_convergence_verdict,
    log_pressure_summary,
    run_mc_loop,
    write_restart,
)

__all__ = ["cycle", "run", "setup_logging"]


def _log_run_header(config: RunConfig) -> None:
    """Write the framed settings block for the current run to the log."""
    unit = config.pressure_unit
    settings = {
        "Energy backend": config.energy_backend,
        "Output directory": config.bdir,
        "Restart file": config.restart_path,
        "Gas species": ", ".join(config.gas_list),
        "Temperature": f"{config.temperature:.2f} K",
    }
    if config.ratios is not None:
        settings["Total pressure"] = f"{config.pressure} {unit}"
        settings["Composition"] = (
            ":".join(config.ratios) + " = " + ":".join(str(r) for r in config.ratios.values())
        )
        settings["Partial pressures"] = ", ".join(
            f"{g}: {p:.4g} {unit}" for g, p in config.partial_pressures.items()
        )
    elif config.partial_pressures is not None:
        settings["Partial pressures"] = ", ".join(
            f"{g}: {p:.4g} {unit}" for g, p in config.partial_pressures.items()
        )
    else:
        settings["Total pressure"] = f"{config.pressure} {unit} (legacy mode)"
    settings["MC steps"] = config.steps
    settings["Run until converged"] = f"{config.run_until_converged} (max_steps: {config.max_steps})"
    settings["Biased moves"] = config.biased_moves
    settings["Seed"] = config.seed if config.seed is not None else "not set (irreproducible)"
    if config.energy_backend == "orca":
        settings["ORCA"] = (
            f"{config.orca_method}/{config.orca_basis}, nprocs={config.orca_nprocs}, "
            f"charge={config.orca_charge}, mult={config.orca_mult}"
        )
    if config.md is not None:
        settings["MD cycles"] = config.md.n_cycles
        settings["MD steps/cycle"] = f"{config.md.md_steps} x {config.md.timestep_fs} fs"
        settings["MD thermostat"] = f"Bussi (CSVR), tau = {config.md.tau_t_ps} ps"
        settings["MD frozen bottom"] = f"z < z_min + {config.md.freeze_bottom} Å"
    log_settings(settings)


def run(config_path: str) -> int:
    """Execute the GCMC workflow defined by the configuration file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        int: Process-style exit status (0 on success, 1 on a setup failure).
    """
    setup_logging()  # attach handlers / create gcmc_run.log now, not at import
    config = load_run_config(config_path)
    log_banner("GCMC pressure control")
    if config.seed is not None:
        # Seed both RNGs the workflow uses: `random` (MC moves) and numpy
        # (Maxwell-Boltzmann velocities) — makes a run fully reproducible.
        random.seed(config.seed)
        np.random.seed(config.seed)
    Path(config.bdir).mkdir(parents=True, exist_ok=True)
    _log_run_header(config)

    backend = build_energy_backend(config)

    log_section("SYSTEM SETUP")
    state = build_initial_system(config, backend)
    if state is None:  # trajectory or initial-energy failure (already logged)
        return 1

    log_section("GCMC SAMPLING")
    state = run_mc_loop(state, config, backend)

    log_section("PRESSURE CONTROL SUMMARY")
    write_restart(state, config)
    return 0


def cycle(config_path: str) -> int:
    """Alternate GCMC and ASE-MD in one process (no external MD engine).

    Requires an ``md:`` block in the configuration and a backend that can
    provide MD forces (currently ``mace``). Each cycle runs the GCMC loop,
    then an MD segment on slab+gas with the same calculator, re-partitions by
    molecule COM, and writes a ``cycle_<i>.lammpsdata`` checkpoint.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        int: Process-style exit status (0 on success, 1 on a setup failure).
    """
    from comet.workflows.md import run_md, write_cycle_checkpoint

    setup_logging()
    config = load_run_config(config_path)
    log_banner("GCMC <-> MD cycling (ASE MD)")
    if config.md is None:
        logger.critical("comet cycle requires an `md:` block in the configuration")
        return 1
    if config.seed is not None:
        random.seed(config.seed)
        np.random.seed(config.seed)
    Path(config.bdir).mkdir(parents=True, exist_ok=True)
    _log_run_header(config)

    backend = build_energy_backend(config)
    if backend.calculator_factory is None:
        logger.critical(
            "comet cycle requires a backend with MD forces; "
            "backend %r provides none (use energy_backend: mace)",
            backend.name,
        )
        return 1
    calculator = backend.calculator_factory()

    log_section("SYSTEM SETUP")
    state = build_initial_system(config, backend)
    if state is None:
        return 1

    n_cycles = config.md.n_cycles
    for i in range(1, n_cycles + 1):
        log_section(f"CYCLE {i}/{n_cycles} — GCMC")
        state = run_mc_loop(state, config, backend)
        log_section(f"CYCLE {i}/{n_cycles} — MD")
        state = run_md(state, config, calculator, backend, cycle_index=i)
        write_cycle_checkpoint(state, config, Path(config.bdir) / f"cycle_{i}.lammpsdata")

    log_section("PRESSURE CONTROL SUMMARY")
    logger.info("Finished %d GCMC <-> MD cycles. Final gas counts: %s",
                n_cycles, _fmt_counts(state.gas_counts))
    log_convergence_verdict(state)
    log_pressure_summary(state, config)
    bdir = Path(config.bdir)
    logger.info("Output files:")
    logger.info("  %s", bdir / "cycle_<i>.lammpsdata")
    logger.info("      Per-cycle checkpoints (i = 1..%d): slab+gas after each MD segment,", n_cycles)
    logger.info("      with the current velocities in LAMMPS metal units.")
    logger.info("  %s", bdir / "mc_cycle.extxyz")
    logger.info("      Gas-box trajectory across all GCMC segments (extended XYZ).")
    if config.md.traj_every > 0:
        logger.info("  %s", bdir / "md_cycle.extxyz")
        logger.info("      Slab+gas MD trajectory, one frame every %d MD steps.", config.md.traj_every)
    return 0
