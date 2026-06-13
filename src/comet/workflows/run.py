"""GCMC workflow orchestration.

`run()` wires the pipeline together — load+validate config, build the energy
backend, build the initial system, run the MC loop, write the restart — while
the actual work lives in :mod:`comet.workflows.stages` and the energy backend in
:mod:`comet.potentials.backends`.
"""

from __future__ import annotations

from pathlib import Path

from comet.config.schema import RunConfig, load_run_config
from comet.potentials.backends import build_energy_backend
from comet.workflows.logging_utils import logger, setup_logging
from comet.workflows.stages import build_initial_system, run_mc_loop, write_restart

__all__ = ["run", "setup_logging"]


def _log_run_header(config: RunConfig) -> None:
    """Write a compact summary of the current run settings to the log."""
    unit = config.pressure_unit
    if config.partial_pressures is not None:
        pressure_desc = f"partial_pressures={config.partial_pressures} {unit}"
    else:
        pressure_desc = f"{config.pressure} {unit}"
    logger.info(
        "Run header: backend=%s, bdir=%s, temperature=%.3f K, pressure=%s, "
        "run_until_converged=%s, steps=%s, max_steps=%s, biased_moves=%s",
        config.energy_backend,
        config.bdir,
        config.temperature,
        pressure_desc,
        config.run_until_converged,
        config.steps,
        config.max_steps,
        config.biased_moves,
    )
    logger.info("Log settings: log_mu_diagnostics=%s", config.log_mu_diagnostics)
    if config.energy_backend == "orca":
        logger.info(
            "ORCA settings: method=%s, basis=%s, nprocs=%s, charge=%s, mult=%s",
            config.orca_method,
            config.orca_basis,
            config.orca_nprocs,
            config.orca_charge,
            config.orca_mult,
        )


def run(config_path: str) -> int:
    """Execute the GCMC workflow defined by the configuration file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        int: Process-style exit status (0 on success, 1 on a setup failure).
    """
    setup_logging()  # attach handlers / create gcmc_run.log now, not at import
    config = load_run_config(config_path)
    Path(config.bdir).mkdir(parents=True, exist_ok=True)
    _log_run_header(config)

    backend = build_energy_backend(config)

    state = build_initial_system(config, backend)
    if state is None:  # trajectory or initial-energy failure (already logged)
        return 1

    state = run_mc_loop(state, config, backend)
    write_restart(state, config)
    return 0
