"""ORCA energy evaluation helpers."""

from __future__ import annotations

import logging
import shutil
import tempfile
from pathlib import Path
from typing import Dict

from ase import Atoms
from ase.calculators.orca import ORCA, OrcaProfile

logger = logging.getLogger(__name__)


def _orca_simpleinput(config: dict) -> str:
    """Return the ORCA simple-input string for a calculation.

    Args:
        config: Workflow configuration dictionary.

    Returns:
        str: ORCA simple-input line.
    """
    simpleinput = config.get("orca_simpleinput")
    if simpleinput:
        return str(simpleinput)

    method = str(config.get("orca_method", "PBE"))
    basis = str(config.get("orca_basis", "def2-SVP"))
    return f"{method} {basis}"


def _orca_blocks(config: dict) -> str:
    """Return the ORCA block section, including `%pal` settings.

    Args:
        config: Workflow configuration dictionary.

    Returns:
        str: Block text passed to the ASE ORCA calculator.
    """
    nprocs = int(config.get("orca_nprocs", 1))
    extra = str(config.get("orca_blocks", "")).strip()
    base = f"%pal nprocs {nprocs} end"
    return f"{base}\n{extra}" if extra else base


def _orca_profile() -> OrcaProfile:
    """Build an ASE ORCA profile from the `orca` executable on `PATH`.

    Returns:
        OrcaProfile: ASE ORCA execution profile.

    Raises:
        FileNotFoundError: If the `orca` executable is not available.
    """
    orca_path = shutil.which("orca")
    if not orca_path:
        raise FileNotFoundError("ORCA executable not found in PATH")
    return OrcaProfile(command=orca_path)


def get_energy_orca(atoms: Atoms, config: dict, label: str = "sp") -> float:
    """
    Compute a single-point energy with ORCA for a non-periodic copy of `atoms`.
    """
    charge = int(config.get("orca_charge", 0))
    mult = int(config.get("orca_mult", 1))
    keep_workdirs = bool(config.get("orca_keep_workdirs", False))
    work_root = Path(config.get("orca_work_root", config.get("bdir", Path.cwd())))
    work_root.mkdir(parents=True, exist_ok=True)

    run_dir = Path(tempfile.mkdtemp(prefix=f"orca_{label}_", dir=work_root))
    logger.info("Running ORCA single-point in %s", run_dir)

    mol = atoms.copy()
    mol.set_pbc(False)
    mol.set_cell([0.0, 0.0, 0.0])

    try:
        mol.calc = ORCA(
            profile=_orca_profile(),
            directory=str(run_dir),
            charge=charge,
            mult=mult,
            orcasimpleinput=_orca_simpleinput(config),
            orcablocks=_orca_blocks(config),
        )
        energy = float(mol.get_potential_energy())
        logger.info("Computed ORCA energy: %.6f eV", energy)
        return energy
    finally:
        if not keep_workdirs:
            shutil.rmtree(run_dir, ignore_errors=True)


def compute_gas_energies_orca(gas_templates: Dict[str, Atoms], config: dict) -> Dict[str, float]:
    """
    Compute ORCA single-point energies for each gas template.
    """
    gas_en: Dict[str, float] = {}
    for gas, atoms in gas_templates.items():
        gas_en[gas] = get_energy_orca(atoms, config, label=f"gas_{gas}")
        logger.info("Gas energy %s: %.6f eV", gas, gas_en[gas])
    return gas_en
