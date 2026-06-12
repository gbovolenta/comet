"""MACE energy evaluation helpers."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict
from ase import Atoms

import torch
from mace.calculators import MACECalculator

from comet.potentials.templates import load_centered_gas_templates

logger = logging.getLogger(__name__)



def get_energy_mace(atoms: Atoms, model_dir: Path) -> float:
    """
    Compute the potential energy of an Atoms object using a MACE model.

    Args:
        atoms (Atoms): ASE Atoms object for energy evaluation.
        model_dir (Path): Directory containing the MACE model file.

    Returns:
        float: Potential energy in eV.
    """
    model_dir = Path(model_dir)
    #model_path = model_dir / "model2" / "energy_forces_stagetwo.model"
    model_path = model_dir 
    if not model_path.exists():
        logger.error(f"MACE model not found: {model_path}")
        raise FileNotFoundError(model_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Using device '%s' for MACE calculations", device)
    atoms.calc = MACECalculator(str(model_path), device=device)
    energy = atoms.get_potential_energy()
    logger.info("Computed MACE energy: %.6f eV", energy)
    return energy

def compute_gas_energies(
    gas_templates: Dict[str, Atoms],
    model_dir: Path
) -> Dict[str, float]:
    """
    Compute energies for each gas template.

    Args:
        gas_templates: Dict {gas_name: Atoms}

    Returns:
        Dict {gas_name: energy}
    """
    gas_en: Dict[str, float] = {}

    for gas, atoms in gas_templates.items():
        en = get_energy_mace(atoms,model_dir)
        gas_en[gas] = en
        logger.info("Gas energy %s: %.6f eV", gas, en)

    return gas_en
