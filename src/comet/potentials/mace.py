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

# Cache of constructed MACE calculators, keyed by (model_path, device). Building
# a MACECalculator loads the model from disk, so we do it once and reuse the
# instance across every energy evaluation (a single calculator works for any
# Atoms object).
_CALCULATOR_CACHE: Dict[tuple, MACECalculator] = {}


def _get_mace_calculator(model_path: Path, device: str) -> MACECalculator:
    """Return a cached MACE calculator for ``model_path``/``device``.

    The model is loaded from disk on first request and reused thereafter.
    """
    key = (str(model_path), device)
    calc = _CALCULATOR_CACHE.get(key)
    if calc is None:
        logger.info("Loading MACE model %s on device '%s'", model_path, device)
        calc = MACECalculator(str(model_path), device=device)
        _CALCULATOR_CACHE[key] = calc
    return calc


def get_energy_mace(atoms: Atoms, model_dir: Path) -> float:
    """
    Compute the potential energy of an Atoms object using a MACE model.

    The underlying MACE calculator is cached and reused across calls (the model
    is read from disk only once per process), so this is cheap to call in a loop.

    Args:
        atoms (Atoms): ASE Atoms object for energy evaluation.
        model_dir (Path): Path to the MACE model file.

    Returns:
        float: Potential energy in eV.
    """
    model_path = Path(model_dir)
    if not model_path.exists():
        logger.error(f"MACE model not found: {model_path}")
        raise FileNotFoundError(model_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    atoms.calc = _get_mace_calculator(model_path, device)
    energy = atoms.get_potential_energy()
    logger.debug("Computed MACE energy: %.6f eV", energy)
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
