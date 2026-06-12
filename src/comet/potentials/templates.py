"""Shared helpers for gas templates used by energy backends."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
from ase import Atoms
from ase.io import read


def load_centered_gas_templates(
    gas_template_dir: Path,
    gas_list: List[str],
    box_length_A: float = 20.0,
    pbc=(False, False, False),
) -> Dict[str, Atoms]:
    """
    Load diatomic gas templates and center each molecule in a cubic box.
    """
    gas_template_dir = Path(gas_template_dir)
    if not gas_template_dir.exists():
        raise FileNotFoundError(gas_template_dir)

    cell = np.eye(3) * float(box_length_A)
    center = np.array([box_length_A / 2] * 3)
    templates: Dict[str, Atoms] = {}

    for gas in gas_list:
        xyz_path = gas_template_dir / f"{gas}.xyz"
        if not xyz_path.exists():
            raise FileNotFoundError(xyz_path)

        mol = read(str(xyz_path))
        mol.positions = mol.positions - mol.get_center_of_mass() + center
        mol.set_cell(cell)
        mol.set_pbc(pbc)
        templates[gas] = mol

    return templates
