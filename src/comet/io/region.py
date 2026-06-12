import logging
from pathlib import Path
from typing import Tuple

from ase.io import lammpsdata

logger = logging.getLogger(__name__)


def bounds_from_restart(
    restart_path: Path,
    z_cutoff: float,
) -> Tuple[float, float, float, float, float, float]:
    """
    Generate GCMC insertion bounds from a LAMMPS restart/data file and z cutoff.

    Bounds are:
        (xlo, xhi, ylo, yhi, z_cutoff, zhi)

    Args:
        restart_path: Path to LAMMPS data file.
        z_cutoff: Lower z bound for gas insertion (Å).

    Returns:
        (xlo, xhi, ylo, yhi, zlo, zhi)
    """
    restart_path = Path(restart_path)
    if not restart_path.exists():
        raise FileNotFoundError(restart_path)

    atoms = lammpsdata.read_lammps_data(
        restart_path,
        sort_by_id=False,
        read_image_flags=False,
    )

    # ASE cell: vectors a, b, c
    cell = atoms.cell

    xlo, xhi = 0.0, cell[0, 0]
    ylo, yhi = 0.0, cell[1, 1]
    zlo, zhi = z_cutoff, cell[2, 2]

    bounds = (xlo, xhi, ylo, yhi, zlo, zhi)
    logger.info("Generated GCMC bounds from restart: %s", bounds)
    return bounds



def read_region(file_path: Path) -> Tuple[float, float, float, float, float, float]:
    """
    Read GCMC insertion region bounds from a file.

    Args:
        file_path (Path): Path to a text file containing six floats:
                          x_low x_high y_low y_high z_low z_high

    Returns:
        Tuple[float, float, float, float, float, float]: The six region bounds.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        logger.error(f"Region file not found: {file_path}")
        raise FileNotFoundError(file_path)
    bounds = tuple(map(float, file_path.read_text().split()))
    logger.info("Loaded region bounds: %s", bounds)
    return bounds
