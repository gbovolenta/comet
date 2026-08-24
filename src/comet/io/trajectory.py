import logging
from pathlib import Path

from ase import Atoms
from ase.io import lammpsdata, write

logger = logging.getLogger(__name__)


def get_last_frame(data_file: Path) -> Atoms:
    """
    Load and return an ASE Atoms object from a LAMMPS data file
    containing Atoms and (optionally) Velocities blocks.

    Args:
        data_file (Path): Path to the LAMMPS data file.

    Returns:
        Atoms: Configuration as an ASE Atoms object (with velocities if present).
    """
    data_file = Path(data_file)
    if not data_file.exists():
        logger.error(f"LAMMPS data file not found: {data_file}")
        raise FileNotFoundError(data_file)

    atoms = lammpsdata.read_lammps_data(data_file, sort_by_id=False, read_image_flags=False)
    logger.info("Loaded LAMMPS data with %d atoms", atoms.get_global_number_of_atoms())
    return atoms


def write_extxyz_sequence(path: Path, atoms: Atoms) -> None:
    """
    Append an ASE Atoms object as a frame in an extxyz trajectory file.

    Args:
        path (Path): Path to the extxyz file.
        atoms (Atoms): ASE Atoms object to append.
    """
    try:
        # Write a calculator-free snapshot: copy() drops `.calc`, so the writer
        # can never pull results that a shared calculator computed for a
        # different (e.g. rejected-trial) structure.
        write(str(path), atoms.copy(), format="extxyz", append=True)
        logger.debug(f"Appended frame to {path}")
    except Exception as e:
        logger.error(f"Failed writing extxyz: {e}")
        raise
