import re
import argparse
import logging
import random
from pathlib import Path
from typing import Tuple, List, Dict, Union
import yaml
import numpy as np
from scipy.spatial import cKDTree
from functools import reduce
import operator
from ase import Atoms
from ase.io import read, write
from ase.io import lammpsdata
from ase.io.lammpsdata import write_lammps_data
import torch
from mace.calculators import MACECalculator
#from ase.units import kB, hbar, amu, Angstrom, eV
from ase.units import Bohr,Rydberg,kJ,kB,fs,Hartree,mol,kcal,eV,Angstrom
from ase.data import atomic_masses, atomic_numbers
from scipy.constants import hbar as hbar_SI

hbar = hbar_SI / eV  # convert J·s → eV·s
amu_to_kg = 1.66053906660e-27  # Conversion factor for amu to kg
h_eV_s = 4.135667696e-15    # Planck constant in eV·s
kB_eV = 8.617333262145e-5  # Boltzmann constant in eV/K

def setup_logging() -> logging.Logger:
    """
    Configure and return a logger that writes DEBUG-level messages to the file 'gcmc_run.log'
    and INFO-level messages to the console. All log entries exclude timestamps.

    Returns:
        logging.Logger: The configured logger instance.
    """
    log = logging.getLogger("gcmc")
    log.setLevel(logging.DEBUG)

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


# Initialize logger
logger = setup_logging()


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments and optionally override them with a YAML config file.

    Returns:
        argparse.Namespace: Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="GCMC-LAMMPS coupling script with MACE energy calculator"
    )
    parser.add_argument("--config", type=Path, help="Path to YAML config file")
    parser.add_argument("--bdir", type=Path, default=None, help="Output directory path")
    parser.add_argument("--model-dir", type=Path, help="Path to MACE model directory")
    parser.add_argument("--h2-path", type=Path, help="Path to H2 molecule xyz file")
    parser.add_argument("--traj-path", type=Path, help="Path to LAMMPS trajectory file")
    parser.add_argument("--restart-path", type=Path, help="Path to LAMMPS restart file")
    parser.add_argument("--box-path", type=Path, help="Path to GCMC region file")
    parser.add_argument("--z-cutoff", type=float, help="Z cutoff for hydrogen box")
    parser.add_argument(
        "--temperature", type=float, help="Simulation temperature in Kelvin"
    )
    parser.add_argument(
        "--chemical-potential", type=float, help="Target chemical potential in eV"
    )
    parser.add_argument(
        "--mass", type=float, help="Mass of H2 molecule in atomic mass units"
    )
    parser.add_argument("--h2-energy", type=float, help="Reference H2 energy in eV")
    parser.add_argument("--steps", type=int, help="Number of Monte Carlo steps")
    parser.add_argument(
    "--elements",
    nargs="+",
    default=None,
    help="List of element symbols in LAMMPS atom-type order (e.g. Fe H). "
         "If omitted, read from input.yaml.")
    args = parser.parse_args()

    # Override with YAML config if provided
    if args.config:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        for key, val in cfg.items():
            if hasattr(args, key) and val is not None:
                setattr(args, key, val)

    # Default output directory
    if args.bdir is None:
        args.bdir = Path.cwd()

    logger.info("Parsed arguments: %s", args)
    return args


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

def write_extxyz_sequence(path: Path, atoms: Atoms) -> None:
    """
    Append an ASE Atoms object as a frame in an extxyz trajectory file.

    Args:
        path (Path): Path to the extxyz file.
        atoms (Atoms): ASE Atoms object to append.
    """
    try:
        write(str(path), atoms, format="extxyz", append=True)
        logger.debug(f"Appended frame to {path}")
    except Exception as e:
        logger.error(f"Failed writing extxyz: {e}")
        raise


def write_string(xyz) -> str:
    """
    Convert a 2D array-like object into a formatted multiline string.
    """
    return "\n".join(
        "     ".join(str(value) for value in row)
        for row in xyz
    )
def format_masses(elements: list[str]) -> str:
    """
    Generate the LAMMPS 'Masses' section from a list of element symbols.
    """
    lines = []
    for i, element in enumerate(elements, start=1):
        Z = atomic_numbers[element]
        mass = atomic_masses[Z]
        lines.append(f"{i} {mass:.6f}")
    return "\n".join(lines)

def write_lammpsdata(
    num_atoms,
    positions,
    velocities,
    x_cell,
    y_cell,
    z_cell,
    elements: list[str],
) -> str:
    masses_string = format_masses(elements)

    return f"""LAMMPS data file

{num_atoms} atoms
{len(elements)} atom types

0 {x_cell} xlo xhi
0 {y_cell} ylo yhi
0 {z_cell} zlo zhi

Masses

{masses_string}

Atoms # atomic

{positions}

Velocities

{velocities}
"""



def parse_lammps_data(file_path):
    atom_block = False
    velocity_block = False
    atom_data = {}

    with open(file_path, 'r') as file:
        for line in file:
            if "Atoms" in line:  # Start of Atoms block
                atom_block = True
                continue
            if "Velocities" in line:  # Start of Velocities block
                atom_block = False
                velocity_block = True
                continue
            if atom_block and line.strip():
                # Parse Atoms block: N type x y z extra1 extra2 extra3
                tokens = list(map(float, re.split(r'\s+', line.strip())))
                N = int(tokens[0]) - 1  # Adjust N to zero-indexed
                #atom_data[N] = tokens[2:8]  # Store x, y, z, extra1, extra2, extra3
                atom_data[N] =  list(map(float, tokens[2:5])) + list(map(int, tokens[5:8]))
            if velocity_block and line.strip():
                # Parse Velocities block: N vx vy vz
                tokens = list(map(float, re.split(r'\s+', line.strip())))
                N = int(tokens[0]) - 1  # Adjust N to zero-indexed
                if N in atom_data:
                    atom_data[N].extend(tokens[1:])  # Append vx, vy, vz

    return atom_data

def get_matching_h(xyz_arr1,xyz_arr2):
    common_h = []
    not_common_h = []
    for xyz1 in xyz_arr1:
        matched = False
        for xyz2 in xyz_arr2:
            if np.array_equal(xyz1,xyz2):
                common_h.append(xyz1)
                matched = True
                break
        if not matched:
            not_common_h.append(xyz1)
    return common_h, not_common_h

def filter_dict(dict_org, common_h):
    # Initialize an empty dictionary to store the filtered key-value pairs
    filtered_dict = {}

    # Loop through the dictionary items
    for key, value in dict_org.items():
        # Check if the value is in the common_h list using numpy's array_equal
        for common_array in common_h:
            if np.array_equal(value, common_array):
                # If a match is found, add the key-value pair to the filtered_dict
                filtered_dict[key] = value
                break  # Exit the inner loop once a match is found
    return filtered_dict.keys()

def boltzmann_velocity_distribution(T, mass):
    """
    Assigns velocities (v_x, v_y, v_z) based on the Boltzmann distribution.
    
    Parameters:
    T (float): Temperature in Kelvin.
    mass (float): Mass of the particle in atomic mass units (amu).
    
    Returns:
    tuple: Velocities (v_x, v_y, v_z) in m/s.
    """
    k_B = 1.380649e-23  # Boltzmann constant (J/K)
    amu_to_kg = 1.66053906660e-27  # Conversion factor from amu to kg
    
    # Convert mass to kg
    mass_kg = mass * amu_to_kg
    
    # Compute the variance of the velocity distribution
    variance = k_B * T / mass_kg
    
    # The velocity components follow a Gaussian distribution with mean 0 and variance proportional to T/mass
    v_x = np.random.normal(0, np.sqrt(variance))
    v_y = np.random.normal(0, np.sqrt(variance))
    v_z = np.random.normal(0, np.sqrt(variance))
    
    return [v_x, v_y, v_z]

def get_last_frame(traj_file: Path) -> Atoms:
    """
    Load and return the last frame from a trajectory file.

    Args:
        traj_file (Path): Path to the trajectory file.

    Returns:
        Atoms: The last frame as an ASE Atoms object.
    """
    traj_file = Path(traj_file)
    if not traj_file.exists():
        logger.error(f"Trajectory file not found: {traj_file}")
        raise FileNotFoundError(traj_file)
    frame = read(str(traj_file), index=-1)
    logger.info(
        "Loaded last trajectory frame with %d atoms", frame.get_global_number_of_atoms()
    )
    return frame


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
    model_path = model_dir / "model2" / "energy_forces_stagetwo.model"
    if not model_path.exists():
        logger.error(f"MACE model not found: {model_path}")
        raise FileNotFoundError(model_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Using device '%s' for MACE calculations", device)
    atoms.calc = MACECalculator(str(model_path), device=device)
    energy = atoms.get_potential_energy()
    logger.info("Computed MACE energy: %.6f eV", energy)
    return energy


def lambda_wl(T: float, m_amu: float) -> float:
    """
    Calculate the thermal de Broglie wavelength for a particle.

    Args:
        T (float): Temperature in Kelvin.
        m_amu (float): Particle mass in atomic mass units (amu).

    Returns:
        float: Thermal wavelength in Ångström.
    """
    m_kg = m_amu * amu_to_kg

    lam = h_eV_s / np.sqrt(2 * np.pi * m_kg * kB_eV * T)
    logger.debug("Thermal wavelength: %.6f Å", lam)
    return lam


def compute_chemical_potential(T: float, m_amu: float, V: float, n: int) -> float:
    """
    Compute the chemical potential for an ideal gas.

    Args:
        T (float): Temperature in Kelvin.
        m_amu (float): Particle mass in amu.
        V (float): Volume in cubic Ångströms.
        n (int): Number of particles.

    Returns:
        float: Chemical potential in eV.
    """
    lam = lambda_wl(T, m_amu)
    density = n / V
    mu = kB_eV * T * np.log(density * lam**3)
    logger.debug("Computed chemical potential: %.6f eV", mu)
    return mu


def extract_box_hydrogens(
    atoms: Atoms, z_cutoff: float
) -> Tuple[Atoms, Atoms, int, List[int], List[int]]:
    """
    Split the system into a hydrogen box and a surface based on z cutoff.

    Args:
        atoms (Atoms): Full system as ASE Atoms.
        z_cutoff (float): Z-coordinate cutoff in Å.

    Returns:
        Tuple containing:
            box_h: Hydrogen-only box above cutoff.
            feh: Surface (Fe/H) below cutoff.
            h2_count: Number of H2 molecules found.
            idx_in: Indices of atoms above cutoff.
            idx_out: Indices of atoms below cutoff.
    """
    positions = atoms.get_positions()
    xyz_dict = {i: pos for i, pos in enumerate(atoms.get_positions())}
    symbols = atoms.get_chemical_symbols()
    n_fe = symbols.count("Fe")

    idx_in, idx_out, coords_in, coords_out = [], [], [], []
    for i, pos in enumerate(positions):
        if pos[2] > z_cutoff:
            idx_in.append(i)
            coords_in.append(pos)
        else:
            idx_out.append(i)
            coords_out.append(pos)
    if len(coords_in) % 2 != 0:
        logger.warning("Odd number of H atoms above cutoff: %d", len(coords_in))

    h2_count = len(coords_in) / 2
    logger.info("Initial H2 count: %f", h2_count)
    box_h = Atoms(["H"] * len(coords_in), coords_in, cell=atoms.cell, pbc=[1, 1, 0])
    feh = Atoms(
        [symbols[i] for i in idx_out], coords_out, cell=atoms.cell, pbc=[1, 1, 0]
    )
    return box_h, feh, h2_count, idx_in, idx_out, xyz_dict, n_fe


def insertion_mc(
    h2_template: Atoms,
    bounds: Tuple[float, float, float, float, float, float],
    box_h: Atoms,
    min_dist: float = 2.5,
    max_attempts: int = 50,
) -> Tuple[Atoms, Atoms]:
    """
    Attempt to insert an H2 molecule into the GCMC box without overlap.

    Args:
        h2_template (Atoms): Template H2 molecule positions.
        bounds (tuple): Region bounds as (x0, x1, y0, y1, z0, z1).
        box_h (Atoms): Current hydrogen box.
        min_dist (float): Minimum allowed distance in Å.
        max_attempts (int): Maximum insertion attempts.

    Returns:
        Tuple[Atoms, Atoms]: Inserted H2 and updated box.

    Raises:
        RuntimeError: If no valid insertion found.
    """
    x0, x1, y0, y1, z0, z1 = bounds
    coords = h2_template.get_positions()
    for attempt in range(1, max_attempts + 1):
        offset = np.array(
            [random.uniform(x0, x1), random.uniform(y0, y1), random.uniform(z0, z1)]
        )
        new = Atoms("H2", coords + offset)
        new.rotate(random.uniform(0, 360), 'x', center='COM')
        new.rotate(random.uniform(0, 360), 'y', center='COM')
        new.rotate(random.uniform(0, 360), 'z', center='COM')
        new_pos = new.get_positions()
        old_pos = box_h.get_positions()
        dist_matrix = np.linalg.norm(new_pos[:, None, :] - old_pos[None, :, :], axis=2)
        if np.all(dist_matrix >= min_dist):
            logger.debug("Insertion successful on attempt %d", attempt)
            return new, box_h + new
    logger.warning("Insertion failed after %d attempts", max_attempts)
    raise RuntimeError("Insertion failed")


def deletion_mc(box_h: Atoms) -> Tuple[Atoms, Atoms]:
    """
    Attempt to delete a single H2 pair from the box using a KD-tree.

    Args:
        box_h (Atoms): Current hydrogen box.

    Returns:
        Tuple[Atoms, Atoms]: Deleted H2 and updated box.

    Raises:
        RuntimeError: If no deletable pair is found.
    """
    positions = box_h.get_positions()
    tree = cKDTree(positions)
    pairs = tree.query_pairs(r=0.8)
    if not pairs:
        logger.warning("No close H2 pairs found for deletion")
        raise RuntimeError("Deletion failed")
    i, j = pairs.pop()
    h2 = Atoms("H2", [positions[i], positions[j]])
    selector = [atom.index not in (i, j) for atom in box_h]
    logger.debug("Deleted H2 pair indices (%d, %d)", i, j)
    return h2, box_h[selector]


def metropolis_criteria(
    E_current: float,
    E_new: float,
    T: float,
    m_amu: float,
    mu: float,
    ins: bool,
    V: float,
    n: int,
    h2_en: float,
) -> bool:
    """
    Decide acceptance of a GCMC move via the Metropolis criterion.

    Args:
        E_current (float): Current potential energy (eV).
        E_new (float): Proposed new energy (eV).
        T (float): Temperature (K).
        m_amu (float): Mass in amu.
        mu (float): Chemical potential (eV).
        ins (bool): True if insertion move, False if deletion.
        V (float): System volume (Å³).
        n (int): Current number of H2 molecules.
        h2_en (float): Reference energy of H2 (eV).

    Returns:
        bool: True if move is accepted, False otherwise.
    """
    beta = 1.0 / (kB * T)
    lam = lambda_wl(T, m_amu)
    if ins:
        delta_E = E_new - (E_current + h2_en)
        prob = min(1.0, (V / (lam**3 * (n + 1))) * np.exp(beta * (mu - delta_E)))
        move = "insertion"
    else:
        delta_E = E_new - (E_current - h2_en)
        prob = min(1.0, ((lam**3 * n) / V) * np.exp(-beta * (mu + delta_E)))
        move = "deletion"
    accept = random.random() < prob
    logger.debug(
        "Metropolis %s: ΔE=%.6f, prob=%.6f, accepted=%s", move, delta_E, prob, accept
    )
    return accept


def main() -> None:
    """
    Main driver for the GCMC-LAMMPS coupling process.

    Workflow:
        1. Parse arguments and config file.
        2. Read region bounds and H2 template.
        3. Load the last frame of the LAMMPS trajectory.
        4. Extract the hydrogen box and surface.
        5. Compute initial energy and set parameters.
        6. Perform Monte Carlo insertion/deletion steps, logging extxyz frames.
        7. Merge final box with surface and write LAMMPS data file.
    """
    # 1. Parse arguments
    args = parse_arguments()

    # 2. Preload region bounds and H2 template
    region_bounds = read_region(args.box_path)
    h2_template = read(str(args.h2_path))

    # 3. Load last trajectory frame
    try:
        atoms = get_last_frame(args.traj_path)
    except Exception as e:
        logger.critical(f"Cannot load trajectory: {e}")
        return

    # 4. Extract hydrogen box and surface
    box_h, feh, h2_count, idx_in, idx_out, xyz_dict, n_fe = extract_box_hydrogens(atoms, args.z_cutoff)
    
    if h2_count % 1 == 0:
        logger.info(
            "H2 count is integral (%.1f) at z_cutoff = %.2f Å. No cutoff adjustment needed.",
            h2_count,
            args.z_cutoff,
        )
    else: 
        logger.info(
            "Non-integral H2 count detected (%.1f) at z_cutoff = %.2f Å. "
            "Incrementally adjusting cutoff.",
            h2_count,
            args.z_cutoff,
        )
        
        z_cutoff_incr = 0.1
        
        while h2_count % 1 != 0:
            box_h, feh, h2_count, idx_in, idx_out, xyz_dict, n_fe = extract_box_hydrogens(
                atoms, args.z_cutoff - z_cutoff_incr
            )
        
            logger.info(
                "Adjusted z_cutoff = %.2f Å → H2 count = %.1f",
                args.z_cutoff - z_cutoff_incr,
                h2_count,
            )
        
            z_cutoff_incr += 0.1
        
        logger.info(
            "Integral H2 count achieved (%.0f) at z_cutoff = %.2f Å.",
            h2_count,
            args.z_cutoff - (z_cutoff_incr - 0.1),
        )
          
    restart_dict = parse_lammps_data(args.restart_path)
    logger.info(
        "Extracted Atoms and Velocities blocks from old restart file.")




    # 5. Compute initial energy and set parameters
    try:
        E_current = get_energy_mace(box_h, args.model_dir)
    except Exception as e:
        logger.critical(f"Energy calculation failed: {e}")
        return
    #V = np.prod(atoms.cell.diagonal()) - args.z_cutoff * Angstrom
    V = atoms.cell[0][0]*atoms.cell[1][1]*(atoms.cell[2][2]-args.z_cutoff)
    T = args.temperature
    mu = args.chemical_potential
    m_amu = args.mass
    h2_en = args.h2_energy
    logger.info("Initial H2 count: %f, Initial energy: %.6f eV", h2_count, E_current)

    # 6. Monte Carlo loop
    logger.info("Starting GCMC loop with Δμ termination")
    output_extxyz = Path(args.bdir) / "mc_cycle.extxyz"
    
    tol = 1e-3
    mu_current = compute_chemical_potential(T, m_amu, V, h2_count)
    delta_mu = mu - mu_current
    
    for step in range(1, args.steps + 1):
    
        if abs(delta_mu) <= tol:
            logger.info(
                "Δμ termination reached at step %d: mu=%.6e, mu_current=%.6e, |Δμ|=%.3e",
                step, mu, mu_current, abs(delta_mu)
            )
            break
    
        try:
            # --- unbiased proposal ---
            if random.random() < 0.5:
                _, new_box = insertion_mc(h2_template, region_bounds, box_h)
                E_new = get_energy_mace(new_box, args.model_dir)
                ins = True
            else:
                # Optional guard: avoid deletion when empty
                if h2_count == 0:
                    continue
                _, new_box = deletion_mc(box_h)
                E_new = get_energy_mace(new_box, args.model_dir)
                ins = False
    
            # --- Metropolis acceptance ---
            if metropolis_criteria(
                E_current, E_new, T, m_amu, mu, ins, V, h2_count, h2_en
            ):
                box_h = new_box
                E_current = E_new
                h2_count = len(new_box) // 2
    
                logger.info(
                    "Step %d accepted: n_h2=%d, energy=%.6f eV",
                    step, h2_count, E_current
                )
    
            # --- update μ diagnostics ---
            mu_current = compute_chemical_potential(T, m_amu, V, h2_count)
            delta_mu = mu - mu_current
    
            logger.debug(
                "μ diagnostics: mu=%.6e, mu_current=%.6e, Δμ=%.3e",
                mu, mu_current, delta_mu
            )
    
            write_extxyz_sequence(output_extxyz, box_h)
    
        except Exception as e:
            logger.error("MC step %d failed: %s", step, e)
            break
    
    logger.info("GCMC loop finished at step %d", step)
    
    # 7. Merge and write new initial data for next step 
    new_struct = feh + box_h
    
    new_N = new_struct.get_global_number_of_atoms()
    new_in_list_pos = []
    new_in_list_vel = []
    cell_new_in_list_pos = []
    cell_new_in_list_vel = []
    
    # identify unchanged molecules in the pressure-controlled region 
    common_h, not_common_h = get_matching_h(box_h.get_positions(),idx_in)
    org_id_list = filter_dict(xyz_dict, common_h)

    for id_num in org_id_list:
        rec = restart_dict[id_num]
        new_in_list_pos.append([*rec[:3], *map(str, rec[3:6])])
        new_in_list_vel.append(rec[6:9])


    # assign initial velocities to the newly added molecules 
    for xyz in not_common_h:
        new_in_list_pos.extend([xyz.tolist()+ ['0']*3])
        new_v = [i/100 for i in boltzmann_velocity_distribution(T, m_amu/2)]
        new_in_list_vel.append(new_v)

    for id_num in idx_out:
        rec = restart_dict[id_num]
        cell_new_in_list_pos.append([*rec[:3], *map(str, rec[3:6])])
        cell_new_in_list_vel.append(rec[6:9])
    

    # generate final list with positions and velocities 
    cell_new_in_list_pos.extend(new_in_list_pos) 
    cell_new_in_list_vel.extend(new_in_list_vel) 

    flatten_list = reduce(operator.concat, cell_new_in_list_pos)
    new_coord = np.asarray(flatten_list,dtype=object).reshape(new_N,6)
    type_list = [1] *n_fe + [2]*(new_N-n_fe)   
    type_arr = np.asarray(type_list,dtype=int).reshape(new_N,1)
    num_arr = np.asarray(range(1,new_N+1), dtype=int).reshape(new_N,1)

    xyz_c = np.concatenate((num_arr.astype(str), type_arr.astype(str)),axis=1)
    pos_block = np.concatenate((xyz_c.astype(str), new_coord.astype(str)),axis=1)

    vel_arr = np.asarray(cell_new_in_list_vel).reshape(new_N,3)
    vel_block = np.concatenate((num_arr.astype(str), vel_arr),axis=1)

    pos_string = write_string(pos_block)
    vel_string = write_string(vel_block)

    lammpsdata_string = write_lammpsdata(
    num_atoms=new_N,
    positions=pos_string,
    velocities=vel_string,
    x_cell=atoms.cell[0, 0],
    y_cell=atoms.cell[1, 1],
    z_cell=atoms.cell[2, 2],
    elements=args.elements)

    bdir = Path(args.bdir)
    data_file = bdir / 'initial.lammpsdata'
    data_file.write_text(lammpsdata_string)

    logger.info(
        "Finished GCMC. Final H2 count: %d, Final energy: %.6f eV", h2_count, E_current
    )

    logger.info(
        "Written initial.lammpsdata")

if __name__ == "__main__":
    main()
