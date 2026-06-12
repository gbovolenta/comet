import numpy as np


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
