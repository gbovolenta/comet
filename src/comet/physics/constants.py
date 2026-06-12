from ase.units import eV
from scipy.constants import hbar as hbar_SI

hbar = hbar_SI / eV  # convert J·s → eV·s
amu_to_kg = 1.66053906660e-27  # Conversion factor for amu to kg
h_eV_s = 4.135667696e-15    # Planck constant in eV·s
kB_eV = 8.617333262145e-5  # Boltzmann constant in eV/K
eV_to_J = 1.602176634e-19         # J/eV
kB_J = kB_eV * eV_to_J            # J/K
h_J_s = 6.62607015e-34            # J*s
angstrom_to_m = 1e-10             # m
Pa_to_eV_per_A3 = 1e-30 / eV_to_J


