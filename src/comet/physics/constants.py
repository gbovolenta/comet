"""Physical constants and unit conversions, sourced from scipy.constants (CODATA).

Sourcing from :mod:`scipy.constants` rather than hand-typing values removes
transcription risk and keeps everything on one CODATA release. The base SI
constants below are exact by definition (2019 SI redefinition); ``m_u`` is the
CODATA atomic-mass constant.
"""

from scipy import constants as _sc

# --- base SI constants ---
eV_to_J = _sc.eV            # J per eV
kB_J = _sc.k               # Boltzmann constant [J/K]
h_J_s = _sc.h              # Planck constant [J*s]
amu_to_kg = _sc.m_u        # atomic mass constant [kg]
angstrom_to_m = _sc.angstrom  # [m]
atm_to_Pa = _sc.atm        # standard atmosphere [Pa] (= 101325)

# --- eV-based forms ---
h_eV_s = h_J_s / eV_to_J   # Planck constant [eV*s]
kB_eV = kB_J / eV_to_J     # Boltzmann constant [eV/K]
hbar_eV = _sc.hbar / eV_to_J  # reduced Planck constant [eV*s]

# --- derived unit conversion ---
# 1 Pa = 1 J/m^3 -> eV/Angstrom^3  (J->eV: /eV_to_J;  m^3->Angstrom^3: *1e-30)
Pa_to_eV_per_A3 = 1e-30 / eV_to_J
