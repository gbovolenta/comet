import logging
from typing import Dict, Set, Tuple

import numpy as np

from comet.physics.constants import (
    amu_to_kg,
    atm_to_Pa,
    h_eV_s,
    kB_J,
    kB_eV,
    Pa_to_eV_per_A3,
)

logger = logging.getLogger(__name__)


def pressure_to_eV_per_A3(P: float, unit: str = "Pa") -> float:
    """
    Convert pressure to eV/Å^3.

    Supported units: "Pa", "bar", "atm", "torr"
    """
    u = unit.lower()
    if u == "pa":
        return P * Pa_to_eV_per_A3
    if u == "bar":
        return (P * 1e5) * Pa_to_eV_per_A3
    if u == "atm":
        return (P * atm_to_Pa) * Pa_to_eV_per_A3
    if u == "torr":
        return (P * 133.322368) * Pa_to_eV_per_A3
    raise ValueError(f"Unsupported pressure unit: {unit!r}")


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


def compute_pressure_atm(T: float, V_A3: float, n: int) -> float:
    """
    Compute ideal-gas pressure from particle count.

    Uses P = N k_B T / V.

    Args:
        T: Temperature [K]
        V_A3: Volume [Å^3]
        n: Number of particles (total), counted consistently with V.

    Returns:
        Pressure [atm]
    """
    if V_A3 <= 0:
        raise ValueError("V_A3 must be > 0")
    if n < 0:
        raise ValueError("n must be >= 0")

    V_m3 = V_A3 * 1e-30                # 1 Å^3 = 1e-30 m^3
    P_Pa = (n * kB_J * T) / V_m3
    P_atm = P_Pa / atm_to_Pa
    logger.debug("Computed pressure: %.6f atm", P_atm)
    return P_atm


def chemical_potentials_from_particles(
    T: float,
    V_A3: float,
    gas_counts: Dict[str, int],
    gas_masses: Dict[str, float],
) -> Dict[str, float]:
    """
    Compute ideal-gas chemical potentials for a gas mixture.

    Args:
        T: Temperature [K]
        V_A3: Volume [Å^3] (shared by all gases)
        gas_counts: Dict {gas_name: number_of_molecules}
        gas_masses: Dict {gas_name: molecular_mass_amu}

    Returns:
        Dict {gas_name: mu_i [eV]}
    """
    if V_A3 <= 0:
        raise ValueError("V_A3 must be > 0")

    mu_dict: Dict[str, float] = {}

    for gas, n_i in gas_counts.items():
        if n_i <= 0:
            # Convention: no particles → -inf chemical potential
            mu_dict[gas] = -np.inf
            continue

        if gas not in gas_masses:
            raise KeyError(f"Missing mass for gas '{gas}'")

        m_amu = gas_masses[gas]
        lam = lambda_wl(T, m_amu)   # thermal de Broglie wavelength [Å]
        density = n_i / V_A3        # number density [Å^-3]

        mu = kB_eV * T * np.log(density * lam**3)
        mu_dict[gas] = float(mu)

        logger.debug(
            "μ(%s): n=%d, ρ=%.3e Å⁻³, μ=%.6f eV",
            gas, n_i, density, mu
        )

    return mu_dict


def chemical_potential_pure(T: float, m_amu: float, p: float, p_unit: str = "bar") -> float:
    """
    Chemical potential for a *pure* ideal gas using pressure:
        μ = kB T ln( (p λ^3) / (kB T) )

    Args:
        T: temperature [K]
        m_amu: mass [amu]
        p: pressure
        p_unit: unit of p ("Pa", "bar", "atm", "torr")

    Returns:
        μ [eV]
    """
    lam = lambda_wl(T, m_amu)                  # Å
    p_eV_A3 = pressure_to_eV_per_A3(p, p_unit) # eV/Å^3

    arg = (p_eV_A3 * lam**3) / (kB_eV * T)
    mu = kB_eV * T * np.log(arg)

    logger.debug("μ(pure): %.6f eV (arg=%.6e)", mu, arg)
    return mu

def load_gas_masses(gas_list, gas_masses):
    """
    Map gas names to their molecular masses.
    """
    if len(gas_list) != len(gas_masses):
        raise ValueError("gas_list and gas_masses must have the same length")

    return dict(zip(gas_list, gas_masses))


def chemical_potentials_binary_mixture(
    T: float,
    P_total: float,
    P_unit: str,
    y1: float,
    m1_amu: float,
    m2_amu: float
) -> tuple[float, float]:
    """
    Chemical potentials for a binary ideal-gas mixture at total pressure P_total.

    For ideal mixtures:
        p1 = y1 * P_total
        p2 = (1 - y1) * P_total
        μ_i = kB T ln( (p_i λ_i^3)/(kB T) )

    Args:
        T: temperature [K]
        P_total: total pressure of mixture
        y1: mole fraction of species 1 (0<y1<1)
        m1_amu: mass of species 1 [amu]
        m2_amu: mass of species 2 [amu]
        P_unit: unit of P_total ("Pa", "bar", "atm", "torr")

    Returns:
        (μ1, μ2) in eV
    """
    if not (0.0 <= y1 <= 1.0):
        raise ValueError("y1 must be between 0 and 1")
    if P_total <= 0:
        raise ValueError("P_total must be > 0")

    y2 = 1.0 - y1
    p1 = y1 * P_total
    p2 = y2 * P_total

    mu1 = chemical_potential_pure(T, m1_amu, p1, P_unit)
    mu2 = chemical_potential_pure(T, m2_amu, p2, P_unit)

    logger.debug("Binary mixture: y1=%.6f y2=%.6f P=%g %s -> μ1=%.6f eV μ2=%.6f eV",
                 y1, y2, P_total, P_unit, mu1, mu2)
    return mu1, mu2

def compute_chemical_potentials(
    T: float,
    gas_dict: dict,
    pressure: float | None = None,
    pressure_unit: str = "bar",
    y1: float = 0.75,
    partial_pressures: dict | None = None,
):
    """
    Compute target chemical potentials for a gas system.

    Two input modes are supported:

    * **Per-species partial pressures (preferred).** Pass ``partial_pressures``
      as a ``{gas_name: partial_pressure}`` mapping. Each species' target μ is
      computed from its own partial pressure via ``chemical_potential_pure``.
      A species present in ``gas_dict`` but omitted from ``partial_pressures``
      (or given a non-positive value) is treated as **inactive/frozen** and
      assigned ``μ = -inf`` — it is never inserted/deleted but still counted
      and logged. Works for any number of species.

    * **Total pressure + mole fraction (legacy fallback).** When
      ``partial_pressures`` is ``None``, behaves as before: a single species is
      a pure gas at ``pressure``; two species are a binary mixture split by the
      mole fraction ``y1``; three or more raise ``NotImplementedError``.

    Always returns:
        Dict[str, float]: {gas_name: mu}

    Args:
        T: Temperature (K)
        gas_dict: {"H2": mass_H2, "N2": mass_N2, ...}
        pressure: Total pressure (legacy fallback mode only)
        pressure_unit: Unit of pressure / partial pressures
        y1: Mole fraction of species 1 (legacy binary fallback only)
        partial_pressures: Optional {gas_name: partial_pressure} mapping
    """
    # Preferred path: explicit per-species partial pressures.
    if partial_pressures is not None:
        mu_dict: dict = {}
        for gas, m_amu in gas_dict.items():
            p_i = partial_pressures.get(gas)
            if p_i is None or float(p_i) <= 0.0:
                # Omitted / non-positive partial pressure => frozen (inactive).
                mu_dict[gas] = -np.inf
            else:
                mu_dict[gas] = chemical_potential_pure(T, m_amu, float(p_i), pressure_unit)
        return mu_dict

    # Legacy fallback path: total pressure split by mole fraction.
    if pressure is None:
        raise ValueError(
            "compute_chemical_potentials requires either 'partial_pressures' or 'pressure'"
        )

    ngas = len(gas_dict)

    if ngas == 1:
        gas, m_amu = next(iter(gas_dict.items()))
        mu = chemical_potential_pure(
            T,
            m_amu,
            pressure,
            pressure_unit,
        )
        return {gas: mu}

    elif ngas == 2:
        (gas1, m1), (gas2, m2) = gas_dict.items()

        mu1, mu2 = chemical_potentials_binary_mixture(
            T,
            pressure,
            pressure_unit,
            y1,
            m1,
            m2,
        )

        return {
            gas1: mu1,
            gas2: mu2,
        }

    else:
        raise NotImplementedError(
            f"Chemical potentials for {ngas}-component mixtures not implemented"
        )

def mu_converged(mu_target: dict, mu_current: dict, tol: float) -> bool:
    """Return whether every active species satisfies the μ tolerance.

    Args:
        mu_target: Mapping from species name to target chemical potential.
        mu_current: Mapping from species name to current chemical potential.
        tol: Absolute convergence tolerance in eV.

    Returns:
        bool: `True` when no active species remain unconverged.
    """
    _, _, unconverged = mu_convergence_status(mu_target, mu_current, tol)
    return len(unconverged) == 0


def mu_convergence_status(
    mu_target: Dict[str, float],
    mu_current: Dict[str, float],
    tol: float,
) -> Tuple[Set[str], Set[str], Set[str]]:
    """
    Classify species as (inactive, converged, unconverged).

    Rules:
      - inactive: mu_target is non-finite (e.g. -inf) => ignore from convergence
      - converged: finite target & finite current & |Δμ| <= tol
      - unconverged: finite target but (current non-finite OR |Δμ| > tol)

    Returns:
      (inactive, converged, unconverged)
    """
    if not isinstance(mu_target, dict) or not isinstance(mu_current, dict):
        raise TypeError(
            f"mu_convergence_status expects dicts; got {type(mu_target)}, {type(mu_current)}"
        )

    inactive: Set[str] = set()
    converged: Set[str] = set()
    unconverged: Set[str] = set()

    for gas, mu_t in mu_target.items():
        if gas not in mu_current:
            raise KeyError(f"Missing mu_current for species '{gas}'")

        if not np.isfinite(mu_t):
            inactive.add(gas)
            continue

        mu_c = mu_current[gas]
        if (not np.isfinite(mu_c)) or (abs(mu_t - mu_c) > tol):
            unconverged.add(gas)
        else:
            converged.add(gas)

    return inactive, converged, unconverged

def filter_gas_templates_by_species(gas_templates: dict, species_to_keep: set) -> dict:
    """Filter a template dictionary to a selected set of species.

    Args:
        gas_templates: Mapping from species name to ASE template.
        species_to_keep: Species names to retain.

    Returns:
        dict: Filtered gas-template mapping.
    """
    filtered = {g: t for g, t in gas_templates.items() if g in species_to_keep}
    removed = set(gas_templates) - set(filtered)
    if removed:
        logger.info("Filtered out templates: %s", sorted(removed))
    return filtered
