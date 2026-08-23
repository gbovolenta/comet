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

def target_counts_from_mu(
    T: float,
    V_A3: float,
    mu_dict: Dict[str, float],
    gas_masses: Dict[str, float],
) -> Dict[str, float]:
    """
    Ideal-gas expected molecule counts implied by the target chemical potentials.

    Inverts μ = kB T ln(N λ³ / V): N* = (V / λ³) · exp(μ / kB T), which equals
    p_i·V/(kB T) regardless of how the pressures were specified. Inactive
    species (μ = -inf) map to 0.

    Returns:
        Dict {gas_name: N*_i} (float; the grand-canonical ensemble average).
    """
    if V_A3 <= 0:
        raise ValueError("V_A3 must be > 0")

    targets: Dict[str, float] = {}
    for gas, mu in mu_dict.items():
        if not np.isfinite(mu):
            targets[gas] = 0.0
            continue
        if gas not in gas_masses:
            raise KeyError(f"Missing mass for gas '{gas}'")
        lam = lambda_wl(T, gas_masses[gas])
        targets[gas] = float((V_A3 / lam**3) * np.exp(mu / (kB_eV * T)))
    return targets


def quantized_target_counts(
    T: float,
    V_A3: float,
    mu_dict: Dict[str, float],
    gas_masses: Dict[str, float],
    ratios: Dict[str, int] | None = None,
) -> Dict[str, int]:
    """
    Integer per-species convergence targets nearest the ideal-gas expectations.

    With ``ratios`` (integer composition), the total count is quantized to the
    nearest multiple of Σr and split exactly as r_i : ... — the composition is
    preserved by construction. Without ratios (absolute partial pressures),
    each active species is rounded independently; no composition constraint is
    claimed.

    There is deliberately no fallback: if the gas volume cannot accommodate the
    requested state (total quantizes to zero, or an active species' target
    rounds to zero), a ValueError is raised stating the minimum factor by which
    the gas volume or the pressure must be increased.

    Returns:
        Dict {gas_name: integer target}; inactive species map to 0.
    """
    n_star = target_counts_from_mu(T, V_A3, mu_dict, gas_masses)
    active = [g for g, mu in mu_dict.items() if np.isfinite(mu)]
    targets: Dict[str, int] = {g: 0 for g in mu_dict}
    if not active:
        return targets

    if ratios is not None:
        missing = [g for g in active if g not in ratios]
        if missing:
            raise ValueError(f"ratios missing active species: {sorted(missing)}")
        r_sum = sum(int(ratios[g]) for g in active)
        n_star_tot = sum(n_star[g] for g in active)
        units = round(n_star_tot / r_sum)
        if units < 1:
            factor = np.inf if n_star_tot <= 0 else r_sum / n_star_tot
            raise ValueError(
                f"Expected counts N* = { {g: round(n_star[g], 2) for g in active} } "
                f"sum to {n_star_tot:.2f}, below one composition unit "
                f"({r_sum} molecules for ratios "
                f"{':'.join(str(ratios[g]) for g in active)}). Increase the gas "
                f"volume or the total pressure by a factor >= {factor:.1f}."
            )
        for g in active:
            targets[g] = int(ratios[g]) * int(units)
        return targets

    for g in active:
        t = round(n_star[g])
        if t < 1:
            raise ValueError(
                f"Expected count N*({g}) = {n_star[g]:.3f} rounds to zero at the "
                f"requested partial pressure. Increase the gas volume or the "
                f"partial pressure of {g} by a factor >= {0.5 / n_star[g]:.1f}."
            )
        targets[g] = int(t)
    return targets


def count_convergence_status(
    n_targets: Dict[str, int],
    gas_counts: Dict[str, int],
    mu_target: Dict[str, float],
) -> Tuple[Set[str], Set[str], Set[str]]:
    """
    Classify species as (inactive, converged, unconverged) by integer count.

    Rules:
      - inactive: mu_target is non-finite (frozen species)
      - converged: current molecule count equals the integer target
      - unconverged: otherwise

    Returns:
      (inactive, converged, unconverged)
    """
    inactive: Set[str] = set()
    converged: Set[str] = set()
    unconverged: Set[str] = set()

    for gas, mu_t in mu_target.items():
        if not np.isfinite(mu_t):
            inactive.add(gas)
        elif int(gas_counts.get(gas, 0)) == int(n_targets.get(gas, 0)):
            converged.add(gas)
        else:
            unconverged.add(gas)

    return inactive, converged, unconverged
