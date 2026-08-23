import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from comet.physics.constants import Pa_to_eV_per_A3
from comet.physics.thermo import (
    chemical_potential_pure,
    chemical_potentials_binary_mixture,
    chemical_potentials_from_particles,
    compute_chemical_potentials,
    compute_pressure_atm,
    count_convergence_status,
    load_gas_masses,
    pressure_to_eV_per_A3,
    quantized_target_counts,
    target_counts_from_mu,
)


def test_pressure_to_ev_per_a3_supported_units():
    assert pressure_to_eV_per_A3(2.0, "Pa") == pytest.approx(2.0 * Pa_to_eV_per_A3)
    assert pressure_to_eV_per_A3(2.0, "bar") == pytest.approx(2.0e5 * Pa_to_eV_per_A3)
    assert pressure_to_eV_per_A3(2.0, "atm") == pytest.approx(2.0 * 101325.0 * Pa_to_eV_per_A3)
    assert pressure_to_eV_per_A3(2.0, "torr") == pytest.approx(2.0 * 133.322368 * Pa_to_eV_per_A3)


def test_pressure_to_ev_per_a3_invalid_unit():
    with pytest.raises(ValueError):
        pressure_to_eV_per_A3(1.0, "psi")


def test_compute_pressure_atm_zero_particles_and_invalid_volume():
    assert compute_pressure_atm(T=300.0, V_A3=1000.0, n=0) == pytest.approx(0.0)

    with pytest.raises(ValueError):
        compute_pressure_atm(T=300.0, V_A3=0.0, n=1)

    with pytest.raises(ValueError):
        compute_pressure_atm(T=300.0, V_A3=1000.0, n=-1)


def test_chemical_potentials_from_particles_handles_inactive_species():
    mu = chemical_potentials_from_particles(
        T=300.0,
        V_A3=1000.0,
        gas_counts={"H2": 10, "N2": 0},
        gas_masses={"H2": 2.0, "N2": 28.0},
    )

    assert np.isfinite(mu["H2"])
    assert mu["N2"] == -np.inf


def test_chemical_potentials_from_particles_missing_mass():
    with pytest.raises(KeyError):
        chemical_potentials_from_particles(
            T=300.0,
            V_A3=1000.0,
            gas_counts={"H2": 1},
            gas_masses={},
        )


def test_chemical_potential_pure_matches_single_species_dispatch():
    mu_direct = chemical_potential_pure(T=300.0, m_amu=2.0, p=1.0, p_unit="bar")
    mu_dispatch = compute_chemical_potentials(
        T=300.0,
        gas_dict={"H2": 2.0},
        pressure=1.0,
        pressure_unit="bar",
        y1=1.0,
    )

    assert mu_dispatch == {"H2": pytest.approx(mu_direct)}


def test_binary_mixture_dispatch_matches_direct_call():
    mu1, mu2 = chemical_potentials_binary_mixture(
        T=300.0,
        P_total=1.0,
        P_unit="bar",
        y1=0.25,
        m1_amu=2.0,
        m2_amu=28.0,
    )
    mu_dispatch = compute_chemical_potentials(
        T=300.0,
        gas_dict={"H2": 2.0, "N2": 28.0},
        pressure=1.0,
        pressure_unit="bar",
        y1=0.25,
    )

    assert mu_dispatch["H2"] == pytest.approx(mu1)
    assert mu_dispatch["N2"] == pytest.approx(mu2)


def test_compute_chemical_potentials_partial_pressures_per_species():
    mu = compute_chemical_potentials(
        T=300.0,
        gas_dict={"H2": 2.0, "N2": 28.0},
        partial_pressures={"H2": 0.5, "N2": 1.5},
        pressure_unit="bar",
    )
    assert mu["H2"] == pytest.approx(chemical_potential_pure(300.0, 2.0, 0.5, "bar"))
    assert mu["N2"] == pytest.approx(chemical_potential_pure(300.0, 28.0, 1.5, "bar"))


def test_compute_chemical_potentials_freeze_by_omission():
    # A species omitted from partial_pressures is frozen (mu_target = -inf).
    mu = compute_chemical_potentials(
        T=300.0,
        gas_dict={"H2": 2.0, "N2": 28.0},
        partial_pressures={"H2": 0.5},
        pressure_unit="bar",
    )
    assert np.isfinite(mu["H2"])
    assert mu["N2"] == -np.inf


def test_compute_chemical_potentials_rejects_more_than_two_species():
    with pytest.raises(NotImplementedError):
        compute_chemical_potentials(
            T=300.0,
            gas_dict={"H2": 2.0, "N2": 28.0, "CO2": 44.0},
            pressure=1.0,
            pressure_unit="bar",
            y1=0.5,
        )


def test_load_gas_masses_validates_parallel_lists():
    assert load_gas_masses(["H2", "N2"], [2.0, 28.0]) == {"H2": 2.0, "N2": 28.0}

    with pytest.raises(ValueError):
        load_gas_masses(["H2"], [2.0, 28.0])


def test_target_counts_from_mu_roundtrips_pressure():
    from comet.physics.constants import atm_to_Pa, kB_J

    T, V_A3, m_amu, p_atm = 723.0, 11650.0, 2.016, 100.0
    mu = chemical_potential_pure(T, m_amu, p_atm, "atm")
    targets = target_counts_from_mu(T, V_A3, {"H2": mu}, {"H2": m_amu})

    # Analytic ideal-gas count: N* = p V / (kB T)
    expected = (p_atm * atm_to_Pa) * (V_A3 * 1e-30) / (kB_J * T)
    assert targets["H2"] == pytest.approx(expected, rel=1e-6)


def _mu_for(T, masses, pressures_atm):
    return {
        g: chemical_potential_pure(T, masses[g], p, "atm")
        for g, p in pressures_atm.items()
    }


def test_quantized_targets_preserve_ratio_on_lattice():
    # 3:1 at conditions where N* = (13.3, 4.4): total 17.7 -> nearest multiple
    # of 4 is 16 -> exact 12:4 split.
    T, V = 723.0, 11649.0
    masses = {"H2": 2.016, "N2": 28.014}
    mu = _mu_for(T, masses, {"H2": 112.5, "N2": 37.5})
    targets = quantized_target_counts(T, V, mu, masses, ratios={"H2": 3, "N2": 1})
    assert targets == {"H2": 12, "N2": 4}


def test_quantized_targets_fail_below_one_composition_unit():
    # Same composition at 1/100th the pressure: N*_tot << one unit of 4.
    T, V = 723.0, 11649.0
    masses = {"H2": 2.016, "N2": 28.014}
    mu = _mu_for(T, masses, {"H2": 1.125, "N2": 0.375})
    with pytest.raises(ValueError, match="Increase the gas volume"):
        quantized_target_counts(T, V, mu, masses, ratios={"H2": 3, "N2": 1})


def test_quantized_targets_partial_pressure_mode_rounds_independently():
    T, V = 723.0, 11649.0
    masses = {"H2": 2.016, "N2": 28.014}
    mu = _mu_for(T, masses, {"H2": 112.5, "N2": 37.5})
    targets = quantized_target_counts(T, V, mu, masses, ratios=None)
    assert targets == {"H2": 13, "N2": 4}   # round(13.3), round(4.4)


def test_quantized_targets_partial_pressure_mode_fails_on_zero_target():
    T, V = 723.0, 11649.0
    masses = {"H2": 2.016}
    mu = _mu_for(T, masses, {"H2": 1.0})    # N* ~ 0.12
    with pytest.raises(ValueError, match="rounds to zero"):
        quantized_target_counts(T, V, mu, masses, ratios=None)


def test_quantized_targets_inactive_species_zero():
    T, V = 723.0, 11649.0
    masses = {"H2": 2.016, "N2": 28.014}
    mu = _mu_for(T, masses, {"H2": 112.5})
    mu["N2"] = -np.inf
    targets = quantized_target_counts(T, V, mu, masses, ratios={"H2": 1})
    assert targets["N2"] == 0
    assert targets["H2"] >= 1


def test_count_convergence_status_classifies_by_integer_count():
    inactive, converged, unconverged = count_convergence_status(
        n_targets={"H2": 12, "N2": 4, "CO2": 0},
        gas_counts={"H2": 12, "N2": 6, "CO2": 3},
        mu_target={"H2": -0.8, "N2": -1.1, "CO2": -np.inf},
    )
    assert inactive == {"CO2"}
    assert converged == {"H2"}
    assert unconverged == {"N2"}


