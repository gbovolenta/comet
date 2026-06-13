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
    filter_gas_templates_by_species,
    load_gas_masses,
    mu_convergence_status,
    mu_converged,
    pressure_to_eV_per_A3,
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


def test_mu_convergence_status_and_wrapper():
    inactive, converged, unconverged = mu_convergence_status(
        mu_target={"H2": 1.0, "N2": -np.inf, "CO2": 0.5},
        mu_current={"H2": 1.01, "N2": 99.0, "CO2": np.inf},
        tol=0.05,
    )

    assert inactive == {"N2"}
    assert converged == {"H2"}
    assert unconverged == {"CO2"}
    assert not mu_converged(
        mu_target={"H2": 1.0, "N2": -np.inf, "CO2": 0.5},
        mu_current={"H2": 1.01, "N2": 99.0, "CO2": np.inf},
        tol=0.05,
    )


def test_mu_convergence_status_validates_inputs():
    with pytest.raises(TypeError):
        mu_convergence_status([], {}, 0.1)

    with pytest.raises(KeyError):
        mu_convergence_status({"H2": 1.0}, {}, 0.1)


def test_filter_gas_templates_by_species_filters_only_requested_keys():
    templates = {"H2": "h2-template", "N2": "n2-template", "CO2": "co2-template"}

    filtered = filter_gas_templates_by_species(templates, {"H2", "CO2"})

    assert filtered == {"H2": "h2-template", "CO2": "co2-template"}
