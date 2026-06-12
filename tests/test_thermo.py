import sys
from pathlib import Path

import pytest

pytest.importorskip("numpy")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from comet.physics.constants import kB_eV
from comet.physics.thermo import chemical_potentials_from_particles, lambda_wl


def test_lambda_wl_positive():
    wavelength = lambda_wl(T=300.0, m_amu=2.0)
    assert isinstance(wavelength, float)
    assert wavelength > 0


def test_chemical_potentials_from_particles_matches_ideal_gas_formula():
    # μ = kB T ln((n/V) λ³) for an ideal gas; check the density-based μ against
    # the closed-form value for known inputs (not just self-consistency).
    T, V, n, m = 300.0, 1000.0, 10, 2.0

    mu = chemical_potentials_from_particles(
        T=T, V_A3=V, gas_counts={"H2": n}, gas_masses={"H2": m}
    )

    lam = lambda_wl(T, m)
    expected = kB_eV * T * np.log((n / V) * lam ** 3)
    assert mu["H2"] == pytest.approx(expected)
