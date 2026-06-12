import sys
from pathlib import Path

import pytest

pytest.importorskip("numpy")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from comet.physics.thermo import compute_chemical_potential, lambda_wl


def test_lambda_wl_positive():
    wavelength = lambda_wl(T=300.0, m_amu=2.0)
    assert isinstance(wavelength, float)
    assert wavelength > 0


def test_compute_chemical_potential_deterministic():
    params = dict(T=300.0, m_amu=2.0, V=1000.0, n=10)
    mu1 = compute_chemical_potential(**params)
    mu2 = compute_chemical_potential(**params)
    assert isinstance(mu1, float)
    assert mu1 == mu2
