import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from comet.mc.accept import metropolis_criteria
from comet.physics.velocities import boltzmann_velocity_distribution


def test_metropolis_accepts_high_probability_insertion(monkeypatch):
    monkeypatch.setattr("comet.mc.accept.random.random", lambda: 0.5)

    accepted = metropolis_criteria(
        E_current=0.0,
        E_new=0.0,
        T=300.0,
        m_amu=2.0,
        mu=10.0,
        ins=True,
        V=1000.0,
        n=1,
        gas_en=0.0,
    )

    assert accepted is True


def test_metropolis_rejects_low_probability_deletion(monkeypatch):
    monkeypatch.setattr("comet.mc.accept.random.random", lambda: 0.5)

    accepted = metropolis_criteria(
        E_current=0.0,
        E_new=0.0,
        T=300.0,
        m_amu=2.0,
        mu=10.0,
        ins=False,
        V=1000.0,
        n=1,
        gas_en=0.0,
    )

    assert not accepted


def test_boltzmann_velocity_distribution_uses_expected_gaussian_scale(monkeypatch):
    calls = []
    returned = iter([1.0, 2.0, 3.0])

    def fake_normal(mean, std):
        calls.append((mean, std))
        return next(returned)

    monkeypatch.setattr(np.random, "normal", fake_normal)

    velocities = boltzmann_velocity_distribution(T=300.0, mass=2.0)

    k_B = 1.380649e-23
    amu_to_kg = 1.66053906660e-27
    expected_std = np.sqrt(k_B * 300.0 / (2.0 * amu_to_kg))

    assert velocities == [1.0, 2.0, 3.0]
    assert calls == [(0, pytest.approx(expected_std))] * 3
