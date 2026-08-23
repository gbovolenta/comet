"""`mole_fractions` + total `pressure` input: resolution and validation."""

import sys
from pathlib import Path

import pytest

pytest.importorskip("pydantic")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from comet.config.schema import RunConfig

BASE = dict(
    energy_backend="orca",
    restart_path="restart.lammps",
    gas_template_dir="templates/",
    elements=["Fe", "N", "H"],
    gas_list=["H2", "N2"],
    z_cutoff=15.0,
    temperature=700.0,
    steps=10,
)


def _config(**overrides):
    return RunConfig(**{**BASE, **overrides})


def test_mole_fractions_resolve_to_partial_pressures():
    config = _config(pressure=150.0, mole_fractions={"H2": 0.6, "N2": 0.4})
    assert config.partial_pressures == pytest.approx({"H2": 90.0, "N2": 60.0})
    assert config.mole_fractions == {"H2": 0.6, "N2": 0.4}   # kept for logging


def test_omitted_species_stays_frozen():
    config = _config(pressure=10.0, mole_fractions={"H2": 1.0})
    assert config.partial_pressures == pytest.approx({"H2": 10.0})
    assert "N2" not in config.partial_pressures   # frozen, same as with partials


def test_fractions_may_sum_below_one():
    config = _config(pressure=100.0, mole_fractions={"H2": 0.5, "N2": 0.2})
    assert config.partial_pressures == pytest.approx({"H2": 50.0, "N2": 20.0})


def test_mole_fractions_require_total_pressure():
    with pytest.raises(ValueError, match="requires a total 'pressure'"):
        _config(mole_fractions={"H2": 1.0})


def test_mole_fractions_exclusive_with_partial_pressures():
    with pytest.raises(ValueError, match="not both"):
        _config(
            pressure=10.0,
            mole_fractions={"H2": 0.5},
            partial_pressures={"N2": 5.0},
        )


def test_fractions_must_not_exceed_one_total():
    with pytest.raises(ValueError, match="sum to"):
        _config(pressure=10.0, mole_fractions={"H2": 0.7, "N2": 0.4})


def test_fraction_out_of_range_rejected():
    with pytest.raises(ValueError, match="must be in"):
        _config(pressure=10.0, mole_fractions={"H2": 1.4})
    with pytest.raises(ValueError, match="must be in"):
        _config(pressure=10.0, mole_fractions={"H2": 0.0})


def test_unknown_species_rejected():
    with pytest.raises(ValueError, match="not in gas_list"):
        _config(pressure=10.0, mole_fractions={"CO2": 1.0})


def test_partial_pressures_still_accepted():
    config = _config(partial_pressures={"H2": 1.5, "N2": 0.5})
    assert config.partial_pressures == {"H2": 1.5, "N2": 0.5}
    assert config.mole_fractions is None
