"""`ratios` + total `pressure` input: resolution and validation."""

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


def test_ratios_resolve_to_partial_pressures():
    config = _config(pressure=150.0, ratios={"H2": 3, "N2": 1})
    assert config.partial_pressures == pytest.approx({"H2": 112.5, "N2": 37.5})
    assert config.ratios == {"H2": 3, "N2": 1}   # retained for target quantization


def test_omitted_species_stays_frozen():
    config = _config(pressure=10.0, ratios={"H2": 1})
    assert config.partial_pressures == pytest.approx({"H2": 10.0})
    assert "N2" not in config.partial_pressures


def test_ratios_require_total_pressure():
    with pytest.raises(ValueError, match="requires a total 'pressure'"):
        _config(ratios={"H2": 1})


def test_ratios_exclusive_with_partial_pressures():
    with pytest.raises(ValueError, match="not both"):
        _config(pressure=10.0, ratios={"H2": 1}, partial_pressures={"N2": 5.0})


def test_non_positive_ratio_rejected():
    with pytest.raises(ValueError, match="positive integer"):
        _config(pressure=10.0, ratios={"H2": 0})


def test_non_integer_ratio_rejected():
    with pytest.raises(ValueError):
        _config(pressure=10.0, ratios={"H2": 2.5})


def test_unknown_species_rejected():
    with pytest.raises(ValueError, match="not in gas_list"):
        _config(pressure=10.0, ratios={"CO2": 1})


def test_partial_pressures_still_accepted():
    config = _config(partial_pressures={"H2": 100.0, "N2": 50.0})
    assert config.partial_pressures == {"H2": 100.0, "N2": 50.0}
    assert config.ratios is None
