"""Validation tests for the typed RunConfig (the 0.3.0 config layer)."""

import sys
from pathlib import Path

import pytest

pytest.importorskip("pydantic")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pydantic import ValidationError

from comet.config.schema import RunConfig, load_run_config


def _base(**override):
    """A valid config dict (paths need not exist — the schema doesn't stat them)."""
    d = dict(
        energy_backend="orca",
        restart_path="restart.lammps",
        gas_template_dir="templates/",
        elements=["Fe", "N", "H"],
        slab="Fe",
        gas_list=["H2", "N2"],
        gas_masses=[2.01568, 28.014],
        z_cutoff=11.96,
        temperature=723.0,
        pressure=40.0,
        pressure_unit="atm",
        steps=20,
    )
    d.update(override)
    return d


def test_valid_config_constructs():
    c = RunConfig(**_base())
    assert c.slab == "Fe"
    assert c.gas_masses_by_species() == {"H2": 2.01568, "N2": 28.014}


def test_example_config_loads():
    c = load_run_config(str(PROJECT_ROOT / "examples" / "config.yaml"))
    assert c.energy_backend in ("mace", "orca")


def test_slab_accepts_scalar_and_single_item_list():
    assert RunConfig(**_base(slab="Fe")).slab == "Fe"
    assert RunConfig(**_base(slab=["Fe"])).slab == "Fe"


def test_slab_multi_item_list_rejected():
    with pytest.raises(ValidationError):
        RunConfig(**_base(slab=["Fe", "N"]))


def test_max_steps_defaults_to_steps():
    assert RunConfig(**_base(steps=20)).max_steps == 20
    assert RunConfig(**_base(steps=20, max_steps=200)).max_steps == 200


def test_unknown_key_rejected():
    with pytest.raises(ValidationError):
        RunConfig(**_base(presure=40.0))  # typo of 'pressure'


def test_gas_masses_length_must_match_gas_list():
    with pytest.raises(ValidationError):
        RunConfig(**_base(gas_masses=[2.0]))  # 1 mass, 2 species


def test_slab_must_be_in_elements():
    with pytest.raises(ValidationError):
        RunConfig(**_base(slab="Cu"))


def test_requires_pressure_or_partial_pressures():
    with pytest.raises(ValidationError):
        RunConfig(**_base(pressure=None, partial_pressures=None))


def test_bad_pressure_unit_rejected():
    with pytest.raises(ValidationError):
        RunConfig(**_base(pressure_unit="psi"))


def test_mace_requires_model_dir():
    with pytest.raises(ValidationError):
        RunConfig(**_base(energy_backend="mace", model_dir=None))
    # with a model_dir, the mace backend validates
    assert RunConfig(**_base(energy_backend="mace", model_dir="m.model")).model_dir is not None


def test_partial_pressures_unknown_species_rejected():
    with pytest.raises(ValidationError):
        RunConfig(**_base(pressure=None, partial_pressures={"H2": 0.5, "CO2": 0.5}))


def test_partial_pressures_subset_of_gas_list_ok():
    c = RunConfig(**_base(pressure=None, partial_pressures={"H2": 0.5}))
    assert c.partial_pressures == {"H2": 0.5}
