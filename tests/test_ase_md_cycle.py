"""ASE-MD stage and `comet cycle` orchestration (no LAMMPS required).

Uses EMT (Cu/H) for MD forces and stub energies for the GCMC half, on a
synthetic Cu slab + H2 fixture.
"""

import random
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("numpy")
pytest.importorskip("ase")
pytest.importorskip("pydantic")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ase import Atoms
from ase.calculators.emt import EMT
from ase.io import write as ase_write
from ase.io.lammpsdata import write_lammps_data

from comet.config.schema import load_run_config
from comet.potentials.backends import EnergyBackend
from comet.workflows.md import _reflect_top, run_md
from comet.workflows.stages import build_initial_system

Z_CUTOFF = 10.0


def _stub_backend(with_calc: bool = True) -> EnergyBackend:
    return EnergyBackend(
        name="stub",
        energy=lambda atoms, label: -100.0,
        gas_energies=lambda templates: {name: -10.0 for name in templates},
        calculator_factory=(lambda: EMT()) if with_calc else None,
    )


def _write_fixture(tmp_path: Path, md_block: str = "") -> Path:
    system = Atoms(
        "Cu8",
        [[x, y, z] for z in (0.0, 1.8) for x in (2.0, 8.0) for y in (2.0, 8.0)],
    )
    for center in [(3.0, 3.0, 14.0), (8.0, 8.0, 18.0), (3.0, 8.0, 22.0)]:
        h2 = Atoms("HH", [[0, 0, -0.37], [0, 0, 0.37]])
        h2.positions += np.asarray(center)
        system += h2
    system.set_cell([12.0, 12.0, 28.0])
    system.set_pbc([True, True, False])

    restart = tmp_path / "seed.lammps"
    write_lammps_data(restart, system, specorder=["Cu", "H"], atom_style="atomic", masses=True)

    templates = tmp_path / "templates"
    templates.mkdir(exist_ok=True)
    ase_write(templates / "H2.xyz", Atoms("HH", [[0, 0, 0], [0, 0, 0.74]]))

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
bdir: {tmp_path / 'out'}
energy_backend: orca
restart_path: {restart}
elements: [Cu, H]
gas_list: [H2]
gas_template_dir: {templates}
z_cutoff: {Z_CUTOFF}
temperature: 300.0
partial_pressures:
  H2: 40.0
pressure_unit: bar
steps: 3
seed: 5
{md_block}
"""
    )
    return config_path


MD_BLOCK = """
md:
  n_cycles: 2
  md_steps: 5
  timestep_fs: 0.25
  freeze_bottom: 1.0
"""


def test_reflect_top_mirrors_position_and_momentum():
    atoms = Atoms("H2", [[1.0, 1.0, 9.7], [1.0, 1.0, 5.0]], cell=[10, 10, 10])
    atoms.set_momenta([[0.0, 0.0, 0.5], [0.0, 0.0, 0.1]])
    atoms.positions[0, 2] = 10.3   # crossed the lid
    _reflect_top(atoms, z_top=10.0)
    assert atoms.positions[0, 2] == pytest.approx(9.7)
    assert atoms.get_momenta()[0, 2] < 0          # reflected downward
    assert atoms.positions[1, 2] == pytest.approx(5.0)   # untouched


def test_run_md_moves_mobile_atoms_and_repartitions(tmp_path):
    random.seed(5)
    np.random.seed(5)
    config = load_run_config(str(_write_fixture(tmp_path, MD_BLOCK)))
    config.bdir.mkdir(parents=True, exist_ok=True)
    backend = _stub_backend()

    state = build_initial_system(config, backend)
    assert state is not None
    assert state.gas_counts == {"H2": 3}
    frozen_before = state.slab_ads.get_positions()[:4].copy()  # bottom layer z=0

    state = run_md(state, config, EMT(), backend, cycle_index=1)

    # Bottom layer pinned, bookkeeping consistent after re-partition.
    assert np.allclose(state.slab_ads.get_positions()[:4], frozen_before)
    assert state.gas_count == sum(state.gas_counts.values())
    tags = state.box_gas.get_tags()
    assert set(tags) == set(state.mol_species.keys())
    assert state.E_current == -100.0   # recomputed via the (stub) backend


def test_cycle_end_to_end_with_stub_backend(tmp_path, monkeypatch):
    import importlib

    run_mod = importlib.import_module("comet.workflows.run")
    monkeypatch.setattr(run_mod, "build_energy_backend", lambda config: _stub_backend())
    monkeypatch.chdir(tmp_path)

    rc = run_mod.cycle(str(_write_fixture(tmp_path, MD_BLOCK)))
    assert rc == 0
    for i in (1, 2):
        assert (tmp_path / "out" / f"cycle_{i}.lammpsdata").exists()


def test_cycle_requires_md_block(tmp_path, monkeypatch):
    import importlib

    run_mod = importlib.import_module("comet.workflows.run")
    monkeypatch.chdir(tmp_path)
    rc = run_mod.cycle(str(_write_fixture(tmp_path, md_block="")))
    assert rc == 1


def test_cycle_requires_md_capable_backend(tmp_path, monkeypatch):
    import importlib

    run_mod = importlib.import_module("comet.workflows.run")
    monkeypatch.setattr(
        run_mod, "build_energy_backend", lambda config: _stub_backend(with_calc=False)
    )
    monkeypatch.chdir(tmp_path)
    rc = run_mod.cycle(str(_write_fixture(tmp_path, MD_BLOCK)))
    assert rc == 1
