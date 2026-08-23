"""End-to-end GCMC workflow with a polyatomic gas (CH3OH) on a Cu slab.

Exercises the full pipeline — config validation (no gas_masses, no slab key),
molecule recognition on a synthetic restart, tag-based insertion/deletion,
and slab-ID-preserving restart output — with a stub energy backend.
"""

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
from ase.io import write as ase_write
from ase.io.lammpsdata import write_lammps_data

from comet.config.schema import load_run_config
from comet.io.lammps_datafile import parse_lammps_data
from comet.potentials.backends import EnergyBackend
from comet.workflows.stages import build_initial_system, run_mc_loop, write_restart

MEOH = Atoms(
    "COHHHH",
    [
        [-0.046423, 0.662445, 0.000000],
        [-0.046423, -0.758170, 0.000000],
        [0.986955, 1.021560, 0.000000],
        [-0.541793, 1.062351, 0.889607],
        [-0.541793, 1.062351, -0.889607],
        [0.864827, -1.058866, 0.000000],
    ],
)

ELEMENTS = ["Cu", "C", "O", "H"]
Z_CUTOFF = 10.0


def _place(mol: Atoms, center) -> Atoms:
    out = mol.copy()
    out.positions += np.asarray(center) - out.get_center_of_mass()
    return out


def _write_fixture(tmp_path: Path) -> Path:
    """Synthetic restart: 8-atom Cu slab, one adsorbed + two gas CH3OH."""
    system = Atoms(
        "Cu8",
        [[x, y, z] for z in (0.0, 1.8) for x in (2.0, 8.0) for y in (2.0, 8.0)],
    )
    system += _place(MEOH, (5.0, 5.0, 4.0))    # adsorbed, COM below cutoff
    system += _place(MEOH, (3.0, 3.0, 15.0))   # gas
    system += _place(MEOH, (9.0, 9.0, 20.0))   # gas
    system.set_cell([12.0, 12.0, 30.0])
    system.set_pbc([True, True, False])

    restart = tmp_path / "cu_meoh.lammps"
    write_lammps_data(restart, system, specorder=ELEMENTS, atom_style="atomic", masses=True)

    templates = tmp_path / "templates"
    templates.mkdir()
    ase_write(templates / "CH3OH.xyz", MEOH)

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
bdir: {tmp_path / 'out'}
energy_backend: orca
restart_path: {restart}
elements: [Cu, C, O, H]
gas_list: [CH3OH]
gas_template_dir: {templates}
z_cutoff: {Z_CUTOFF}
temperature: 400.0
partial_pressures:
  CH3OH: 40.0
pressure_unit: bar
steps: 4
seed: 7
"""
    )
    return config_path


def _stub_backend() -> EnergyBackend:
    return EnergyBackend(
        name="stub",
        energy=lambda atoms, label: -100.0,
        gas_energies=lambda templates: {name: -10.0 for name in templates},
    )


def test_ch3oh_on_cu_full_pipeline(tmp_path: Path):
    config = load_run_config(str(_write_fixture(tmp_path)))
    assert config.gas_masses is None and config.slab is None
    config.bdir.mkdir(parents=True, exist_ok=True)
    backend = _stub_backend()

    # Build: two gas CH3OH recognized; the adsorbed one is frozen on the slab side.
    state = build_initial_system(config, backend)
    assert state is not None
    assert state.gas_counts == {"CH3OH": 2}
    assert len(state.box_gas) == 12
    assert len(state.slab_ads) == 8 + 6
    # Mass derived from the template, not the config.
    assert state.gas_dict["CH3OH"] == pytest.approx(32.042, abs=0.01)
    assert sorted(state.mol_species.values()) == ["CH3OH", "CH3OH"]

    # Loop: bookkeeping stays consistent whatever gets accepted.
    state = run_mc_loop(state, config, backend)
    assert state.gas_count == sum(state.gas_counts.values())
    assert state.gas_count == sum(1 for s in state.mol_species.values() if s == "CH3OH")
    tags = state.box_gas.get_tags()
    assert set(tags) == set(state.mol_species.keys())
    for mol_id in state.mol_species:
        assert int((tags == mol_id).sum()) == 6   # every molecule is a whole CH3OH

    # Write: restart exists and every slab-side atom kept its original ID.
    write_restart(state, config)
    out_file = config.bdir / "initial.lammpsdata"
    assert out_file.exists()

    old_atoms, _ = parse_lammps_data(config.restart_path)
    new_atoms, new_vels = parse_lammps_data(out_file)
    assert len(new_atoms) == 14 + 6 * state.gas_count
    assert len(new_vels) == len(new_atoms)
    for aid in range(1, 15):                      # slab block was written first
        old_type, old_pos, _ = old_atoms[aid]
        new_type, new_pos, _ = new_atoms[aid]
        assert new_type == old_type
        assert np.allclose(new_pos, old_pos, atol=1e-4)
