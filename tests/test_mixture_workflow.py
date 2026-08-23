"""End-to-end GCMC pipeline tests for gas mixtures (H2/N2 and H2/N2/NH3).

Drives config validation -> molecule recognition -> mixture MC loop ->
restart write on synthetic Fe-slab fixtures with a stub energy backend.
The ternary case exercises a polyatomic (NH3) in a mixture — the GCMC half
of what production MLP cycling runs (classical testbed MD cannot hold a
triatomic together, so ternary MD coverage lives with the MLP setup).
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

H2 = Atoms("HH", [[0, 0, 0], [0, 0, 0.741]])
N2 = Atoms("NN", [[0, 0, 0], [0, 0, 1.098]])
NH3 = Atoms(
    "NHHH",
    [
        [0.000000, 0.000000, 0.116489],
        [0.000000, 0.939731, -0.271808],
        [0.813831, -0.469865, -0.271808],
        [-0.813831, -0.469865, -0.271808],
    ],
)
MOLS = {"H2": H2, "N2": N2, "NH3": NH3}
Z_CUTOFF = 10.0


def _place(mol: Atoms, center) -> Atoms:
    out = mol.copy()
    out.positions += np.asarray(center) - out.get_center_of_mass()
    return out


def _stub_backend() -> EnergyBackend:
    return EnergyBackend(
        name="stub",
        energy=lambda atoms, label: -100.0,
        gas_energies=lambda templates: {name: -10.0 for name in templates},
    )


def _write_fixture(tmp_path: Path, gas_list, molecules) -> Path:
    """Fe slab + the given molecules; templates + config for `gas_list`."""
    system = Atoms(
        "Fe8",
        [[x, y, z] for z in (0.0, 1.8) for x in (2.0, 8.0) for y in (2.0, 8.0)],
    )
    for name, center in molecules:
        system += _place(MOLS[name], center)
    system.set_cell([14.0, 14.0, 30.0])
    system.set_pbc([True, True, False])

    elements = ["Fe"] + sorted({s for a in system if (s := a.symbol) != "Fe"})
    restart = tmp_path / "seed.lammps"
    write_lammps_data(restart, system, specorder=elements, atom_style="atomic", masses=True)

    templates = tmp_path / "templates"
    templates.mkdir()
    for name in gas_list:
        ase_write(templates / f"{name}.xyz", MOLS[name])

    pressures = "\n".join(f"  {name}: 40.0" for name in gas_list)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
bdir: {tmp_path / 'out'}
energy_backend: orca
restart_path: {restart}
elements: {elements}
gas_list: {gas_list}
gas_template_dir: {templates}
z_cutoff: {Z_CUTOFF}
temperature: 500.0
partial_pressures:
{pressures}
pressure_unit: bar
steps: 6
seed: 11
"""
    )
    return config_path


def _run_pipeline(tmp_path, gas_list, molecules, expected_counts):
    # Stage-level tests bypass run(), so the config `seed` is never applied —
    # seed the global RNGs directly for deterministic MC outcomes.
    import random
    random.seed(11)
    np.random.seed(11)
    config = load_run_config(str(_write_fixture(tmp_path, gas_list, molecules)))
    config.bdir.mkdir(parents=True, exist_ok=True)
    backend = _stub_backend()

    state = build_initial_system(config, backend)
    assert state is not None
    assert state.gas_counts == expected_counts

    state = run_mc_loop(state, config, backend)

    # Bookkeeping stays consistent across mixture moves.
    assert state.gas_count == sum(state.gas_counts.values())
    for name in gas_list:
        n_tagged = sum(1 for s in state.mol_species.values() if s == name)
        assert state.gas_counts[name] == n_tagged
    tags = state.box_gas.get_tags()
    sizes = {int(t): int((tags == t).sum()) for t in set(tags)}
    for mol_id, species in state.mol_species.items():
        assert sizes[mol_id] == len(MOLS[species])   # whole molecules only

    write_restart(state, config)
    out_file = config.bdir / "initial.lammpsdata"
    assert out_file.exists()

    # Every slab-side atom keeps its ORIGINAL id (not necessarily 1..n_slab:
    # an adsorbed molecule keeps its old gas-range id while freed low ids are
    # recycled to new gas molecules). Match by position — the slab is frozen
    # during GCMC, so positions are exact.
    old_atoms, _ = parse_lammps_data(config.restart_path)
    new_atoms, _ = parse_lammps_data(out_file)
    n_gas_atoms = sum(len(MOLS[s]) for s in state.mol_species.values())
    assert len(new_atoms) == len(state.slab_ads) + n_gas_atoms
    for pos in state.slab_ads.get_positions():
        matches = [
            aid for aid, (_, old_pos, _) in old_atoms.items()
            if np.allclose(old_pos, pos, atol=1e-6)
        ]
        assert len(matches) == 1
        assert np.allclose(new_atoms[matches[0]][1], pos, atol=1e-4)
    return state


def test_binary_h2_n2_mixture_pipeline(tmp_path):
    molecules = [
        ("H2", (3.0, 3.0, 15.0)), ("H2", (10.0, 3.0, 18.0)), ("H2", (3.0, 10.0, 21.0)),
        ("N2", (10.0, 10.0, 16.0)), ("N2", (6.5, 6.5, 24.0)),
        ("H2", (6.0, 11.0, 5.0)),   # adsorbed: COM below cutoff -> frozen slab side
    ]
    state = _run_pipeline(
        tmp_path, ["H2", "N2"], molecules, expected_counts={"H2": 3, "N2": 2}
    )
    assert len(state.slab_ads) == 8 + 2      # slab Fe + adsorbed H2


def test_ternary_h2_n2_nh3_mixture_pipeline(tmp_path):
    molecules = [
        ("H2", (3.0, 3.0, 15.0)), ("H2", (10.0, 3.0, 18.0)),
        ("N2", (10.0, 10.0, 16.0)), ("N2", (3.0, 10.0, 22.0)),
        ("NH3", (6.5, 6.5, 25.0)), ("NH3", (11.0, 6.0, 13.0)),
    ]
    state = _run_pipeline(
        tmp_path, ["H2", "N2", "NH3"], molecules,
        expected_counts={"H2": 2, "N2": 2, "NH3": 2},
    )
    # NH3 composition (N + 3H) never collides with N2 or H2 recognition:
    # every surviving molecule is one of the three species, no spectators.
    assert set(state.mol_species.values()) <= {"H2", "N2", "NH3"}
    assert None not in state.mol_species.values()
