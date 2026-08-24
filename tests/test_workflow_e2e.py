"""End-to-end test of the GCMC workflow stages with a stub energy backend.

Drives the real pipeline (config validation -> build_initial_system ->
run_mc_loop -> write_restart) on the bundled `examples/final.lammps` fixture,
using constant stub energies so no MACE/ORCA backend is required. This is the
coverage the orchestration layer previously lacked.
"""

import sys
from pathlib import Path

import pytest

pytest.importorskip("numpy")
pytest.importorskip("ase")
pytest.importorskip("pydantic")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from comet.config.schema import load_run_config
from comet.potentials.backends import EnergyBackend
from comet.workflows.stages import build_initial_system, run_mc_loop, write_restart

EXAMPLE_RESTART = PROJECT_ROOT / "examples" / "pure_H2" / "H2_P150bar_T723K.lammpsdata"


def _stub_backend() -> EnergyBackend:
    """A deterministic, dependency-free energy backend for the pipeline test."""
    return EnergyBackend(
        name="stub",
        energy=lambda atoms, label: -100.0,
        gas_energies=lambda templates: {name: -10.0 for name in templates},
    )


def _write_fixture(tmp_path: Path) -> Path:
    """Create a config + templates dir pointing at the bundled Fe/H2 restart."""
    templates = tmp_path / "templates"
    templates.mkdir()
    (templates / "H2.xyz").write_text("2\nH2\nH 0.0 0.0 0.0\nH 0.0 0.0 0.74\n")

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
bdir: {tmp_path / 'out'}
energy_backend: orca
restart_path: {EXAMPLE_RESTART}
elements: [Fe, H]
slab: [Fe]
gas_list: [H2]
gas_template_dir: {templates}
z_cutoff: 15.14698231002696
temperature: 823.0
pressure: 120.0
pressure_unit: atm
gas_masses: [2.01568]
steps: 2
"""
    )
    return config_path


def test_workflow_stages_end_to_end(tmp_path: Path):
    config_path = _write_fixture(tmp_path)
    config = load_run_config(str(config_path))
    config.bdir.mkdir(parents=True, exist_ok=True)
    backend = _stub_backend()

    # Build
    state = build_initial_system(config, backend)
    assert state is not None
    assert state.gas_counts["H2"] > 0
    assert state.gas_counts["H2"] % 1 == 0          # integral molecule count
    assert state.gas_count == sum(state.gas_counts.values())
    assert state.V > 0

    # Loop
    state = run_mc_loop(state, config, backend)
    assert state.gas_count == sum(state.gas_counts.values())

    # Write
    write_restart(state, config)
    out = config.bdir
    assert (out / "initial.lammpsdata").exists()
    assert (out / "mc_cycle.extxyz").exists()

    # The written restart has a well-formed Masses section (type-indexed floats).
    data = (out / "initial.lammpsdata").read_text()
    masses = data.split("Masses", 1)[1].split("Atoms", 1)[0]
    assert "1 55.845000" in masses   # Fe
    assert "2 1.008000" in masses    # H


def test_build_initial_system_missing_restart_raises(tmp_path: Path):
    """A missing restart file raises (region bounds are read first)."""
    templates = tmp_path / "templates"
    templates.mkdir()
    (templates / "H2.xyz").write_text("2\nH2\nH 0.0 0.0 0.0\nH 0.0 0.0 0.74\n")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
bdir: {tmp_path / 'out'}
energy_backend: orca
restart_path: {tmp_path / 'does_not_exist.lammps'}
elements: [Fe, H]
slab: [Fe]
gas_list: [H2]
gas_template_dir: {templates}
z_cutoff: 15.0
temperature: 823.0
pressure: 120.0
pressure_unit: atm
gas_masses: [2.01568]
steps: 1
"""
    )
    config = load_run_config(str(config_path))
    with pytest.raises(FileNotFoundError):
        # region bounds are read first and raise on the missing file
        build_initial_system(config, _stub_backend())


def test_run_orchestration_with_stub_backend(tmp_path, monkeypatch):
    """`run()` wires the pipeline together; inject a stub backend and exercise it."""
    import importlib

    # `comet.workflows.run` resolves to the re-exported function, so fetch the
    # actual module object to monkeypatch its build_energy_backend reference.
    run_mod = importlib.import_module("comet.workflows.run")

    config_path = _write_fixture(tmp_path)
    # Replace the real backend factory so run() drives the stub end-to-end.
    monkeypatch.setattr(run_mod, "build_energy_backend", lambda config: _stub_backend())
    # Keep the gcmc_run.log that run() creates inside the tmp dir.
    monkeypatch.chdir(tmp_path)

    rc = run_mod.run(str(config_path))
    assert rc == 0
    assert (tmp_path / "out" / "initial.lammpsdata").exists()
    assert (tmp_path / "out" / "mc_cycle.extxyz").exists()

    # The final block states the convergence verdict explicitly: 2 steps
    # cannot bring 16 H2 to the target of 12, so the warning form appears.
    log_text = (tmp_path / "gcmc_run.log").read_text()
    assert "Species not at their target counts: H2" in log_text


def test_build_energy_backend_orca_constructs(tmp_path):
    """build_energy_backend returns a usable ORCA backend without running ORCA."""
    from comet.potentials.backends import build_energy_backend

    config = load_run_config(str(_write_fixture(tmp_path)))  # energy_backend: orca
    backend = build_energy_backend(config)
    assert backend.name == "orca"
    assert callable(backend.energy)
    assert callable(backend.gas_energies)


def test_run_is_reproducible_with_seed(tmp_path, monkeypatch):
    """Two runs with the same seed produce an identical restart file."""
    import importlib

    run_mod = importlib.import_module("comet.workflows.run")
    monkeypatch.setattr(run_mod, "build_energy_backend", lambda config: _stub_backend())

    def _run_once(subdir: str) -> str:
        d = tmp_path / subdir
        templates = d / "templates"
        templates.mkdir(parents=True)
        (templates / "H2.xyz").write_text("2\nH2\nH 0.0 0.0 0.0\nH 0.0 0.0 0.74\n")
        cfg = d / "config.yaml"
        cfg.write_text(
            f"""
bdir: {d / 'out'}
energy_backend: orca
restart_path: {EXAMPLE_RESTART}
elements: [Fe, H]
slab: [Fe]
gas_list: [H2]
gas_template_dir: {templates}
z_cutoff: 15.14698231002696
temperature: 823.0
pressure: 120.0
pressure_unit: atm
gas_masses: [2.01568]
steps: 5
seed: 123
"""
        )
        monkeypatch.chdir(d)
        assert run_mod.run(str(cfg)) == 0
        return (d / "out" / "initial.lammpsdata").read_text()

    assert _run_once("run_a") == _run_once("run_b")
