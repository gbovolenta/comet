# COMET

Pressure COntroller using METropolis algorithm (COMET). 

## Quickstart

Install the package in editable mode:

```bash
pip install -e .
```

Run the example workflow configuration from the repository root:

```bash
comet run examples/config.yaml
```

You will need to provide your own:

- An energy backend: a **MACE model file** (`mace`, the default) or the
  **`orca` executable on your `PATH`** plus `orca_*` settings (`energy_backend: orca`)
- A **gas-template directory** with one `<species>.xyz` file per gas (e.g. `H2.xyz`,
  `CH3OH.xyz`) — any molecular formula works; species are recognized by composition
- A **LAMMPS restart/data file** — a small example (`examples/final.lammps`, an Fe slab
  with H₂ gas) is included with the repository, so the quickstart runs once you supply a backend and templates

Neither the surface nor the gas is hardcoded: the slab may contain any mix of
elements (including elements shared with the gas), molecular masses are derived
from the templates, and molecules are assigned to the gas region by center of
mass, so they are never split at the `z_cutoff` boundary.

Show the available commands:

```bash
comet --help
```

The `comet` module can also be invoked directly:

```bash
python -m comet --help
```

## Configuration

An example configuration is provided at `examples/config.yaml`, loaded with `yaml.safe_load`.
The energy backend is selected with `energy_backend` (`mace` or `orca`). Pressure control is
set with a total `pressure` split by per-species `mole_fractions` (omit a species to freeze it);
absolute per-species `partial_pressures` are also accepted. The GCMC insertion region is derived from the restart
file and `z_cutoff`.
