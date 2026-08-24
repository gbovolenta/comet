# COMET

Pressure **CO**ntroller using the **MET**ropolis algorithm (COMET).

COMET performs grand-canonical Monte Carlo (GCMC) pressure control of the gas
reservoir above a slab: molecules are inserted into and deleted from the gas
region until each species reaches the integer molecule count corresponding to
its target partial pressure. It is built for hybrid GCMC/MD workflows, where
pressure-control steps alternate with MD of the full slab+gas system.

COMET was used to generate the results of:

> G. M. Bovolenta and M. Parrinello, *The role of the N2/H2 reactant gas
> mixture on nitrogen activation in the Haber–Bosch process*,
> Nature Communications (2026).
> [doi:10.1038/s41467-026-76887-5](https://www.nature.com/articles/s41467-026-76887-5)

The MD inputs and computational details are available in the accompanying data
repository: [doi:10.5281/zenodo.21745812](https://zenodo.org/records/21745812).

Key properties:

- **No hardcoded chemistry.** Any molecular formula works as a gas species
  (H2, N2, NH3, CH3OH, ...); molecules are recognized once at load time by
  bond connectivity and composition, then tracked by persistent molecule ids.
  The slab may contain any mix of elements, including elements shared with
  the gas. Molecular masses are derived from the gas templates.
- **Mixtures with exact composition.** The composition is given as integer
  `ratios` and a total `pressure`; convergence targets are the integer counts
  nearest pV/kBT that preserve the ratio exactly. Species can be frozen by
  omission. States that do not fit the gas volume fail with the required
  volume/pressure factor instead of being silently approximated.
- **Molecule-aware partitioning.** Gas and slab are separated by molecule
  center of mass at `z_cutoff`, so molecules are never split at the boundary;
  species that adsorb during MD are frozen on the slab side on re-entry.
- **Energy backends**: MACE (`pip install comet[mace]`; the model file is
  supplied via `model_dir`, foundation models work directly) or ORCA
  (`orca` executable on `PATH`).

## Installation

```bash
pip install -e .          # ORCA backend only
pip install -e ".[mace]"  # + MACE backend (mace-torch)
```

## Quickstart

Two self-contained examples ship with the repository:

| Example | System |
|---|---|
| `examples/pure_H2/` | single-component H2 over Fe at 150 bar, 723 K |
| `examples/H2_N2_gas_mixture/` | binary H2:N2 = 3:1 mixture, 150 bar total, 723 K |

Each folder contains the LAMMPS seed structure, the gas templates, and a
`config.yaml`; supply your own MACE model (`model_dir`) and run from inside
the folder:

```bash
cd examples/pure_H2
comet run config.yaml
```

`comet --help` lists the commands; `python -m comet` is equivalent.

## Configuration

The configuration is a single YAML file (see the examples for annotated
versions). The essentials:

```yaml
energy_backend: mace            # or orca
model_dir: path/to/mace.model
restart_path: seed.lammpsdata   # LAMMPS data file (slab + gas)
elements: [Fe, N, H]            # LAMMPS atom-type order
gas_list: [H2, N2]
gas_template_dir: templates/    # one <species>.xyz per gas
z_cutoff: 15.147                # gas region: molecule COM above this [Å]
temperature: 723.0

pressure: 150.0                 # total pressure ...
ratios: {H2: 3, N2: 1}          # ... split by integer composition ratios
pressure_unit: bar

steps: 1000
biased_moves: true              # recommended for mixtures
seed: 42                        # reproducible runs (omit for independent runs)
```

Absolute per-species `partial_pressures` are accepted in place of
`pressure` + `ratios` (targets are then rounded per species, without a
composition constraint). The `seed` makes runs bit-reproducible on CPU.

## GCMC ↔ MD cycling

Two supported routes:

1. **External MD via files**: alternate `comet run` with your MD engine
   through the restart-file contract — comet writes `initial.lammpsdata`
   (the pressure-controlled configuration), MD writes back a data file that
   becomes the next `restart_path`. Slab atom ids survive the whole chain,
   and molecules are re-recognized after MD (adsorbed species move to the
   frozen slab side). In production, LAMMPS runs `pair_style mace` with the
   same model as the GCMC half so both halves share one PES.
2. **Built-in ASE MD** (`comet cycle config.yaml`, MACE backend only): with an
   `md:` block in the config, comet alternates GCMC with ASE dynamics
   (Bussi/CSVR thermostat, frozen bottom slab layers, reflective lid) using
   the same calculator instance for both halves — no external MD engine
   required. Per-cycle `cycle_<i>.lammpsdata` checkpoints are written.

## Output

Each run writes `gcmc_run.log` (banner, framed settings, sectioned narrative,
per-species status lines with ✔/━ convergence marks) and closes with a
PRESSURE CONTROL SUMMARY: the convergence verdict, the final vs requested
pressure per species in the configured unit, and a description of the output
files (`initial.lammpsdata` — the pressure-controlled restart for the next MD
stage; `mc_cycle.extxyz` — the gas-box trajectory).

## Tests

```bash
python -m pytest tests/
```

## Authors

Giulia M. Bovolenta
