# COMET

COMET (Pressure **CO**ntroller using the **MET**ropolis algorithm) performs grand-canonical Monte Carlo (GCMC) pressure control for gas–surface simulations. It inserts and deletes molecules in a designated gas-phase region using Metropolis moves at prescribed chemical potentials, enabling control of the partial pressure of one or more gas species.

COMET is intended as the GCMC component of hybrid GCMC/MD workflows: molecular dynamics is performed by an external engine, while COMET periodically updates the gas reservoir. By restricting GCMC moves to a control region separated from the reactive surface by a buffer, pressure can be maintained without directly perturbing interfacial chemistry.

COMET was used to generate the results of:

> G. M. Bovolenta and M. Parrinello, *The role of the N2/H2 reactant gas
> mixture on nitrogen activation in the Haber–Bosch process*,
> Nature Communications (2026).
> [doi:10.1038/s41467-026-76887-5](https://www.nature.com/articles/s41467-026-76887-5)

The MD inputs and computational details are available in the accompanying data
repository: [doi:10.5281/zenodo.21745812](https://zenodo.org/records/21745812).

General features:

- **Gas handling.** Any molecular formula can be used as a gas species (H₂, N₂, NH₃, CH₃OH, ...). Molecules are identified once at load time from their bond connectivity and elemental composition, then tracked using persistent molecule IDs. 
- **Gas mixtures composition.** The mixture composition is specified by integer ratios, for example `ratios: {H2: 3, N2: 1}`, together with a total pressure. COMET maintains the specified ratio while choosing the molecule counts that give the closest possible ideal-gas target pressure.
- **Molecule-aware partitioning.** The `z_cutoff` separates the buffer region from the pressure-control region used for GCMC insertion and deletion moves. Molecules are assigned by center-of-mass position, so they are never split across the boundary.
- **Energy evaluation.** COMET supports MACE (https://mace-docs.readthedocs.io/en/latest/) and ORCA (Neese, F. 2025, WIREs Computational Molecular Science,
15, e70019, doi: 10.1002/wcms.70019) as energy-evaluation engines for the Metropolis acceptance criterion. MACE models are supplied via `model_dir`, with foundation models supported directly; ORCA requires the `orca` executable to be available on `PATH`.


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


## Output

Each run produces:

- `gcmc_run.log`: a run log reporting the per-species convergence status and a final PRESSURE CONTROL SUMMARY, including the final and requested pressure for each species.
- `initial.lammpsdata`: the pressure-controlled structure written in LAMMPS data format. In the hybrid GCMC/MD workflow presented in the paper, molecular dynamics is performed with the external LAMMPS (https://docs.lammps.org) engine, and this file serves as the input structure for the subsequent MD stage.
- `mc_cycle.extxyz`: the trajectory of the gas-box configuration over the GCMC cycle.

## Tests

```bash
python -m pytest tests/
```

## Authors

Giulia M. Bovolenta
