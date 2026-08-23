# End-to-end GCMC ↔ MD cycling

This example alternates COMET's GCMC pressure control with LAMMPS MD, the way
the method is used in production: GCMC equilibrates the gas reservoir above the
slab against target partial pressures, MD relaxes/thermalizes the whole system,
and the cycle repeats.

```
seed.lammpsdata ──> comet run ──> initial.lammpsdata ──> lmp (MD) ──> final.lammpsdata ──┐
        ^                                                                                │
        └────────────────────────────── next cycle restart <─────────────────────────────┘
```

Per cycle, `run_cycles.sh` fills the config template, runs the GCMC half, hands
`initial.lammpsdata` to LAMMPS as `data_GCMC.lammps`, and feeds LAMMPS'
`final.lammpsdata` back as the next restart. Slab atom IDs survive the whole
chain (frozen atoms keep bit-identical coordinates), and molecules are
re-recognized after MD — species that adsorbed (molecule COM below `z_cutoff`)
move to the frozen slab side, desorbed ones re-enter the GCMC gas box, and
anything unrecognizable is kept as a frozen spectator with a warning.

## Quickstart (CPU testbed)

Needs an environment with `comet`, `lammps`, and `mace-torch`, plus any MACE
`.model` file (a foundation model works):

```bash
export MODEL=/path/to/mace.model
./run_cycles.sh 3 config_h2.yaml input_h2.lammps seeds/fe_slab_h2.lammps
```

or adjust the variables in `submit_slurm.sh` and `sbatch` it.

## The two variants

| | config | MD input | seed |
|---|---|---|---|
| Pure H2 on Fe | `config_h2.yaml` | `input_h2.lammps` | `seeds/fe_slab_h2.lammps` (400 Fe + 16 H2) |
| Binary H2/N2 | `config_h2_n2.yaml` | `input_h2_n2.lammps` | `seeds/fe_slab_h2_n2.lammps` (400 Fe + 16 H2 + 6 N2) |

The binary variant drives each species to its own partial pressure, specified
as a total `pressure` split by `mole_fractions` (absolute `partial_pressures`
are also accepted). Atom-type order is defined by `elements` in the config
and must match the seed file and the `pair_coeff` type indices in the MD input.

**Ternary+ mixtures (e.g. H2/N2/NH3):** the GCMC half handles any number of
species and any molecular formula (see `tests/test_mixture_workflow.py` for a
full H2/N2/NH3 pipeline). The *testbed MD potential*, however, is pair-style
only — a polyatomic like NH3 needs intramolecular angle terms it cannot
provide. Cycling polyatomics end-to-end therefore requires an MLP for the MD
half (production setup below); with `pair_style mace` the ternary case needs
nothing beyond a third template and a `partial_pressures` entry.

## No LAMMPS? `comet cycle` (built-in ASE MD)

If no MD engine is available, comet can run the whole alternation itself:
add an `md:` block to the config (see `examples/config.yaml`) and run

```bash
comet cycle config.yaml
```

The MD half then uses ASE dynamics with the SAME MACE calculator as the GCMC
half (Bussi/CSVR thermostat, frozen bottom slab, reflective lid — the same
recipe as the LAMMPS inputs here), which also makes the two halves
PES-consistent by construction. Per-cycle `cycle_<i>.lammpsdata` checkpoints
are written. The file-based LAMMPS route in this directory remains the choice
when you want LAMMPS' MD feature set (Kokkos GPU throughput, PLUMED, etc.).

## Testbed vs production

The example as committed is a **CPU testbed**: it validates the cycle
mechanics (file contract, ID continuity, re-partitioning) with a classical
stand-in potential — EAM Fe plus a real Morse bond per diatomic and weak LJ
cross terms. It is deliberately NOT matched-PES: GCMC uses MACE, MD uses the
stand-in.

For a production GPU run, swap the three marked blocks:

1. **MD potential** (`input_*.lammps`): the `PRODUCTION` block —
   `pair_style mace no_domain_decomposition` with your fine-tuned
   `.model-lammps.pt` (compile from the same model with
   `mace create_lammps_model`), launched `lmp -k on g 1 -sf kk` (Kokkos GPU
   build of LAMMPS-MACE).
2. **GCMC model** (`config_*.yaml`): the same fine-tuned model's `.model`
   file — using one model for both halves is what makes the sampled ensemble
   consistent.
3. **Scheduler wrapper**: replace `submit_slurm.sh` with your HPC's;
   `run_cycles.sh` itself is scheduler-agnostic.

Also revert the testbed shortcuts marked in the inputs: `NSTEPS` (2000 →
200000) and `timestep` (0.25 fs → 0.5 fs; the small value only exists because
the Morse H2 vibrates with a ~8 fs period).

## Known testbed artifacts (not bugs)

- The Morse bond acts between ALL atom pairs of its element, so colliding gas
  molecules can stick (H2 + H2 → H4). COMET quarantines such clusters as
  frozen spectators and keeps going; production MLPs don't have this artifact.
- Classical stand-ins cannot describe H dissolved in the metal. Seeds here are
  gas-over-clean-slab; a restart from a production MLP run (which contains
  dissolved H) needs the MLP potential.

## Deployment traps (each cost a debugging session)

- **conda-pack/conda-sync**: refuses editable installs (`pip install -e`) and
  rewrites console-script shebangs to `/usr/bin/env` — install comet
  normally in the synced env and invoke `python -m comet` (the driver does).
- **conda-forge OpenMPI + Slurm**: the openmpi LAMMPS build can segfault at
  startup inside Slurm jobs (PMIx psec/munge mismatch). Use the mpich build
  variant (`lammps=*=cpu_*mpich*`); the driver additionally scrubs
  `SLURM_*`/`PMI*` around the `lmp` call.
- **`LAMMPS_POTENTIALS`**: conda-forge LAMMPS ships no potentials folder; the
  driver points it at `potentials/` here (Mendelev Fe EAM included).
- **The synced env snapshots comet at sync time**: since comet is a regular
  (non-editable) install in the conda-sync'd env, pulling new comet source
  requires `pip install <repo>` into the env AND a re-run of `conda-sync`,
  or compute-node jobs run the old package (symptom: new config keys rejected
  with "Extra inputs are not permitted").
