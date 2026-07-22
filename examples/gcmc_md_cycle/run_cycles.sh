#!/bin/bash
# End-to-end GCMC<->MD cycle driver — scheduler-agnostic: run it inside an
# allocation (Slurm batch script here; any other scheduler's wrapper works the
# same on a production HPC).
#
# Usage:
#   export MODEL=/path/to/mace.model          # GCMC energy model (required)
#   [COMET_ENV=/path/to/env]                  # env with comet + lmp (default below)
#   run_cycles.sh <n_cycles> <config_template.yaml> <input.lammps> <seed.lammpsdata> [workdir]
#
# Example (Fe slab + H2, testbed potential):
#   MODEL=~/models/mace-small.model ./run_cycles.sh 3 config_h2.yaml input_h2.lammps seeds/fe_slab_h2.lammps
set -euo pipefail

EXDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
N_CYCLES=$1
CONFIG="$(readlink -f "$2")"
INPUT="$(readlink -f "$3")"
SEED="$(readlink -f "$4")"
WORK="${5:-$PWD/cycles}"

ENV="${COMET_ENV:-/opt/user-envs/$USER/comet-lmp}"
: "${MODEL:?export MODEL=/path/to/mace.model (GCMC energy backend)}"
MODEL="$(readlink -f "$MODEL")"

export LAMMPS_POTENTIALS="${LAMMPS_POTENTIALS:-$EXDIR/potentials}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

RESTART="$SEED"
for i in $(seq 1 "$N_CYCLES"); do
    CDIR="$WORK/cycle_$i"
    mkdir -p "$CDIR"
    cd "$CDIR"

    echo "=== cycle $i / $N_CYCLES : GCMC (comet) ==="
    sed -e "s|@RESTART@|$RESTART|" -e "s|@BDIR@|$CDIR/gcmc|" \
        -e "s|@MODEL@|$MODEL|"     -e "s|@EXDIR@|$EXDIR|" \
        "$CONFIG" > config.yaml
    # python -m: immune to the /usr/bin/env shebang conda-pack writes into bin/comet
    "$ENV/bin/python" -m comet run config.yaml

    echo "=== cycle $i / $N_CYCLES : MD (LAMMPS) ==="
    cp "$CDIR/gcmc/initial.lammpsdata" data_GCMC.lammps
    cp "$INPUT" input.lammps
    # PRODUCTION: lmp -k on g 1 -sf kk -in input.lammps  (Kokkos GPU LAMMPS-MACE)
    # conda-forge MPI builds can crash at startup if they see the scheduler's
    # PMI environment; scrub it so the single-node lmp runs as a plain singleton.
    (
        unset $(compgen -v SLURM_) $(compgen -v PMI_) $(compgen -v PMIX_) 2>/dev/null || true
        "$ENV/bin/lmp" -in input.lammps -log log.lammps
    )

    RESTART="$CDIR/final.lammpsdata"
    echo "=== cycle $i done: next restart -> $RESTART"
done

echo "All $N_CYCLES cycles completed."
