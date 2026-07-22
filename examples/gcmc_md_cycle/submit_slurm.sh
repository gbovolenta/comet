#!/bin/bash -l
#SBATCH --job-name=gcmc-md-cycle
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%j.out

# Slurm wrapper for the cycle driver (CPU testbed). Adjust the three
# variables, then sbatch from the directory where you want ./cycles/ created.
# On a production GPU HPC, replace this wrapper with your scheduler's and use
# the PRODUCTION-marked lines inside run_cycles.sh / the input files.

EXDIR="$(dirname "$(readlink -f "$0")")"

export MODEL=/path/to/mace.model             # <-- GCMC energy model (.model)
CONFIG=$EXDIR/config_h2.yaml                 # or config_h2_n2.yaml
INPUT=$EXDIR/input_h2.lammps                 # or input_h2_n2.lammps
SEED=$EXDIR/seeds/fe_slab_h2.lammps          # or seeds/fe_slab_h2_n2.lammps

bash "$EXDIR/run_cycles.sh" 3 "$CONFIG" "$INPUT" "$SEED"
