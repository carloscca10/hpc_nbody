#!/bin/bash
#SBATCH --reservation=Course-math-454-final
#SBATCH --account=math-454
#SBATCH --time=00:10:00
#SBATCH -N 1  # Request 1 node
#SBATCH -n 1  # Request 2 tasks (processors)
module purge
module load intel intel-oneapi-mpi

ulimit -l 127590
srun nbody-code $1