#!/bin/bash

#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --account=math-454
#SBATCH --reservation=Course-math-454-final

module purge
module load intel intel-oneapi-mpi

srun ./nbody-code ../../examples/cluster_LI.txt