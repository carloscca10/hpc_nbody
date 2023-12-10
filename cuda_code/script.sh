#!/bin/bash

#SBATCH --job-name=nbody_simulation
#SBATCH -N 1            # Number of nodes
#SBATCH -n 1            # Number of cores
#SBATCH --account=math-454
#SBATCH --reservation=Course-math-454-final
#SBATCH --time=01:00:00 # Time limit hrs:min:sec
#SBATCH --output=nbody_%j.out # Standard output and error log
#SBATCH --error=nbody_%j.err

module purge
module load gcc cuda
module list

srun nvprof $1
