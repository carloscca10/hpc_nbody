#!/bin/bash

#SBATCH --job-name=nbody_simulation
#SBATCH -N 1            # Number of nodes
#SBATCH -n 1            # Number of cores
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu_free
#SBATCH --account=math-454
#SBATCH --reservation=Course-math-454-final
#SBATCH --time=01:00:00 # Time limit hrs:min:sec

module purge
module load gcc cuda
module list

srun nvprof $1