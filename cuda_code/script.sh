#!/bin/bash
#SBATCH --reservation=Course-math-454-final
#SBATCH --account=math-454
#SBATCH --time=01:00:00 # Time limit hrs:min:sec
#SBATCH -N 1            # Number of nodes
#SBATCH -n 1            # Number of cores
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

module purge
module load gcc cuda
module list

srun nbody-code $1
