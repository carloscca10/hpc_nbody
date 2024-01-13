#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=01:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --account=math-454
#SBATCH --reservation=Course-math-454-final

module purge
module load gcc cuda
module list

#nvprof is used to profile our code
srun nvprof nbody-code $1
