#!/bin/bash
#SBATCH --reservation=Course-math-454-final
#SBATCH --account=math-454
#SBATCH --time=00:10:00
#SBATCH -N 1  # Request 1 node
#SBATCH --array=1,2,4,8,16,32  # Set processor counts
module purge
module load intel intel-oneapi-mpi

ulimit -l 127590

# Define the base directory for the subset files
BASE_DIR="../../examples/strong_scaling"

# Loop over the subset files from 10000 to 140000
for i in {10000..140000..10000}
do
   # Loop over the number of processors
   for procs in 1 2 4 8 16 32
   do
      # Update the number of tasks
      sbatch -n $procs --wrap="srun nbody-code ${BASE_DIR}/subset_${i}.txt"
   done
done
