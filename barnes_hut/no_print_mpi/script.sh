#!/bin/bash
#SBATCH --reservation=Course-math-454-final
#SBATCH --account=math-454
#SBATCH --time=02:00:00 # Adjust the total time as needed
#SBATCH -N 1  # Request 1 node
#SBATCH -n 32  # Maximum number of tasks required
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
      # Run the nbody-code with the specified number of processors
      srun -n $procs nbody-code "${BASE_DIR}/subset_${i}.txt"
   done
done
