// CUDA runtime
#include <cuda_runtime.h>
#include <iostream>

__global__ void hello_world() {
  // TODO: experiment with printf here and the thread and grid idx
}

int main() {
  printf("printf() is called. Output:\n");

  // Kernel configuration, where a two-dimensional grid and
  // three-dimensional blocks are configured.
  dim3 dimGrid(2, 2);
  dim3 dimBlock(2, 2, 2);

  hello_world<<<dimGrid, dimBlock>>>();
  // TODO: what do you need to do in order to ensure printing?
  return 0;
}
