#include "nbody_bruteforce.h"

#include <math.h>
#include <cuda_runtime.h>
#include <cstdio>

/*
Compute force (brute force method) of particle p2 on particle p1
Update particle p1
*/

__global__ void compute_brute_force(
	particle_t * array, int nbr_particles, double step) {
	
	int pos = threadIdx.x + blockIdx.x * blockDim.x;
	particle_t *p1 = &array[pos];

	double x_sep, y_sep, z_sep, dist_sq, grav_base;
	double F_x = 0.0, F_y = 0.0, F_z = 0.0;
	double a_x = 0.0, a_y = 0.0, a_z = 0.0;
	particle_t tmp;

	for (int i = 0 ; i  < nbr_particles ; i++){
		tmp = array[i];
		if (tmp.id!=p1->id){
			x_sep = tmp.x - p1->x;
			y_sep = tmp.y - p1->y;
			z_sep = tmp.z - p1->z;
			dist_sq = fmax((x_sep*x_sep) + (y_sep*y_sep) + (z_sep*z_sep), 0.01);
			grav_base = GRAV_CONSTANT*(p1->m)*(tmp.m)/dist_sq / sqrt(dist_sq);
			F_x += grav_base*x_sep;
			F_y += grav_base*y_sep;
			F_z += grav_base*z_sep;
		}
	}

// F = m a
// a = F/m
// V = a step
// pos = V * step
	a_x = F_x/p1->m;
	a_y = F_y/p1->m;
	a_z = F_z/p1->m;
	p1->vx += a_x * step;
	p1->vy += a_y * step;
	p1->vz += a_z * step;

}


__global__ void update_positions(particle_t * array, int nbr_particles, double step) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;

	particle_t *p1 = &array[i];
	p1->x += p1->vx * step;
	p1->y += p1->vy * step;
	p1->z += p1->vz * step;
}


/*
Implementation of a simple N-Body code in brute force.
The parallelization target is CUDA
Input format is 
*/
void nbodybruteforce (particle_t * host_array, int nbr_particles, int nbr_iterations) {

	particle_t *device_array;
	cudaMallocManaged(&device_array, nbr_particles * sizeof(particle_t));

	// Send data to device memory
	cudaMemcpy(device_array, host_array, nbr_particles * sizeof(particle_t), cudaMemcpyHostToDevice);

	int i,n;
	double step = 1.;

	dim3 block_size(256);
	dim3 grid_size((nbr_particles + block_size.x - 1) / block_size.x);

	for (n = 0 ; n  < nbr_iterations ; n++){
		printf("ITERATION %d \n",n);
		for (i = 0 ; i  < nbr_particles ; i++){
			//compute_brute_force(&array[i], array, nbr_particles,step);
			compute_brute_force<<<grid_size, block_size>>>(device_array, nbr_particles, step);
			cudaDeviceSynchronize();
		}
	  //update_positions(array, nbr_particles,step);
	  update_positions<<<grid_size, block_size>>>(device_array, nbr_particles, step);
	  cudaDeviceSynchronize();
	}

	cudaMemcpy(host_array, device_array, nbr_particles * sizeof(particle_t), cudaMemcpyDeviceToHost);
	cudaFree(device_array);



    auto error = cudaGetLastError();
    if(error != cudaSuccess) {
        printf("Error Launching Kernel: %s - %s\n", cudaGetErrorName(error), cudaGetErrorString(error));
        return; // or handle the error as appropriate
    }

	printf("[Particle %d]", host_array[7].id);
	printf(" Position ([%lf:%lf:%lf])", host_array[7].x, host_array[7].y, host_array[7].z);
	// printf(" Velocity ([%lf:%lf:%lf])", host_array[7].vx, host_array[7].vy, host_array[7].vz);
	printf(" Force ([%lf:%lf:%lf])", host_array[7].fx, host_array[7].fy, host_array[7].fz);
	printf(" M = %lf", host_array[7].m);
	// printf(" Volume = %lf", host_array[7].V);
	printf("\n");

}
