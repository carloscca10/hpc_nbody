#ifndef NBODYBRUTEFORCE_H_
#define NBODYBRUTEFORCE_H_

#include <stdio.h> 
#include "parameters.h"

void nbodybruteforce (particle_t * array, int nbr_particles, int nbr_iterations) ;
//void compute_brute_force(particle_t * p1, particle_t * array, int nbr_particles, double step) ;
__global__ void compute_brute_force(particle_t * array, int nbr_particles, double step) ;
__global__ void update_positions_kernel(particle_t * array, int nbr_particles, double step) ;
//double max(double x, double y);
//double min(double x, double y);


#endif /*NBODYBRUTEFORCE_H_*/
