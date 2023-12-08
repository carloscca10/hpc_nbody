#ifndef NBODYBRUTEFORCE_H_
#define NBODYBRUTEFORCE_H_

#include <stdio.h> 
#include "parameters.h"

#define max(a,b) ((a) > (b) ? (a) : (b))
#define min(a,b) ((a) < (b) ? (a) : (b))


void nbodybruteforce (particle_t * array, int nbr_particles, int nbr_iterations) ;
void compute_brute_force(particle_t * p1, particle_t * array, int nbr_particles, double step) ;
void update_positions(particle_t * array, int nbr_particles, double step) ;


#endif /*NBODYBRUTEFORCE_H_*/
