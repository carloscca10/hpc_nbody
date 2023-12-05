#include "nbody_bruteforce.h"

#include <math.h>

/*
Min and max functions
*/

// condition ? expression_if_true : expression_if_false;

double max(double x, double y) 
{
	return ((x) > (y) ? (x) : (y));
} 

// double min(double x, double y) 
// {
// 	return ((x) < (y) ? (x) : (y));
// } 


/*
Implementation of a simple N-Body code in brute force.
The parallelization target is CUDA
Input format is 
*/
void nbodybruteforce (particle_t * array, int nbr_particles, int nbr_iterations) {

	int i,n;
	double step = 1.;
	for (n = 0 ; n  < nbr_iterations ; n++){
		printf("ITERATION %d \n",n);
		for (i = 0 ; i  < nbr_particles ; i++){
			compute_brute_force(&array[i], array, nbr_particles,step);
		}
	  update_positions(array, nbr_particles,step);
	}
	// print final values of element number 8 of array (array[7])
	printf("final values of array[7]:\n");
	printf("x: %lf\n", array[7].x);
	printf("y: %lf\n", array[7].y);
	printf("z: %lf\n", array[7].z);
	printf("vx: %lf\n", array[7].vx);
	printf("vy: %lf\n", array[7].vy);
	printf("vz: %lf\n", array[7].vz);
	printf("fx: %lf\n", array[7].fx);
	printf("fy: %lf\n", array[7].fy);
	printf("fz: %lf\n", array[7].fz);
	printf("m: %lf\n", array[7].m);
	printf("id: %d\n", array[7].id);
	printf("V: %lf\n", array[7].V);
}

/*
Compute force (brute force method) of particle p2 on particle p1
Update particle p1
*/

void compute_brute_force(particle_t * p1, particle_t * array, int nbr_particles, double step) {
	double x_sep, y_sep, z_sep, dist_sq, grav_base;
	double F_x=0.;
	double F_y=0.;
	double F_z=0.;
	double a_x=0.;
	double a_y=0.;
	double a_z=0.;
	particle_t tmp;

	for (int i = 0 ; i  < nbr_particles ; i++){
		tmp = array[i];
		if (tmp.id!=p1->id){
			x_sep = tmp.x - p1->x;
			y_sep = tmp.y - p1->y;
			z_sep = tmp.z - p1->z;
			dist_sq = max((x_sep*x_sep) + (y_sep*y_sep) + (z_sep*z_sep), 0.01);
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

void update_positions(particle_t * array, int nbr_particles, double step) {
	for (int i = 0 ; i  < nbr_particles ; i++){
		particle_t *p1 = &array[i];
      p1->x += p1->vx * step;
	  p1->y += p1->vy * step;
	  p1->z += p1->vz * step;
	}
}




