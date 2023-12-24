#ifndef NBODYBARNESHUT_H_
#define NBODYBARNESHUT_H_

#include <stdio.h>
#include <stdlib.h>

#include "parameters.h"
//#include "nbody_bruteforce.h"
#include "reader.h"
#include "math.h"

void nbodybarneshut (particle_t * array, int nbr_particles, int nbr_iterations, int prank, int psize) ;
void construct_bh_tree(particle_t * array, node * root, int nbr_particles);
void compute_bh_force(node * n, int prank, int psize) ;
void compute_force(particle_t *p, double xpos, double ypos,  double zpos, double mass) ;
void compute_force_particle(node * n, particle_t * p);
void compute_force_in_node(node *n,node *root, int prank, int psize);
void move_all_particles(node * root, node * n, double step) ;
void move_particle(node * root, node * n, particle_t * p, double step) ;
bool is_particle_out_of_scope(particle_t * p, node * root);
void clean_tree(node * root) ;


void insert_particle(particle_t * p, node * n);
void init_tree(particle_t * part, node * root);
int get_octrant(particle_t * p, node * n);
void create_children(node * parent);
void init_node(node * n, node * parent, double minx, double maxx, double miny, double maxy, double minz, double maxz );

// Utilities
void print_tree(node * n);
void print_node(node * n);
void print_particle(particle_t * p, int prank, int psize);
void print_particle_it(particle_t * p, int prank, int psize, int n);

// MPI
// void gather_force_vector(particle_t *array, int nbr_particles, double *forces);
// void broadcast_force_vector(particle_t *array, int nbr_particles, double *forces);
// void gather_force_vector(node * n, double *forces);
// void broadcast_force_vector(node *n, double *forces);
void compare_arrays(particle_t *array, int nbr_particles, int prank, int psize);
void compare_arrays_except_forces(particle_t * array, int nbr_particles, int prank, int psize);

void gather_force_vector_array(particle_t * array, double *forces, int nbr_particles, int prank, int psize);
void broadcast_force_vector_array(particle_t * array, double *forces, int nbr_particles);
void check_no_f_if_not_rank(particle_t * array, int nbr_particles, int prank, int psize);
void check_no_f_if_not_rank_forces(double *forces, int nbr_particles, int prank, int psize);

#endif /*NBODYBARNESHUT_H_*/
