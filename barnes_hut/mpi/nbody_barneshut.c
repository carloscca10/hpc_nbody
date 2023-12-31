#include "nbody_barneshut.h"
#include <mpi.h>



#include <stdbool.h>


/*
Implementation of a barnes-hut algorithm for the N-Body problem.
*/
void nbodybarneshut (particle_t * array, int nbr_particles, int nbr_iterations, int prank, int psize) 
{
	int n;
	double step = TIMESTEP;
	node * root1;
	node * root2;
	node * root;
	particle_t tmp;
	double forces[3 * nbr_particles];

	//printf("Creation of the tree ...");
	root1 = malloc(sizeof(node));	
	root2 = malloc(sizeof(node));	
	tmp = getMinMax(array, nbr_particles);
	init_tree(&tmp, root1);	
	init_tree(&tmp, root2);	
	//printf("OK \n");

	//printf("Construction of the tree from file ...");
	construct_bh_tree(array, root1, nbr_particles);
	//printf("OK \n");
	// printf("Init forces ...");
	print_particle(&array[7], prank, psize);
	// compute_force_in_node(root1, root1);
	// printf(" OK \n");
	//printf("Compute forces ...\n");
	
	for (n = 0 ; n  < nbr_iterations ; n++){

		compute_force_in_node(root1, root1, prank, psize);
		compute_bh_force(root1, prank, psize);

		//gather_force_vector(root1, forces);
		gather_force_vector_array(array, forces, nbr_particles, prank, psize);
		MPI_Allreduce(MPI_IN_PLACE, &forces, nbr_particles*3, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
		//broadcast_force_vector(root1, forces);
		broadcast_force_vector_array(array, forces, nbr_particles);

		move_all_particles(root2, root1, step);

		root = root1;
		root1 = root2;
		root2 = root;
		clean_tree(root2);
		if(prank==0) {
			printf("%d: ITERATION %d \n",prank, n);
			print_particle_it(&array[7], prank, psize, n);
			printf("%d particles still in space \n",root1->sub_nbr_particles);
		}
	}

	printf("It remains %d particles in space \n",root1->sub_nbr_particles);	

	clean_tree(root1);
	clean_tree(root2);

	free(root1);
	free(root2);

	
	// print final values of element number 8 of array (array[7])
	//if(prank == 0) {
	print_particle(&array[7], prank, psize);
	//}
}

/*

1. If the current node is an external node (and it is not body b), calculate the force exerced by the current node on b, and add this amount to b’s net force.
    
2. Otherwise, calculate the ratio s/d. If s/d < θ, treat this internal node as a single body, and calculate the force it exerts on body b, and add this amount to b’s net force.

3. Otherwise, run the procedure recursively on each of the current node’s children.

Once the computation of the force applied to the particles is complete, the new position of the particles is computed, and a new tree corresponding to the new position is created. 

*/

/*
Move all the particles from node n to new_root
*/

void move_all_particles(node * new_root, node * n, double step) {
	int i;
	if(n->children != NULL){
		for (i = 0; i < 8; i++){
			move_all_particles(new_root, &n->children[i], step);
		}
	}else{
		particle_t * p = n->particle;
		move_particle(new_root, n, p,step);
	}
}

/*
Compute new position/velocity of the particle
*/

void move_particle(node * root, node * n, particle_t * p, double step) {
	double ax,ay,az;

	if ((p==NULL)||(n==NULL)) return;

	ax = p->fx/p->m;
	ay = p->fy/p->m;
	az = p->fz/p->m;
	p->vx += ax*step;
	p->vy += ay*step;
	p->vz += az*step;
	p->x += p->vx * step;
	p->y += p->vy * step;
	p->z += p->vz * step;

	if (! is_particle_out_of_scope(p,root)) {
		insert_particle(p,root);
	}else{
		//printf("Particle %d | %d is out of scope. (%f, %f, %f) \n",p->id, p->mpi_id, p->x, p->y, p->z);
		n->particle = NULL;
	}
}

/*
Check if a particle is out of scope (lost body in space)
*/

bool is_particle_out_of_scope(particle_t * p, node * root){
	bool ret = false;
	if ((p->x < root->minx)||(p->y < root->miny)||(p->z < root->minz)) ret = true;
	if ((p->x > root->maxx)||(p->y > root->maxy)||(p->z > root->maxz)) ret = true;	
//	printf("\tmin abs : (%f : %f : %f)\t max abs : (%f : %f : %f)\n",root->minx,root->miny,root->minz, root->maxx,root->maxy,root->maxz);
//	printf("\tpar pos : (%f : %f : %f)\n",p->x,p->y,p->z);
	return ret;
}


/*
Clean tree root
*/
void clean_tree(node * root) {
	int i;
	if (root == NULL) {return;}

	if(root->children != NULL){
		for (i = 0; i < 8; i++){
			clean_tree(&root->children[i]);
		}
		free(root->children);
		root->children = NULL;
		root->sub_nbr_particles=0;
	}
/*
	}else{
		free(root->children);
		root->children = NULL;
	}
	free(root->children);
	root->children = NULL;
*/
}



/*
compute the forces on the BH tree (I think it is the force by other particles in its sub-tree)
1. If the current node is an external node (and it is not body b), calculate the force exerced by the current node on b, and add this amount to b’s net force.
*/

void compute_bh_force(node * n, int prank, int psize) {
	int i, j;
	if(n->children != NULL){
		for (i = 0; i < 8; i++){
			compute_bh_force(&n->children[i], prank, psize);
		}
	}else{
		particle_t *particles = n->particle;
		//if(n->particle->mpi_id % psize == prank){
		for (j = 0; j < n->sub_nbr_particles; j++) {
			particle_t *p = &particles[j];
			if (p->mpi_id % psize == prank) {
			//if (prank == 0){
				compute_force_particle(n, p);
			}
		//}
		}
	}
}

/*
Compute force of node n on particle p
*/

void compute_force_particle(node * n, particle_t * p){
	int i;
	double diffx,diffy,diffz,distance;
	double size;
	if ((n==NULL)||(n->sub_nbr_particles==0)){ return;}

	if ((n->particle != NULL)&&(n->children==NULL)) {
		compute_force(p, n->centerx, n->centery,  n->centerz, n->mass) ;
	}
	else{
		size = n->maxx - n->minx;
		diffx = n->centerx - p->x;
		diffy = n->centery - p->y;
		diffz = n->centerz - p->z;
		distance = sqrt(diffx*diffx + diffy*diffy + diffz*diffz);

//	The particle is far away. Use an approximation of the force
		if(size / distance < THETA) {
			compute_force(p, n->centerx, n->centery, n->centerz, n->mass);
		} else {

//      Otherwise, run the procedure recursively on each of the current node's children.
			for(i=0; i<8; i++) {
				compute_force_particle(&n->children[i], p);
			}
		}
	}
}


/*
Compute force 
*/

void compute_force(particle_t *p, double xpos, double ypos,  double zpos, double mass) {
	double xsep, ysep, zsep, dist_sq, gravity;

	xsep = xpos - p->x;
	ysep = ypos - p->y;
	zsep = zpos - p->z;
	dist_sq = max((xsep*xsep) + (ysep*ysep) + (zsep*zsep), 0.01);

	gravity = GRAV_CONSTANT*(p->m)*(mass)/ dist_sq / sqrt(dist_sq);

	p->fx += gravity*xsep;
	p->fy += gravity*ysep;
	p->fz += gravity*zsep;
}

/*
Compute all the forces in the particles
*/

void compute_force_in_node(node *n, node *root, int prank, int psize) {
    int i, j;
    if (n == NULL) return;

    if ((n->particle != NULL) && (n->children == NULL)) {
        particle_t *particles = n->particle;

        for (i = 0; i < n->sub_nbr_particles; i++) {
            particle_t *p = &particles[i];
			p->fx = 0;
			p->fy = 0;
			p->fz = 0;
			//printf("Particle %i | %i\n", p->id, p->mpi_id);
			if ((p->mpi_id % psize) == prank) {
			//if (prank == 0) {
				compute_force_particle(root, p);
			} 
    	}
	}

    if (n->children != NULL) {
        for (j = 0; j < 8; j++) {
            compute_force_in_node(&n->children[j], root, prank, psize);
        }
    }
}


/*
Construction of the barnes-hut tree

Reminder:
Construction of the Barnes-Hut Tree in 2D
http://arborjs.org/docs/barnes-hut
*/

void construct_bh_tree(particle_t * array, node * root, int nbr_particles){
	int i;
	for (i=0;i < nbr_particles; i++){
		insert_particle(&array[i],root);
	}
}


/*
Add particle p in node n or one of its children
*/

void insert_particle(particle_t * p, node * n){
	int octrant ;
	double totalmass = 0.;
	double totalx = 0.;
	double totaly = 0.;
	double totalz = 0.;
	int i;
// there is no particle
	if ((n->sub_nbr_particles == 0)&&(n->children==NULL)) {
		n->particle = p;
		n->centerx = p->x;
		n->centery = p->y;
		n->centerz = p->z;
		n->mass = p->m;
		n->sub_nbr_particles++;
		p->node = n;
// There is already a particle
	}else{
		if (n->children==NULL){
			create_children(n);
			particle_t * particule_parent = n->particle;
// Insert the particle in the correct children
			octrant = get_octrant(particule_parent,n);
			n->particle = NULL;
			insert_particle(particule_parent,&n->children[octrant]);
		}
// insert the particle p
		octrant = get_octrant(p,n);
		insert_particle(p,&n->children[octrant]);

// Update mass and barycenter (sum of momentums / total mass)
		for(i=0; i<8; i++) {
			totalmass += n->children[i].mass;
			totalx += n->children[i].centerx*n->children[i].mass;
			totaly += n->children[i].centery*n->children[i].mass;
			totalz += n->children[i].centerz*n->children[i].mass;
		}
		n->mass = totalmass;
		n->centerx = totalx/totalmass;
		n->centery = totaly/totalmass;
		n->centerz = totalz/totalmass;
		p->node = n;
		n->sub_nbr_particles++;
	}
}

/*
create 8 children from 1 node
*/

void create_children(node * n){
	n->children = malloc(8*sizeof(node));

	double x12 = n->minx+(n->maxx-n->minx)/2.;
	double y12 = n->miny+(n->maxy-n->miny)/2.;
	double z12 = n->minz+(n->maxz-n->minz)/2.;

	init_node(&n->children[SW_DOWN], n, n->minx, x12, n->miny, y12, n->minz, z12 );
	init_node(&n->children[NW_DOWN], n, n->minx, x12, n->miny, y12, z12, n->maxz );

	init_node(&n->children[SE_DOWN], n, n->minx, x12, y12, n->maxy, n->minz, z12 );
	init_node(&n->children[NE_DOWN], n, n->minx, x12, y12, n->maxy, z12, n->maxz );

	init_node(&n->children[SW_UP], n, x12, n->maxx, n->miny, y12, n->minz, z12 );
	init_node(&n->children[NW_UP], n, x12, n->maxx, n->miny, y12, z12, n->maxz );

	init_node(&n->children[SE_UP], n, x12, n->maxx, y12, n->maxy, n->minz, z12 );
	init_node(&n->children[NE_UP], n, x12, n->maxx, y12, n->maxy, z12, n->maxz );	
}

/*
Init a node n attached to parent parent. 
*/

void init_node(node * n, node * parent,  double minx, double maxx, double miny, double maxy, double minz, double maxz ){
	n->parent=parent;
	n->children = NULL;
	n->minx = minx;
	n->maxx = maxx;
	n->miny = miny;
	n->maxy = maxy;
	n->minz = minz;
	n->maxz = maxz;
	n->depth = parent->depth + 1;
	n->particle = NULL;
	n->sub_nbr_particles = 0.;
	n->centerx = 0.;
	n->centery = 0.;
	n->centerz = 0.;
	n->mass = 0.;
}



/*
get the "octrant" where the particle resides (octrant is a generalization in 3D of a 2D quadrant)
*/

int get_octrant(particle_t * p, node * n){
	int octrant=-1;
	double xmin = n->minx;
	double xmax = n->maxx;
	double x_center = xmin+(xmax-xmin)/2;

	double ymin = n->miny;
	double ymax = n->maxy;
	double y_center = ymin+(ymax-ymin)/2;

	double zmin = n->minz;
	double zmax = n->maxz;
	double z_center = zmin+(zmax-zmin)/2;
	if (n==NULL) printf("ERROR: node is NULL \n");
	if (p==NULL) printf("ERROR: particle is NULL \n");

	// order : x -> y -> z
	if(p->x <= x_center) {
		if(p->y <= y_center) {
			if(p->z <= z_center) {
				octrant = SW_DOWN;
			}else{
				octrant = NW_DOWN;
			}
		} else {
			if(p->z <= z_center) {
				octrant = SE_DOWN;
			}else{
				octrant = NE_DOWN;
			}
		}
	} else {
		if(p->y <= y_center) {
			if(p->z <= z_center) {
				octrant = SW_UP;
			}else{
				octrant = NW_UP;
			}
		} else {
			if(p->z <= z_center) {
				octrant = SE_UP;
			}else{
				octrant = NE_UP;
			}
		}
	}
	return octrant;
}

/*
Init the tree

Remark :We use a particle struct to transfer min and max values from main
*/

void init_tree(particle_t * particle, node * root){
	root->minx = particle->x;
	root->maxx = particle->vx;
	root->miny = particle->y;
	root->maxy = particle->vy;
	root->minz = particle->z;
	root->maxz = particle->vz;
	root->particle = NULL;
	root->sub_nbr_particles = 0;
	root->parent = NULL;
	root->children = NULL;
	root->centerx = 0.;
	root->centery = 0.;
	root->centerz = 0.;
	root->mass = 0.;
	root->depth = 0;
}

/*
============================================
Utilities for testing
============================================
*/


/* print the tree */
void print_tree(node * root){
	node * tmp;
	int i;
	if (root->children!=NULL){
		for (i =0;i<8;i++){
			tmp = &root->children[i];
			print_tree(tmp);
		}
	}
	print_node(root);

}


/* 
print a node 
*/
void print_node(node * n){
	int d = n->depth;
	int i;
	for (i=0;i<d;i++){
		printf("\t");
	}
	printf("[level %d]",d);
	printf(" ([%f:%f:%f])",n->centerx, n->centery,n->centerz);
	printf(" Node ");
	printf(" M = %f", n->mass);
	printf(" has %d particles ", n->sub_nbr_particles);
	if (n->particle!=NULL){
		particle_t * p = n->particle;
		printf(". Particle ID = %d",p->id);
	}
	printf("\n");
}


/*
print a particle 
*/
void print_particle(particle_t * p, int prank, int psize){
	printf("Prank %d/%d | ", prank, psize);
	printf("[Particle %d, %d]",p->id, p->mpi_id);
	printf(" position ([%f:%f:%f])",p->x, p->y, p->z);
	printf("force ([%f:%f:%f])",p->fx, p->fy, p->fz);
	printf(" M = %f", p->m);
	printf("\n");
}

void print_particle_it(particle_t * p, int prank, int psize, int n){
	printf("It %d ||", n);
	printf("Prank %d/%d | ", prank, psize);
	printf("[Particle %d, %d]",p->id, p->mpi_id);
	printf(" position ([%f:%f:%f])",p->x, p->y, p->z);
	printf("force ([%f:%f:%f])",p->fx, p->fy, p->fz);
	printf(" M = %f", p->m);
	printf("\n");
}


/*
OTHER MPI FUNCTIONS
*/


// void gather_force_vector(node * n, double *forces) {
//     int i, j;
//     if (n == NULL) return;

//     if ((n->particle != NULL) && (n->children == NULL)) {
//         particle_t *particles = n->particle;

//         for (i = 0; i < n->sub_nbr_particles; i++) {
//             particle_t *p = &particles[i];
// 			forces[3 * p->mpi_id] = p->fx;    // x-component of force for particle i
// 			forces[3 * p->mpi_id + 1] = p->fy;    // y-component of force for particle i
// 			forces[3 * p->mpi_id + 2] = p->fz;    // z-component of force for particle i
//         }
//     }

//     if (n->children != NULL) {
//         for (j = 0; j < 8; j++) {
//             gather_force_vector(&n->children[j], forces);
//         }
//     }
// }



void broadcast_force_vector_array(particle_t * array, double *forces, int nbr_particles) {
	for(int i=0; i<nbr_particles; i++) {
		//if(i%psize == prank) {
			array[i].fx = forces[3 * i];    // x-component of force for particle i
			array[i].fy = forces[3 * i + 1];    // y-component of force for particle i
			array[i].fz = forces[3 * i + 2];    // z-component of force for particle i
		//}
	}
}

void gather_force_vector_array(particle_t * array, double *forces, int nbr_particles, int prank, int psize) {
	for(int i=0; i<nbr_particles; i++) {
		if(i%psize == prank) {
			forces[3 * i] = array[i].fx;    // x-component of force for particle i
			forces[3 * i + 1] = array[i].fy;    // y-component of force for particle i
			forces[3 * i + 2] = array[i].fz;    // z-component of force for particle i
		} else {
			forces[3 * i] = 0;    // x-component of force for particle i
			forces[3 * i + 1] = 0;    // y-component of force for particle i
			forces[3 * i + 2] = 0;    // z-component of force for particle i
		}
	}
}


// void broadcast_force_vector(node * n, double *forces) {
//     int i, j;
//     if (n == NULL) return;

//     if ((n->particle != NULL) && (n->children == NULL)) {
//         particle_t *particles = n->particle;

//         for (i = 0; i < n->sub_nbr_particles; i++) {
//             particle_t *p = &particles[i];
// 			p->fx = forces[3 * p->mpi_id];    // x-component of force for particle i
// 			p->fy = forces[3 * p->mpi_id + 1];    // y-component of force for particle i
// 			p->fz = forces[3 * p->mpi_id + 2];    // z-component of force for particle i
//         }
//     }

//     if (n->children != NULL) {
//         for (j = 0; j < 8; j++) {
//             broadcast_force_vector(&n->children[j], forces);
//         }
//     }
// }




void compare_arrays(particle_t * array, int nbr_particles, int prank, int psize) {
	particle_t *gathered_arrays = NULL;
	bool equal = true;

	if (prank == 0) {
		gathered_arrays = malloc(sizeof(particle_t) * nbr_particles * psize);
	}
	
	MPI_Gather(array, nbr_particles * sizeof(particle_t), MPI_BYTE, gathered_arrays, nbr_particles * sizeof(particle_t), MPI_BYTE, 0, MPI_COMM_WORLD);

	if (prank == 0) {
		for (int i = 0; i < psize - 1 && equal; ++i) {
			for (int j = i + 1; j < psize && equal; ++j) {
				for (int k = 0; k < nbr_particles && equal; ++k) {
					int index1 = i * nbr_particles + k;
					int index2 = j * nbr_particles + k;

					particle_t *particle1 = &gathered_arrays[index1];
					particle_t *particle2 = &gathered_arrays[index2];

					// Compare each field of the particle struct
					if (particle1->x != particle2->x || particle1->y != particle2->y || particle1->z != particle2->z ||
						particle1->vx != particle2->vx || particle1->vy != particle2->vy || particle1->vz != particle2->vz ||
						particle1->fx != particle2->fx || particle1->fy != particle2->fy || particle1->fz != particle2->fz ||
						particle1->m != particle2->m || particle1->id != particle2->id || particle1->mpi_id != particle2->mpi_id ||
						particle1->V != particle2->V) {
						printf(" \n\n %d || ERROR: Arrays are not the same!\n\n", prank);
						equal = false;
					}
				}
			}
		}
		if(equal) {
		printf(" \n\n %d || GOOD: Arrays equal!\n\n", prank);
		}
		if (prank == 0) {
			free(gathered_arrays);
		}
	}
}




void compare_arrays_except_forces(particle_t * array, int nbr_particles, int prank, int psize) {
	particle_t *gathered_arrays = NULL;
	bool equal = true;

	if (prank == 0) {
		gathered_arrays = malloc(sizeof(particle_t) * nbr_particles * psize);
	}
	
	MPI_Gather(array, nbr_particles * sizeof(particle_t), MPI_BYTE, gathered_arrays, nbr_particles * sizeof(particle_t), MPI_BYTE, 0, MPI_COMM_WORLD);

	if (prank == 0) {
		for (int i = 0; i < psize - 1 && equal; ++i) {
			for (int j = i + 1; j < psize && equal; ++j) {
				for (int k = 0; k < nbr_particles && equal; ++k) {
					int index1 = i * nbr_particles + k;
					int index2 = j * nbr_particles + k;

					particle_t *particle1 = &gathered_arrays[index1];
					particle_t *particle2 = &gathered_arrays[index2];

					// Compare each field of the particle struct
					if (particle1->x != particle2->x || particle1->y != particle2->y || particle1->z != particle2->z ||
						particle1->vx != particle2->vx || particle1->vy != particle2->vy || particle1->vz != particle2->vz ||
						// particle1->fx != particle2->fx || particle1->fy != particle2->fy || particle1->fz != particle2->fz ||
						particle1->m != particle2->m || particle1->id != particle2->id || particle1->mpi_id != particle2->mpi_id ||
						particle1->V != particle2->V) {
						printf(" \n\n %d || ERROR: Arrays are not the same!\n\n", prank);
						equal = false;
					}
				}
			}
		}
		if(equal) {
		printf(" \n\n %d || GOOD: Arrays equal!\n\n", prank);
		}
		if (prank == 0) {
			free(gathered_arrays);
		}
	}
}

void check_no_f_if_not_rank(particle_t * array, int nbr_particles, int prank, int psize) {
	bool equal = true;
	for(int i=0; i<nbr_particles; i++) {
		if(array[i].mpi_id % psize != prank) {
			if(array[i].fx != 0 || array[i].fy != 0 || array[i].fz != 0) {
				printf("%d || ERROR: Particle %d has non-zero forces!\n", prank, array[i].mpi_id);
				equal = false;
			}
		}
	}
	if(equal) {
	printf(" \n\n %d || GOOD: All forces in other ranks 0!\n\n", prank);
	}
}


void check_no_f_if_not_rank_forces(double *forces, int nbr_particles, int prank, int psize) {
	bool equal = true;
	for(int i=0; i<nbr_particles && equal; i++) {
		if(i % psize != prank) {
			if(forces[3 * i] != 0 || forces[3 * i + 1] != 0 || forces[3 * i + 2] != 0) {
				printf("\n\n%d || ERROR: in force vector, particle %d has non-zero forces \n\n",prank, i);
				equal = false;
			}
		}
	}
	if(equal) {
	printf(" \n\n%d || GOOD: Vector forces is ok. \n\n", prank);
	}

}