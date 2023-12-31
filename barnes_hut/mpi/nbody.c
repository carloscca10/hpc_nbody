#include <stdio.h> 
#include <stdlib.h> 
#include <math.h>
#include <stdarg.h>
#include <stddef.h>
#include "parameters.h"
//#include "nbody_bruteforce.h"
#include "nbody_barneshut.h"
#include "reader.h"
#include <sys/time.h>
#include <mpi.h>


void print_parameters(void){
	printf("====================================================\n");
	printf("N-Body 3D simulation code for MATH-454 course EPFL  \n");
	printf("Parameters for the Barnes-Hut algorithm:\n");
	printf("\n");
	printf("Gravitational constant : %f\n",GRAV_CONSTANT);
	printf("Theta                  : %f\n",THETA);
	printf("Time step              : %f\n",TIMESTEP);
	printf("Space multiplicator    : %f\n",SIZEOFSPACE);
	printf("Number of iterations   : %d\n",NBRITERATIONS);
	printf("\n");
	printf("These parameters can be modified in \"parameters.h\"\n");
	printf("\n");
	printf("(c) 2020, Vincent Keller (Vincent.Keller@epfl.ch)\n");
	printf("====================================================\n");
	printf("\n");
}

double second(void)
{
        struct timeval tp;
        struct timezone tzp;
	gettimeofday(&tp,&tzp);
        return ( (double) tp.tv_sec + (double) tp.tv_usec * 1.e-6 );
}



/*
Implementation of a 3D N-Body code in C
Input files are in XYZ
It is possible to download the Gadget2 test cases to test your code

Code largely inspired by http://www-inf.telecom-sudparis.eu/COURS/CSC5001/new_site/Supports/Projet/NBody/sujet.php

*/

// argc: argument count,    argv: argument vector.

// example: ./program arg1 arg2
// argc = 3
// argv[0] = "./program",   argv[1] = "arg1",   argv[2] = "arg2"

int main ( int argc, char **argv ) {

	MPI_Init(&argc, &argv);
	int prank, psize;

	MPI_Comm_rank(MPI_COMM_WORLD, &prank);
	MPI_Comm_size(MPI_COMM_WORLD, &psize);
	//printf("%d: hello (p=%d)\n", prank, psize);

	particle_t * array;
	int nbr_iterations;
	int nbr_particles;
	double t1, t2;

	nbr_iterations = NBRITERATIONS;

	if (prank==0) print_parameters();

	if (argc < 2)
	{
		fprintf(stderr, "Usage: %s [martix-market-filename]\n", argv[0]);
		exit(1); // exit with an error
	}
	else
	{
		//printf("Read data from file \n");
		nbr_particles = get_nbr_particles(argv[1]);
		array = read_test_case(argv[1]);
		//printf("Number of particles : %d\n",nbr_particles);
	}

	/*
	printf("BRUTE FORCE simulation starting \n");
	t1 = second();
	nbodybruteforce(array, nbr_particles, nbr_iterations);
	t2 = second();
	printf("N-Body brute force for %d particles : %f [s] \n",nbr_iterations, (t2-t1));


	printf("BARNES-HUT simulation starting \n");
	array = read_test_case(argv[1]);
*/

	//printf("BARNES-HUT simulation starting \n");
	t1 = second();
	nbodybarneshut(array, nbr_particles, nbr_iterations, prank, psize);
	t2 = second();
	printf("N-Body barnes-hut for %d particles : %f [s] \n",nbr_particles, (t2-t1));
	free(array);


	printf("Simulation finished \n");
	MPI_Finalize();
	return 0;
}


