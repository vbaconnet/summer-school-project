/*
 * Simplified simulation of high-energy particle storms
 *
 * Parallel computing (Degree in Computer Engineering)
 * 2017/2018
 *
 * Version: 2.0
 *
 * Code prepared to be used with the Tablon on-line judge.
 * The current Parallel Computing course includes contests using:
 * OpenMP, MPI, and CUDA.
 *
 * (c) 2018 Arturo Gonzalez-Escribano, Eduardo Rodriguez-Gutiez
 * Grupo Trasgo, Universidad de Valladolid (Spain)
 *
 * This work is licensed under a Creative Commons Attribution-ShareAlike 4.0 International License.
 * https://creativecommons.org/licenses/by-sa/4.0/
 */
#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<sys/time.h>

/* Headers for the MPI assignment versions */
#include<mpi.h>

/* Use fopen function in local tests. The Tablon online judge software
   substitutes it by a different function to run in its sandbox */
#ifdef CP_TABLON
#include "cputilstablon.h"
#else
#define    cp_open_file(name) fopen(name,"r")
#endif

/* Function to get wall time */
double cp_Wtime(){
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + 1.0e-6 * tv.tv_usec;
}


#define THRESHOLD    0.001f

/* Structure used to store data for one storm of particles */
typedef struct {
    int size;    // Number of particles
    int *posval; // Positions and values
} Storm;

/* Compute global index from the local_sizes and the rank that has the maximum
 * Example:
 *   total size = 14
 *   local_sizes = [5,5,4]
 *   local_index = 3
 *
 *   LOCAL  --> 0 1 2 3 4 | 0 1 2 3 4 |  0  1  2  3
 *   GLOBAL --> 0 1 2 3 4 | 5 6 7 8 9 | 10 11 12 13
 *                    ^           ^               ^
 *                    |           |               |
 *   for rank = 0, the global index would be 3
 *   for rank = 1, the global index would be 8
 *   for rank = 2, the global index would be 13
 *
 * */
int get_global_index(int local_index, int rank, int local_sizes[]) {
    int global_index = local_index;

    for (int i=0; i < rank; i++)
        global_index += local_sizes[i];

    return global_index;
}

/* THIS FUNCTION CAN BE MODIFIED */
/* Function to update a single position of the layer */
void update( float *layer, int layer_size, int k, int pos, float energy, int rank, int local_sizes[]) {

    /* 1. Compute the absolute value of the distance between the
        impact position and the k-th position of the layer */
    int distance = pos - get_global_index(k, rank, local_sizes);
    if ( distance < 0 ) distance = - distance;

    /* 2. Impact cell has a distance value of 1 */
    distance = distance + 1;

    /* 3. Square root of the distance */
    /* NOTE: Real world atenuation typically depends on the square of the distance.
       We use here a tailored equation that affects a much wider range of cells */
    float atenuacion = sqrtf( (float)distance );

    /* 4. Compute attenuated energy */
    float energy_k = energy / layer_size / atenuacion;

    /* 5. Each process checks if it should update the layer at position k */
    if (energy_k >= THRESHOLD / layer_size || energy_k <= -THRESHOLD / layer_size) {
        layer[k] = layer[k] + energy_k;
    }
}


/* ANCILLARY FUNCTIONS: These are not called from the code section which is measured, leave untouched */
/* DEBUG function: Prints the layer status */
void debug_print(int layer_size, float *layer, int *positions, float *maximum, int num_storms, int rank, int local_sizes[] ) {
    int i,k;
    /* Only print for array size up to 35 (change it for bigger sizes if needed) */
    if ( layer_size <= 35 ) {
        /* Traverse layer */
        for( k=0; k<layer_size; k++ ) {
            /* Print the energy value of the current cell */
            printf("%10.4f |", layer[k] );

            /* Compute the number of characters.
               This number is normalized, the maximum level is depicted with 60 characters */
            int ticks = (int)( 60 * layer[k] / maximum[num_storms-1] );

            /* Print all characters except the last one */
            for (i=0; i<ticks-1; i++ ) printf("o");

            /* If the cell is a local maximum print a special trailing character */
            if ( k>0 && k<layer_size-1 && layer[k] > layer[k-1] && layer[k] > layer[k+1] )
                printf("x");
            else
                printf("o");

            /* If the cell is the maximum of any storm, print the storm mark */
            for (i=0; i<num_storms; i++)
                if ( positions[i] == get_global_index(k, rank, local_sizes) ) printf(" M%d", i );

            /* Line feed */
            printf("\n");
        }
    }
}

/*
 * Function: Read data of particle storms from a file
 */
Storm read_storm_file( char *fname ) {
    FILE *fstorm = cp_open_file( fname );
    if ( fstorm == NULL ) {
        fprintf(stderr,"Error: Opening storm file %s\n", fname );
        exit( EXIT_FAILURE );
    }

    Storm storm;
    int ok = fscanf(fstorm, "%d", &(storm.size) );
    if ( ok != 1 ) {
        fprintf(stderr,"Error: Reading size of storm file %s\n", fname );
        exit( EXIT_FAILURE );
    }

    storm.posval = (int *)malloc( sizeof(int) * storm.size * 2 );
    if ( storm.posval == NULL ) {
        fprintf(stderr,"Error: Allocating memory for storm file %s, with size %d\n", fname, storm.size );
        exit( EXIT_FAILURE );
    }

    int elem;
    for ( elem=0; elem<storm.size; elem++ ) {
        ok = fscanf(fstorm, "%d %d\n",
                    &(storm.posval[elem*2]),
                    &(storm.posval[elem*2+1]) );
        if ( ok != 2 ) {
            fprintf(stderr,"Error: Reading element %d in storm file %s\n", elem, fname );
            exit( EXIT_FAILURE );
        }
    }
    fclose( fstorm );

    return storm;
}

/* Subdivide the domain for each rank */
int redistribute_layer_size(int rank, int size, int layer_size, int layer_sizes[]) {

    int local_layer_size, remaining;

    /* 1. Divide the domain into equal chunks */
    local_layer_size = layer_size / size;

    /* 2. Determine is the domain size is proportional to number of ranks */
    remaining = layer_size % size;

    for (int i=0; i < size; i++) layer_sizes[i] = local_layer_size;

    if (remaining == 0) return local_layer_size;

    /* 3. Redistribute the points to each rank. remaining is at most equal to size - 2 */
    for (int i=0; i < remaining; i++) {
        if (rank == i) local_layer_size += 1;
        layer_sizes[i] += 1;
    }
    return local_layer_size;
}

/* Find the maximum */
void find_maximum(float *maximum, int *position, float layer[], int local_layer_size, int rank, int size) {

    /* Look for the maximum */
    for(int k=1; k<local_layer_size-1; k++ ) {
        /* Check it only if it is a local maximum */
        if ( layer[k] > layer[k-1] && layer[k] > layer[k+1] ) {
            if ( layer[k] > *maximum ) {
                *maximum = layer[k];
                *position = k;
            }
        }
    }

    /* Check the boundaries too*/
    if (rank != 0 && layer[0] > *maximum) {
        *maximum = layer[0];
        *position = 0;
    }
    if (rank != size - 1 && layer[local_layer_size-1] > *maximum) {
        *maximum = layer[local_layer_size - 1];
        *position = local_layer_size - 1;
    }
}


/*
 * MAIN PROGRAM
 */
int main(int argc, char *argv[]) {
    int i,j,k;

    int rank, size;

    int local_layer_size, local_position;
    float local_maximum;
    int *local_sizes;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    local_sizes = (int *)malloc(size * sizeof(int)); // will contain the sizes of all ranks
    for (i = 0 ; i<size; i++) local_sizes[i] = 0;

    /* 1.1. Read arguments */
    if (argc<3) {
        fprintf(stderr,"Usage: %s <size> <storm_1_file> [ <storm_i_file> ] ... \n", argv[0] );
        exit( EXIT_FAILURE );
    }

    int layer_size = atoi( argv[1] );
    int num_storms = argc-2;
    Storm storms[ num_storms ];

    // Subdivide the domain for each rank
    local_layer_size = redistribute_layer_size(rank, size, layer_size, local_sizes);

    printf("1. Rank %d has size %d (%d)\n", rank, local_layer_size, local_sizes[rank]);

    /* 1.2. Read storms information */
    if (rank == 0) {
        printf("2. Rank %d is reading storm file (%d/%d)\n", rank, rank+1, size);
        for( i=2; i<argc; i++ ) storms[i-2] = read_storm_file( argv[i] );
    }

    for (i = 0; i < num_storms; i++) {
        MPI_Bcast(&storms[i].size , 1            , MPI_INTEGER, 0, MPI_COMM_WORLD);
        if (rank != 0) storms[i].posval = (int *)malloc(2*storms[i].size * sizeof(int));
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Bcast(storms[i].posval, 2*storms[i].size, MPI_INTEGER, 0, MPI_COMM_WORLD);
    }

    printf("33. Rank %d has posval %d %d\n", rank, storms[0].posval[0], storms[0].posval[1]);

    printf("3. Rank %d has %d storms and 1st storm has %d particles\n", rank, num_storms, storms[0].size);

    /* 1.3. Intialize maximum levels to zero */
    float maximum[num_storms];
    int positions[num_storms];

    for (i=0; i<num_storms; i++) {
        maximum[i] = 0.0f;
        positions[i] = 0;
    }

    /* This will help with the maximum at the end of each storm iteration */
    struct {
        float max;
        int rank_owner;
    } max_and_rank_IN, max_and_rank_OUT;

    max_and_rank_IN.rank_owner = rank;
    printf("4. Rank %d initialized max_and_rank_IN rank owner to %d\n", rank,  max_and_rank_IN.rank_owner );

    /* 2. Begin time measurement */
    MPI_Barrier(MPI_COMM_WORLD);

    double ttotal = cp_Wtime();

    /* START: Do NOT optimize/parallelize the code of the main program above this point */

    printf("5. Rank %d is allocating layers\n", rank);

    /* 3. Allocate memory for the layer and initialize to zero */
    float *layer = (float *)malloc( sizeof(float) * local_layer_size );
    float *layer_copy = (float *)malloc( sizeof(float) * local_layer_size );
    if ( layer == NULL || layer_copy == NULL ) {
        fprintf(stderr,"Error: Allocating the layer memory\n");
        exit( EXIT_FAILURE );
    }
    for( k=0; k<local_layer_size; k++ ) layer[k] = 0.0f;
    for( k=0; k<local_layer_size; k++ ) layer_copy[k] = 0.0f;

    printf("6. Rank %d done allocating\n", rank);

    /* 4. Storms simulation */
    /* Parallelize the storm simulation loop */
    for (i = 0; i < num_storms; i++) {

        printf("7. Rank %d storm %d\n", rank, i+1);

        /* 4.1. Add impacts energies to layer cells */
        /* For each particle */
        for( j=0; j<storms[i].size; j++ ) {
            /* Get impact energy (expressed in thousandths) */
            float energy = (float)storms[i].posval[j*2+1] * 1000;
            /* Get impact position */
            int position = storms[i].posval[j*2];

            /* For each cell in the layer */
            for( k=0; k<local_layer_size; k++ ) {
                /* Update the energy value for the cell */
                update( layer, layer_size, k, position, energy, rank, local_sizes);
            }
        }

        /* Synchronize after the storm simulation */
        MPI_Barrier(MPI_COMM_WORLD);
        
        /* 4.2. Energy relaxation between storms */
        /* 4.2.1. Copy values to the ancillary array */
        /* for( k=0; k<local_layer_size; k++ ) */
        /*     layer_copy[k] = layer[k]; */

        /* 4.2.2. Update layer using the ancillary values.
                  Skip updating the first and last positions */
        /* for( k=1; k<layer_size-1; k++ ) */
        /*     layer[k] = ( layer_copy[k-1] + layer_copy[k] + layer_copy[k+1] ) / 3; */

        /* 4.3. Locate the maximum value in the layer, and its position */
        find_maximum(&local_maximum, &local_position, layer, local_layer_size, rank, size);

        printf("8. Rank %d has maximum %f at position %d (global %d)\n", rank, local_maximum, local_position,
get_global_index(local_position, rank, local_sizes));

        max_and_rank_IN.max = local_maximum;

        // Reduce to find the global maximum
        // NOTE: MPI_MAXLOC will also return the rank that owns the maximum, which will allow
        // us to find the global_
        printf("9. Rank %d launching MPI_Allreduce\n", rank);
        MPI_Allreduce(&max_and_rank_IN, &max_and_rank_OUT, 1, MPI_FLOAT_INT, MPI_MAXLOC, MPI_COMM_WORLD);

        printf("10. Looks like rank %d has the maximum %f\n", max_and_rank_OUT.rank_owner, max_and_rank_OUT.max);

        // Let the owning processor broadcast their maximum local position to all others
        MPI_Bcast(&local_position, 1, MPI_INTEGER, max_and_rank_OUT.rank_owner, MPI_COMM_WORLD);

        positions[i] = get_global_index(local_position, max_and_rank_OUT.rank_owner, local_sizes);
        maximum[i] = max_and_rank_OUT.max;
    }
    
    /* Synchronize after the storm simulation */
    MPI_Barrier(MPI_COMM_WORLD);

    /* END: Do NOT optimize/parallelize the code below this point */

    /* 5. End time measurement */
    MPI_Barrier(MPI_COMM_WORLD);
    ttotal = cp_Wtime() - ttotal;

    /* 6.1 DEBUG: Plot the result (only for layers up to 35 points) */
#ifdef DEBUG
        for (i = 0; i<size; i++) {
            if (rank == i)
                debug_print( local_layer_size, layer, positions, maximum, num_storms, rank, local_sizes );
            MPI_Barrier(MPI_COMM_WORLD);
        }
#endif

    /* 6.2 DEBUG: Print data  */
    if ( rank == 0 ) {
        /* 7. Results output, used by the Tablon online judge software */
        printf("\n");
        /* 7.1. Total computation time */
        printf("Time: %lf\n", ttotal );
        /* 7.2. Print the maximum levels */
        printf("Result:");
        for (i=0; i<num_storms; i++)
            printf(" %d %f", positions[i], maximum[i] );
        printf("\n");

    }

    /* 8. Free resources */    
    for( i=0; i<argc-2; i++ )
        free( storms[i].posval );

    free(local_sizes);
    free(layer);
    free(layer_copy);

    /* 9. Program ended successfully */
    MPI_Finalize();
    return 0;
}

