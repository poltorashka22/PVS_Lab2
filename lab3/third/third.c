
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <unistd.h>
#include <getopt.h>
#include <mpi.h>

#define EPSILON 1e-9

void process_operations(const double* a, const double* b, double* results, int size) {
    for (int i = 0; i < size; i++) {
        results[0 * size + i] = a[i] + b[i];
        results[1 * size + i] = a[i] - b[i];
        results[2 * size + i] = a[i] * b[i];
        results[3 * size + i] = (fabs(b[i]) > EPSILON) ? a[i] / b[i] : 0.0;
    }
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    int array_size = 0;

    if (rank == 0) {
        int opt;
        while ((opt = getopt(argc, argv, "n:")) != -1) {
            if (opt == 'n') {
                array_size = atoi(optarg);
                if (array_size <= 100000) {
                    fprintf(stderr, "Error: Array size must be greater than 100000\n");
                    MPI_Abort(MPI_COMM_WORLD, 1);
                }
            }
        }
        if (array_size == 0) {
            fprintf(stderr, "Usage: %s -n <array_size>\n", argv[0]);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    MPI_Bcast(&array_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (array_size % num_procs != 0) {
        if (rank == 0) {
            fprintf(stderr, "Error: Array size must be divisible by the number of processes.\n");
        }
        MPI_Finalize();
        return 1;
    }

    int local_size = array_size / num_procs;

    double* array_a = NULL;
    double* array_b = NULL;
    double* results = NULL;

    double* local_a = (double*)malloc(local_size * sizeof(double));
    double* local_b = (double*)malloc(local_size * sizeof(double));
    double* local_results = (double*)malloc(4 * local_size * sizeof(double));

    if (rank == 0) {
        array_a = (double*)malloc(array_size * sizeof(double));
        array_b = (double*)malloc(array_size * sizeof(double));
        results = (double*)malloc(4 * array_size * sizeof(double));

        srand(42);
        for (int i = 0; i < array_size; i++) {
            array_a[i] = (double)rand() / RAND_MAX * 100.0 + 1.0;
            array_b[i] = (double)rand() / RAND_MAX * 100.0 + 1.0;
        }

        double seq_start = MPI_Wtime();
        process_operations(array_a, array_b, results, array_size);
        double seq_time = MPI_Wtime() - seq_start;
        printf("Sequential time: %.5f s\n", seq_time);
    }

    double par_start = MPI_Wtime();

    MPI_Scatter(array_a, local_size, MPI_DOUBLE, local_a, local_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatter(array_b, local_size, MPI_DOUBLE, local_b, local_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    process_operations(local_a, local_b, local_results, local_size);

    MPI_Gather(local_results, 4 * local_size, MPI_DOUBLE,
        results, 4 * local_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    double par_time = MPI_Wtime() - par_start;

    if (rank == 0) {
        printf("Parallel time: %.5f s\n", par_time);
        free(array_a);
        free(array_b);
        free(results);
    }

    free(local_a);
    free(local_b);
    free(local_results);

    MPI_Finalize();
    return 0;
}