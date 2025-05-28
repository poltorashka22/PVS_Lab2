#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>
#include <unistd.h>
#include <getopt.h>

#define EPSILON 1e-9

void initialize_array(double* arr, int size) {
#pragma omp parallel for
    for (int i = 0; i < size; i++) {
        arr[i] = (double)rand() / RAND_MAX * 100.0 + 1.0;
    }
}

void process_operations(double* a, double* b, double* results, int size, int is_parallel) {
    if (is_parallel) {
#pragma omp parallel for
        for (int i = 0; i < size; i++) {
            results[0 * size + i] = a[i] + b[i];
            results[1 * size + i] = a[i] - b[i];
            results[2 * size + i] = a[i] * b[i];
            results[3 * size + i] = (fabs(b[i]) > EPSILON) ? a[i] / b[i] : 0.0;
        }
    }
    else {
        for (int i = 0; i < size; i++) {
            results[0 * size + i] = a[i] + b[i];
            results[1 * size + i] = a[i] - b[i];
            results[2 * size + i] = a[i] * b[i];
            results[3 * size + i] = (fabs(b[i]) > EPSILON) ? a[i] / b[i] : 0.0;
        }
    }
}

int main(int argc, char* argv[]) {
    int array_size = 0;
    int opt;

    while ((opt = getopt(argc, argv, "n:")) != -1) {
        if (opt == 'n') {
            array_size = atoi(optarg);
            if (array_size <= 100000) {
                fprintf(stderr, "Error: Array size must be greater than 100000\n");
                return 1;
            }
        }
    }

    if (array_size == 0) {
        fprintf(stderr, "Usage: %s -n <array_size>\n", argv[0]);
        return 1;
    }

    double* array_a = (double*)malloc(array_size * sizeof(double));
    double* array_b = (double*)malloc(array_size * sizeof(double));
    double* results = (double*)malloc(4 * array_size * sizeof(double));

    srand(42);
    initialize_array(array_a, array_size);
    initialize_array(array_b, array_size);

    double seq_start = omp_get_wtime();
    process_operations(array_a, array_b, results, array_size, 0);
    double seq_time = omp_get_wtime() - seq_start;

    double par_start = omp_get_wtime();
    process_operations(array_a, array_b, results, array_size, 1);
    double par_time = omp_get_wtime() - par_start;

    printf("Последовательное время: %.5f секунд\n", seq_time);
    printf("Параллельное время: %.5f секунд\n", par_time);


    free(array_a);
    free(array_b);
    free(results);

    return 0;
}