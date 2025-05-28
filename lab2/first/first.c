#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include <unistd.h>

int main(int argc, char *argv[]) {
  printf("1. \n");
    int totalElements = 100000;
    int userOption;

    while (userOption = getopt(argc, argv, "n:")) {
        if (userOption == -1) break;

        switch (userOption) {
            case 'n':
                totalElements = atoi(optarg);
            if (totalElements <= 0) {
                fprintf(stderr, "Invalid input: Size must be positive.\n");
                exit(EXIT_FAILURE);
            }
            break;
            default:
                fprintf(stderr, "Usage: %s [-n element_count]\n", argv[0]);
            exit(EXIT_FAILURE);
        }
    }

    double *dataArray = malloc(totalElements * sizeof(*dataArray));
    if (!dataArray) {
        perror("Memory allocation failed");
        exit(EXIT_FAILURE);
    }

    double sequentialSum = 0.0, parallelSum = 0.0;

    srand((unsigned int)(time(NULL) ^ getpid()));
    for (int idx = 0; idx < totalElements; idx++) {
        dataArray[idx] = (double)rand() / (double)RAND_MAX;
    }

    double seqStartTime = omp_get_wtime();
    for (int idx = 0; idx < totalElements; idx++) {
        sequentialSum += dataArray[idx];
    }
    double seqDuration = omp_get_wtime() - seqStartTime;

    double parStartTime = omp_get_wtime();
#pragma omp parallel for reduction(+:parallelSum)
    for (int idx = 0; idx < totalElements; idx++) {
        parallelSum += dataArray[idx];
    }
    double parDuration = omp_get_wtime() - parStartTime;

    printf("Sequential processing time: %.5f s\n", seqDuration);
    printf("Parallel processing time:   %.5f s\n", parDuration);

    free(dataArray);
    return EXIT_SUCCESS;
}