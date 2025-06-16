
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <mpi.h>

void exchangeElements(double* first, double* second) {
    double temporary = *first;
    *first = *second;
    *second = temporary;
}

void bubbleSort(double* arr, int n) {
    for (int i = 0; i < n - 1; i++) {
        for (int j = 0; j < n - i - 1; j++) {
            if (arr[j] > arr[j + 1]) {
                exchangeElements(&arr[j], &arr[j + 1]);
            }
        }
    }
}

void merge(double* arr1, int n1, double* arr2, int n2, double* result) {
    int i = 0, j = 0, k = 0;
    while (i < n1 && j < n2) {
        result[k++] = (arr1[i] < arr2[j]) ? arr1[i++] : arr2[j++];
    }
    while (i < n1) result[k++] = arr1[i++];
    while (j < n2) result[k++] = arr2[j++];
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int elementCount = 100000;

    if (rank == 0) {
        printf("2. MPI Bubble (Odd-Even) Sort\n");
        int userOption;
        while ((userOption = getopt(argc, argv, "n:")) != -1) {
            if (userOption == 'n') {
                elementCount = atoi(optarg);
            }
        }
        if (elementCount <= 0) {
            fprintf(stderr, "Invalid value: Element count must be positive\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    MPI_Bcast(&elementCount, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (elementCount % size != 0) {
        if (rank == 0) fprintf(stderr, "Element count must be divisible by number of processes.\n");
        MPI_Finalize();
        return 1;
    }

    int local_n = elementCount / size;
    double* global_array = NULL;
    double* local_array = malloc(local_n * sizeof(double));

    if (rank == 0) {
        global_array = malloc(elementCount * sizeof(double));
        double* seq_array = malloc(elementCount * sizeof(double));
        srand((unsigned)(time(NULL)));
        for (int i = 0; i < elementCount; i++) {
            double randomValue = (double)rand() / RAND_MAX * 1000.0;
            global_array[i] = seq_array[i] = randomValue;
        }

        double seqStart = MPI_Wtime();
        bubbleSort(seq_array, elementCount);
        double seqDuration = MPI_Wtime() - seqStart;
        printf("Sequential time: %.5f s\n", seqDuration);
        free(seq_array);
    }

    double parStart = MPI_Wtime();

    MPI_Scatter(global_array, local_n, MPI_DOUBLE, local_array, local_n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    double* received_array = malloc(local_n * sizeof(double));
    double* merged_array = malloc(2 * local_n * sizeof(double));

    for (int phase = 0; phase < size; phase++) {
        bubbleSort(local_array, local_n);

        int partner;
        if ((phase % 2) == 0) { 
            if ((rank % 2) == 0) {
                partner = rank + 1;
            }
            else { 
                partner = rank - 1;
            }
        }
        else {
            if ((rank % 2) != 0) {
                partner = rank + 1;
            }
            else {
                partner = rank - 1;
            }
        }

        if (partner < 0 || partner >= size) continue;

        MPI_Sendrecv(local_array, local_n, MPI_DOUBLE, partner, 0,
            received_array, local_n, MPI_DOUBLE, partner, 0,
            MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        merge(local_array, local_n, received_array, local_n, merged_array);

        if (rank < partner) { 
            for (int i = 0; i < local_n; ++i) local_array[i] = merged_array[i];
        }
        else { 
            for (int i = 0; i < local_n; ++i) local_array[i] = merged_array[i + local_n];
        }
    }

    free(received_array);
    free(merged_array);

    MPI_Gather(local_array, local_n, MPI_DOUBLE, global_array, local_n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    double parDuration = MPI_Wtime() - parStart;

    if (rank == 0) {
        printf("Parallel time:   %.5f s\n", parDuration);
        free(global_array);
    }
    free(local_array);

    MPI_Finalize();
    return EXIT_SUCCESS;
}