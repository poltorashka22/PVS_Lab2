// first_mpi.c
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <mpi.h>

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int totalElements = 100000;
    int userOption;

    // Процесс 0 обрабатывает аргументы командной строки
    if (rank == 0) {
        printf("1. MPI Summation\n");
        while ((userOption = getopt(argc, argv, "n:")) != -1) {
            switch (userOption) {
            case 'n':
                totalElements = atoi(optarg);
                if (totalElements <= 0) {
                    fprintf(stderr, "Invalid input: Size must be positive.\n");
                    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
                }
                break;
            default:
                fprintf(stderr, "Usage: %s [-n element_count]\n", argv[0]);
                MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
            }
        }
    }

    // Рассылаем totalElements всем процессам
    MPI_Bcast(&totalElements, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (totalElements % size != 0) {
        if (rank == 0) {
            fprintf(stderr, "Warning: Total elements is not divisible by the number of processes. Adjusting size.\n");
            totalElements = (totalElements / size) * size; // Округляем до ближайшего кратного
        }
        // Убедимся, что все процессы получили обновленное значение
        MPI_Bcast(&totalElements, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }


    int localElements = totalElements / size;
    double* dataArray = NULL;
    double* localArray = malloc(localElements * sizeof(double));

    if (!localArray) {
        perror("Memory allocation for local array failed");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    // Процесс 0 инициализирует данные
    if (rank == 0) {
        dataArray = malloc(totalElements * sizeof(double));
        if (!dataArray) {
            perror("Memory allocation failed");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
        srand((unsigned int)time(NULL));
        for (int idx = 0; idx < totalElements; idx++) {
            dataArray[idx] = (double)rand() / (double)RAND_MAX;
        }

        // --- Последовательное вычисление для сравнения ---
        double sequentialSum = 0.0;
        double seqStartTime = MPI_Wtime();
        for (int idx = 0; idx < totalElements; idx++) {
            sequentialSum += dataArray[idx];
        }
        double seqDuration = MPI_Wtime() - seqStartTime;
        printf("Sequential time: %.5f s\n", seqDuration);
    }

    double parStartTime = MPI_Wtime();

    // Распределяем данные от процесса 0 ко всем процессам
    MPI_Scatter(dataArray, localElements, MPI_DOUBLE,
        localArray, localElements, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Каждый процесс считает свою локальную сумму
    double localSum = 0.0;
    for (int idx = 0; idx < localElements; idx++) {
        localSum += localArray[idx];
    }

    // Собираем все локальные суммы в одну общую на процессе 0
    double parallelSum = 0.0;

    // ======================================================================
    // ВОТ ИСПРАВЛЕННАЯ СТРОКА. УБЕДИТЕСЬ, ЧТО ОНА ВЫГЛЯДИТ ИМЕННО ТАК.
    MPI_Reduce(&localSum, &parallelSum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    // ======================================================================

    double parDuration = MPI_Wtime() - parStartTime;

    if (rank == 0) {
        printf("Parallel time:   %.5f s\n", parDuration);
        free(dataArray);
    }

    free(localArray);
    MPI_Finalize();
    return EXIT_SUCCESS;
}