#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include <unistd.h>


void exchangeElements(double* first, double* second) {
    double temporary = *first;
    *first = *second;
    *second = temporary;
}

int splitArray(double* dataset, int leftBound, int rightBound) {
    if(leftBound >= rightBound) return leftBound;

    double pivotElement = dataset[rightBound];
    int separatorIndex = leftBound - 1;

    for(int currentPosition = leftBound; currentPosition < rightBound; currentPosition++) {
        if(dataset[currentPosition] < pivotElement) {
            separatorIndex++;
            exchangeElements(&dataset[separatorIndex], &dataset[currentPosition]);
        }
    }
    exchangeElements(&dataset[separatorIndex + 1], &dataset[rightBound]);
    return separatorIndex + 1;
}

void sequentialQuickSort(double* data, int startIndex, int endIndex) {
    if(startIndex < endIndex) {
        int divisionPoint = splitArray(data, startIndex, endIndex);
        sequentialQuickSort(data, startIndex, divisionPoint - 1);
        sequentialQuickSort(data, divisionPoint + 1, endIndex);
    }
}

void parallelQuickSort(double* data, int startIndex, int endIndex) {
    if(startIndex < endIndex) {
        int divisionPoint = splitArray(data, startIndex, endIndex);

        #pragma omp task shared(data)
        {
            parallelQuickSort(data, startIndex, divisionPoint - 1);
        }
        #pragma omp task shared(data)
        {
            parallelQuickSort(data, divisionPoint + 1, endIndex);
        }
    }
}

int main(int argc, char* argv[]) {
    printf("2. \n");
    int elementCount = 100000;
    int userOption;

    while((userOption = getopt(argc, argv, "n:")) != -1) {
        switch(userOption) {
            case 'n':
                elementCount = atoi(optarg);
                if(elementCount <= 0) {
                    fprintf(stderr, "Invalid value: Element count must be positive\n");
                    return EXIT_FAILURE;
                }
                break;
            default:
                fprintf(stderr, "Usage: %s [-n elements_number]\n", argv[0]);
                return EXIT_FAILURE;
        }
    }

    double* sequentialArray = malloc(elementCount * sizeof(*sequentialArray));
    double* parallelArray = malloc(elementCount * sizeof(*parallelArray));

    if(!sequentialArray || !parallelArray) {
        perror("Memory allocation error");
        return EXIT_FAILURE;
    }

    srand((unsigned)(time(NULL) ^ getpid()));
    for(int i = 0; i < elementCount; i++) {
        double randomValue = (double)rand() / RAND_MAX * 1000.0;
        sequentialArray[i] = parallelArray[i] = randomValue;
    }

    double seqStart = omp_get_wtime();
    sequentialQuickSort(sequentialArray, 0, elementCount - 1);
    double seqDuration = omp_get_wtime() - seqStart;

    double parStart = omp_get_wtime();
    #pragma omp parallel
    {
        #pragma omp single nowait
        {
            parallelQuickSort(parallelArray, 0, elementCount - 1);
        }
    }
    double parDuration = omp_get_wtime() - parStart;

    printf("Sequential execution time: %.5f s\n", seqDuration);
    printf("Parallel execution time:   %.5f s\n", parDuration);

    free(sequentialArray);
    free(parallelArray);

    return EXIT_SUCCESS;
}