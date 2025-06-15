#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <sys/time.h>

int is_sorted(int *arr, int n) {
    for (int i = 1; i < n; i++) {
        if (arr[i-1] > arr[i]) {
            return 0;
        }
    }
    return 1;
}

void bubble_sort(int *arr, int n) {
    int i, j, temp;
    for (i = 0; i < n-1; i++) {
        for (j = 0; j < n-i-1; j++) {
            if (arr[j] > arr[j+1]) {
                temp = arr[j];
                arr[j] = arr[j+1];
                arr[j+1] = temp;
            }
        }
    }
}

void parallel_bubble_sort(int *arr, int n) {
    int phase, i, temp;
    for (phase = 0; phase < n; phase++) {
        if (phase % 2 == 0) {
            #pragma omp parallel for private(i, temp)
            for (i = 0; i < n/2; i++) {
                int left = 2*i;
                int right = 2*i + 1;
                if (right < n && arr[left] > arr[right]) {
                    temp = arr[left];
                    arr[left] = arr[right];
                    arr[right] = temp;
                }
            }
        } else {
            #pragma omp parallel for private(i, temp)
            for (i = 0; i < (n-1)/2; i++) {
                int left = 2*i + 1;
                int right = 2*i + 2;
                if (right < n && arr[left] > arr[right]) {
                    temp = arr[left];
                    arr[left] = arr[right];
                    arr[right] = temp;
                }
            }
        }
    }
}

int main() {
    int n = 200000;
    int *arr_seq = malloc(n * sizeof(int));
    int *arr_parallel = malloc(n * sizeof(int));

    for (int i = 0; i < n; i++) {
        arr_seq[i] = arr_parallel[i] = rand();
    }

    struct timeval start_seq, end_seq;
    gettimeofday(&start_seq, NULL);
    bubble_sort(arr_seq, n);
    gettimeofday(&end_seq, NULL);
    double time_seq = (end_seq.tv_sec - start_seq.tv_sec) +
                      (end_seq.tv_usec - start_seq.tv_usec)/1e6;
    printf("Sequential time: %.6f seconds\n", time_seq);

    if (!is_sorted(arr_seq, n)) {
        printf("Sequential sort failed!\n");
    }

    struct timeval start_par, end_par;
    gettimeofday(&start_par, NULL);
    parallel_bubble_sort(arr_parallel, n);
    gettimeofday(&end_par, NULL);
    double time_par = (end_par.tv_sec - start_par.tv_sec) +
                      (end_par.tv_usec - start_par.tv_usec)/1e6;
    printf("Parallel time: %.6f seconds\n", time_par);

    if (!is_sorted(arr_parallel, n)) {
        printf("Parallel sort failed!\n");
    }

    for (int i = 0; i < n; i++) {
        if (arr_seq[i] != arr_parallel[i]) {
            printf("Mismatch at index %d\n", i);
            break;
        }
    }

    free(arr_seq);
    free(arr_parallel);
    return 0;
}