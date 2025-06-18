#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <cuda.h>
#include <cuda_runtime.h>


void merge(double *arr, int l, int m, int r, double *temp) {
    int i = l, j = m + 1, k = l;
    while (i <= m && j <= r) {
        temp[k++] = (arr[i] < arr[j]) ? arr[i++] : arr[j++];
    }
    while (i <= m) temp[k++] = arr[i++];
    while (j <= r) temp[k++] = arr[j++];
    for (int i = l; i <= r; ++i) arr[i] = temp[i];
}

void merge_sort(double *arr, int l, int r, double *temp) {
    if (l < r) {
        int m = (l + r) / 2;
        merge_sort(arr, l, m, temp);
        merge_sort(arr, m + 1, r, temp);
        merge(arr, l, m, r, temp);
    }
}

__global__ void bitonic_sort_step(double *dev_arr, int j, int k) {
    unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int ixj = i ^ j;

    if (ixj > i) {
        if ((i & k) == 0) {
            if (dev_arr[i] > dev_arr[ixj]) {
                double tmp = dev_arr[i];
                dev_arr[i] = dev_arr[ixj];
                dev_arr[ixj] = tmp;
            }
        } else {
            if (dev_arr[i] < dev_arr[ixj]) {
                double tmp = dev_arr[i];
                dev_arr[i] = dev_arr[ixj];
                dev_arr[ixj] = tmp;
            }
        }
    }
}

void bitonic_sort_cuda(double *host_arr, int N) {
    double *dev_arr;
    size_t size = N * sizeof(double);
    cudaMalloc((void **)&dev_arr, size);
    cudaMemcpy(dev_arr, host_arr, size, cudaMemcpyHostToDevice);

    dim3 blocks(N / 1024);
    dim3 threads(1024);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    for (int k = 2; k <= N; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            bitonic_sort_step<<<blocks, threads>>>(dev_arr, j, k);
            cudaDeviceSynchronize();
        }
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    printf("Parallel time: %.10f seconds\n", ms / 1000.0f);

    cudaMemcpy(host_arr, dev_arr, size, cudaMemcpyDeviceToHost);
    cudaFree(dev_arr);
}

int is_power_of_two(int n) {
    return n > 0 && (n & (n - 1)) == 0;
}

int main(int argc, char *argv[]) {
    int N = 1 << 16;
    int opt;
    while ((opt = getopt(argc, argv, "n:")) != -1) {
        if (opt == 'n') {
            N = atoi(optarg);
            if (!is_power_of_two(N)) {
                fprintf(stderr, "Error: N must be a power of 2 and > 0.\n");
                return EXIT_FAILURE;
            }
        } else {
            fprintf(stderr, "Usage: %s [-n array_size]\n", argv[0]);
            return EXIT_FAILURE;
        }
    }

    double *arr_seq = (double *)malloc(N * sizeof(double));
    double *arr_par = (double *)malloc(N * sizeof(double));
    double *temp    = (double *)malloc(N * sizeof(double));
    if (!arr_seq || !arr_par || !temp) {
        perror("malloc");
        free(arr_seq);
        free(arr_par);
        free(temp);
        return EXIT_FAILURE;
    }

    srand(time(NULL));
    for (int i = 0; i < N; ++i) {
        double v = (double)rand() / RAND_MAX;
        arr_seq[i] = arr_par[i] = v;
    }

    clock_t start_seq = clock();
    merge_sort(arr_seq, 0, N - 1, temp);
    clock_t end_seq = clock();
    printf("Sequential time: %.10f seconds\n",
           (double)(end_seq - start_seq) / CLOCKS_PER_SEC);

    bitonic_sort_cuda(arr_par, N);

    free(arr_seq);
    free(arr_par);
    free(temp);
    return 0;
}