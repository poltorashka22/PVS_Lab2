
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <cuda_runtime.h>

__global__ void reduce_sum(double *input, double *output, int N) {
    extern __shared__ double sdata[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (idx < N) ? input[idx] : 0.0;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

int main(int argc, char* argv[]) {
    int N = 100000;
    int opt;
    while ((opt = getopt(argc, argv, "n:")) != -1) {
        if (opt == 'n') {
            N = atoi(optarg);
            if (N <= 0) {
                fprintf(stderr, "Error: N must be a positive integer.\n");
                return EXIT_FAILURE;
            }
        } else {
            fprintf(stderr, "Usage: %s [-n array_size]\n", argv[0]);
            return EXIT_FAILURE;
        }
    }

    double* array = (double*)malloc(N * sizeof(double));
    if (!array) {
        perror("malloc");
        return EXIT_FAILURE;
    }

    srand((unsigned)time(NULL) ^ getpid());
    for (int i = 0; i < N; ++i) {
        array[i] = (double)rand() / RAND_MAX;
    }

    clock_t start_seq = clock();
    double sum_seq = 0.0;
    for (int i = 0; i < N; ++i) {
        sum_seq += array[i];
    }
    clock_t end_seq = clock();
    printf("Sequential time: %.10f seconds\n",
           (double)(end_seq - start_seq) / CLOCKS_PER_SEC);

    double *d_array, *d_partial_sums;
    double *partial_sums = NULL;
    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    cudaMalloc((void**)&d_array, N * sizeof(double));
    cudaMemcpy(d_array, array, N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&d_partial_sums, blocks * sizeof(double));
    partial_sums = (double*)malloc(blocks * sizeof(double));

    cudaEvent_t start, stop;
    float milliseconds = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    reduce_sum<<<blocks, threads, threads * sizeof(double)>>>(d_array, d_partial_sums, N);
    cudaMemcpy(partial_sums, d_partial_sums, blocks * sizeof(double), cudaMemcpyDeviceToHost);

    double total_sum = 0.0;
    for (int i = 0; i < blocks; ++i) {
        total_sum += partial_sums[i];
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Parallel time: %.10f seconds\n", milliseconds / 1000.0);

    cudaFree(d_array);
    cudaFree(d_partial_sums);
    free(array);
    free(partial_sums);

    return 0;
}
