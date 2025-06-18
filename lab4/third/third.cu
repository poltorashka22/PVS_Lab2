
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <unistd.h>
#include <getopt.h>
#include <cuda_runtime.h>

#define EPSILON 1e-9

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

void process_operations_cpu(const double* a, const double* b, double* results, int size) {
    for (int i = 0; i < size; i++) {
        results[0 * size + i] = a[i] + b[i];
        results[1 * size + i] = a[i] - b[i];
        results[2 * size + i] = a[i] * b[i];
        results[3 * size + i] = (fabs(b[i]) > EPSILON) ? a[i] / b[i] : 0.0;
    }
}

__global__ void process_operations_kernel(const double* a, const double* b, double* results, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        int base_res_idx = idx * 4;
        results[base_res_idx + 0] = a[idx] + b[idx];
        results[base_res_idx + 1] = a[idx] - b[idx];
        results[base_res_idx + 2] = a[idx] * b[idx];
        results[base_res_idx + 3] = (fabs(b[idx]) > EPSILON) ? a[idx] / b[idx] : 0.0;
    }
}

void verify_results(const double* seq_results, const double* par_results, int size) {
    for (int i = 0; i < size; i++) {
        if (fabs(seq_results[0 * size + i] - par_results[i * 4 + 0]) > EPSILON) {
            printf("Verification failed for ADD at index %d\n", i); return;
        }
        if (fabs(seq_results[1 * size + i] - par_results[i * 4 + 1]) > EPSILON) {
            printf("Verification failed for SUB at index %d\n", i); return;
        }
        if (fabs(seq_results[2 * size + i] - par_results[i * 4 + 2]) > EPSILON) {
            printf("Verification failed for MUL at index %d\n", i); return;
        }
        if (fabs(seq_results[3 * size + i] - par_results[i * 4 + 3]) > EPSILON) {
            printf("Verification failed for DIV at index %d\n", i); return;
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
    printf("Array size: %d\n", array_size);

    size_t array_bytes = array_size * sizeof(double);
    size_t results_bytes = 4 * array_bytes;

    double* h_a = (double*)malloc(array_bytes);
    double* h_b = (double*)malloc(array_bytes);
    double* h_results_seq = (double*)malloc(results_bytes);
    double* h_results_par = (double*)malloc(results_bytes);

    if (!h_a || !h_b || !h_results_seq || !h_results_par) {
        perror("Host memory allocation failed");
        return 1;
    }

    srand(42);
    for (int i = 0; i < array_size; i++) {
        h_a[i] = (double)rand() / RAND_MAX * 100.0 + 1.0;
        h_b[i] = (double)rand() / RAND_MAX * 100.0 + 1.0;
    }

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    process_operations_cpu(h_a, h_b, h_results_seq, array_size);
    clock_gettime(CLOCK_MONOTONIC, &end);
    double seq_time = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    printf("Sequential (CPU) time: %.5f s\n", seq_time);

    double* d_a, * d_b, * d_results;

    clock_gettime(CLOCK_MONOTONIC, &start);

    gpuErrchk(cudaMalloc((void**)&d_a, array_bytes));
    gpuErrchk(cudaMalloc((void**)&d_b, array_bytes));
    gpuErrchk(cudaMalloc((void**)&d_results, results_bytes));

    gpuErrchk(cudaMemcpy(d_a, h_a, array_bytes, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_b, h_b, array_bytes, cudaMemcpyHostToDevice));

    int threadsPerBlock = 256;
    int blocksPerGrid = (array_size + threadsPerBlock - 1) / threadsPerBlock;

    process_operations_kernel << <blocksPerGrid, threadsPerBlock >> > (d_a, d_b, d_results, array_size);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    gpuErrchk(cudaMemcpy(h_results_par, d_results, results_bytes, cudaMemcpyDeviceToHost));

    clock_gettime(CLOCK_MONOTONIC, &end);
    double par_time = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    printf("Parallel (GPU) time:   %.5f s\n", par_time);

    gpuErrchk(cudaFree(d_a));
    gpuErrchk(cudaFree(d_b));
    gpuErrchk(cudaFree(d_results));

    printf("Verifying results...\n");
    verify_results(h_results_seq, h_results_par, array_size);
    printf("Verification complete.\n");

    free(h_a);
    free(h_b);
    free(h_results_seq);
    free(h_results_par);

    return 0;
}