
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

typedef struct {
    int rows;
    int cols;
    double* data;
} Matrix;


Matrix create_linear_matrix(int rows, int cols) {
    Matrix mat;
    mat.rows = rows;
    mat.cols = cols;
    mat.data = (double*)malloc(rows * cols * sizeof(double));
    if (mat.data == NULL) {
        perror("Host matrix allocation failed");
        exit(1);
    }
    return mat;
}

void free_linear_matrix(Matrix mat) {
    free(mat.data);
}

void fill_matrix_random(Matrix mat) {
    for (int i = 0; i < mat.rows * mat.cols; i++) {
        mat.data[i] = (double)rand() / RAND_MAX * 100.0 + 1.0;
    }
}

void matrix_operations_cpu(Matrix a, Matrix b, Matrix add, Matrix sub, Matrix mul, Matrix div) {
    int total_size = a.rows * a.cols;
    for (int i = 0; i < total_size; i++) {
        add.data[i] = a.data[i] + b.data[i];
        sub.data[i] = a.data[i] - b.data[i];
        mul.data[i] = a.data[i] * b.data[i];
        div.data[i] = (fabs(b.data[i]) > EPSILON) ? a.data[i] / b.data[i] : 0.0;
    }
}


__global__ void matrix_operations_kernel(const double* a, const double* b,
    double* add, double* sub, double* mul, double* div,
    int n_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n_elements) {
        add[idx] = a[idx] + b[idx];
        sub[idx] = a[idx] - b[idx];
        mul[idx] = a[idx] * b[idx];
        div[idx] = (fabs(b[idx]) > EPSILON) ? a[idx] / b[idx] : 0.0;
    }
}

void verify_results(Matrix seq, Matrix par) {
    int n_elements = seq.rows * seq.cols;
    for (int i = 0; i < n_elements; i++) {
        if (fabs(seq.data[i] - par.data[i]) > EPSILON) {
            printf("Verification failed at index %d: seq=%.6f, par=%.6f\n", i, seq.data[i], par.data[i]);
            return;
        }
    }
}


void print_usage(const char* prog_name) {
    printf("Usage: %s -n <size> [-r <rows> -c <cols>]\n", prog_name);
    printf("Options:\n");
    printf("  -n <size>  Total number of elements (must be > 100000)\n");
    printf("  -r <rows>  Number of rows (optional)\n");
    printf("  -c <cols>  Number of columns (optional)\n");
    printf("  -h         Print this help message\n");
}

int main(int argc, char* argv[]) {
    int total_elements = 0;
    int rows = 0, cols = 0;

    int opt;
    while ((opt = getopt(argc, argv, "n:r:c:h")) != -1) {
        switch (opt) {
        case 'n': total_elements = atoi(optarg); break;
        case 'r': rows = atoi(optarg); break;
        case 'c': cols = atoi(optarg); break;
        case 'h': print_usage(argv[0]); return 0;
        default: print_usage(argv[0]); return 1;
        }
    }

    if (total_elements <= 100000) {
        fprintf(stderr, "Error: Total elements must be > 100000\n");
        return 1;
    }

    if (rows == 0 || cols == 0) {
        rows = (int)sqrt(total_elements);
        cols = total_elements / rows;
        total_elements = rows * cols; 
    }
    else if (rows * cols != total_elements) {
        fprintf(stderr, "Warning: total elements will be adjusted to rows * cols = %d\n", rows * cols);
        total_elements = rows * cols;
    }

    printf("Matrix dimensions: %d x %d (Total: %d elements)\n", rows, cols);

    Matrix A = create_linear_matrix(rows, cols);
    Matrix B = create_linear_matrix(rows, cols);
    Matrix ADD_seq = create_linear_matrix(rows, cols);
    Matrix SUB_seq = create_linear_matrix(rows, cols);
    Matrix MUL_seq = create_linear_matrix(rows, cols);
    Matrix DIV_seq = create_linear_matrix(rows, cols);
    Matrix ADD_par = create_linear_matrix(rows, cols);
    Matrix SUB_par = create_linear_matrix(rows, cols);
    Matrix MUL_par = create_linear_matrix(rows, cols);
    Matrix DIV_par = create_linear_matrix(rows, cols);

    srand(time(NULL));
    fill_matrix_random(A);
    fill_matrix_random(B);

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    matrix_operations_cpu(A, B, ADD_seq, SUB_seq, MUL_seq, DIV_seq);
    clock_gettime(CLOCK_MONOTONIC, &end);
    double seq_time = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    printf("Sequential (CPU) time: %.5f s\n", seq_time);

    double* d_A, * d_B, * d_ADD, * d_SUB, * d_MUL, * d_DIV;
    size_t matrix_size_bytes = total_elements * sizeof(double);

    clock_gettime(CLOCK_MONOTONIC, &start);

    gpuErrchk(cudaMalloc((void**)&d_A, matrix_size_bytes));
    gpuErrchk(cudaMalloc((void**)&d_B, matrix_size_bytes));
    gpuErrchk(cudaMalloc((void**)&d_ADD, matrix_size_bytes));
    gpuErrchk(cudaMalloc((void**)&d_SUB, matrix_size_bytes));
    gpuErrchk(cudaMalloc((void**)&d_MUL, matrix_size_bytes));
    gpuErrchk(cudaMalloc((void**)&d_DIV, matrix_size_bytes));

    gpuErrchk(cudaMemcpy(d_A, A.data, matrix_size_bytes, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_B, B.data, matrix_size_bytes, cudaMemcpyHostToDevice));

    int threadsPerBlock = 256;
    int blocksPerGrid = (total_elements + threadsPerBlock - 1) / threadsPerBlock;

    matrix_operations_kernel << <blocksPerGrid, threadsPerBlock >> > (
        d_A, d_B, d_ADD, d_SUB, d_MUL, d_DIV, total_elements);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    gpuErrchk(cudaMemcpy(ADD_par.data, d_ADD, matrix_size_bytes, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(SUB_par.data, d_SUB, matrix_size_bytes, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(MUL_par.data, d_MUL, matrix_size_bytes, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(DIV_par.data, d_DIV, matrix_size_bytes, cudaMemcpyDeviceToHost));

    clock_gettime(CLOCK_MONOTONIC, &end);
    double par_time = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    printf("Parallel (GPU) time:   %.5f s\n", par_time);

    gpuErrchk(cudaFree(d_A));
    gpuErrchk(cudaFree(d_B));
    gpuErrchk(cudaFree(d_ADD));
    gpuErrchk(cudaFree(d_SUB));
    gpuErrchk(cudaFree(d_MUL));
    gpuErrchk(cudaFree(d_DIV));

    printf("Verifying results...\n");
    verify_results(ADD_seq, ADD_par);
    verify_results(SUB_seq, SUB_par);
    verify_results(MUL_seq, MUL_par);
    verify_results(DIV_seq, DIV_par);
    printf("Verification complete.\n");

    free_linear_matrix(A);
    free_linear_matrix(B);
    free_linear_matrix(ADD_seq);
    free_linear_matrix(SUB_seq);
    free_linear_matrix(MUL_seq);
    free_linear_matrix(DIV_seq);
    free_linear_matrix(ADD_par);
    free_linear_matrix(SUB_par);
    free_linear_matrix(MUL_par);
    free_linear_matrix(DIV_par);

    return 0;
}