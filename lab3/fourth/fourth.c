#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <unistd.h>
#include <getopt.h>
#include <mpi.h>

#define EPSILON 1e-9

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
        perror("Matrix allocation failed");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    return mat;
}

void free_linear_matrix(Matrix mat) {
    free(mat.data);
}

void fill_matrix_random(Matrix mat) {
    for (int i = 0; i < mat.rows; i++) {
        for (int j = 0; j < mat.cols; j++) {
            mat.data[i * mat.cols + j] = (double)rand() / RAND_MAX * 100.0 + 1.0;
        }
    }
}

void matrix_operations(Matrix a, Matrix b, Matrix add, Matrix sub, Matrix mul, Matrix div) {
    int rows = a.rows;
    int cols = a.cols;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            int index = i * cols + j;
            add.data[index] = a.data[index] + b.data[index];
            sub.data[index] = a.data[index] - b.data[index];
            mul.data[index] = a.data[index] * b.data[index];
            div.data[index] = (fabs(b.data[index]) > EPSILON) ? a.data[index] / b.data[index] : 0.0;
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
    MPI_Init(&argc, &argv);
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    int total_elements = 0;
    int rows = 0, cols = 0;

    if (rank == 0) {
        int opt;
        while ((opt = getopt(argc, argv, "n:r:c:h")) != -1) {
            switch (opt) {
            case 'n': total_elements = atoi(optarg); break;
            case 'r': rows = atoi(optarg); break;
            case 'c': cols = atoi(optarg); break;
            case 'h': print_usage(argv[0]); MPI_Abort(MPI_COMM_WORLD, 0); break;
            default: print_usage(argv[0]); MPI_Abort(MPI_COMM_WORLD, 1); break;
            }
        }

        if (total_elements <= 100000) {
            fprintf(stderr, "Error: Total elements must be > 100000\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        if (rows == 0 || cols == 0) {
            rows = (int)sqrt(total_elements);
            cols = total_elements / rows;
            while (rows * cols < total_elements) rows++; 
        }
        else if (rows * cols < total_elements) {
            fprintf(stderr, "Error: rows * cols must be >= total number of elements\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        if (rows % num_procs != 0) {
            fprintf(stderr, "Error: Number of rows (%d) must be divisible by number of MPI processes (%d)\n", rows, num_procs);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&cols, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int local_rows = rows / num_procs;
    int local_size = local_rows * cols;

    Matrix A, B, ADD_seq, SUB_seq, MUL_seq, DIV_seq;
    Matrix ADD_par, SUB_par, MUL_par, DIV_par;

    if (rank == 0) {
        A = create_linear_matrix(rows, cols);
        B = create_linear_matrix(rows, cols);
        ADD_seq = create_linear_matrix(rows, cols);
        SUB_seq = create_linear_matrix(rows, cols);
        MUL_seq = create_linear_matrix(rows, cols);
        DIV_seq = create_linear_matrix(rows, cols);

        ADD_par = create_linear_matrix(rows, cols);
        SUB_par = create_linear_matrix(rows, cols);
        MUL_par = create_linear_matrix(rows, cols);
        DIV_par = create_linear_matrix(rows, cols);

        srand(time(NULL));
        fill_matrix_random(A);
        fill_matrix_random(B);

        double seq_start = MPI_Wtime();
        matrix_operations(A, B, ADD_seq, SUB_seq, MUL_seq, DIV_seq);
        double seq_time = MPI_Wtime() - seq_start;
        printf("Sequential time: %.5f s\n", seq_time);
    }

    Matrix local_A = create_linear_matrix(local_rows, cols);
    Matrix local_B = create_linear_matrix(local_rows, cols);
    Matrix local_ADD = create_linear_matrix(local_rows, cols);
    Matrix local_SUB = create_linear_matrix(local_rows, cols);
    Matrix local_MUL = create_linear_matrix(local_rows, cols);
    Matrix local_DIV = create_linear_matrix(local_rows, cols);

    double par_start = MPI_Wtime();

    MPI_Scatter(A.data, local_size, MPI_DOUBLE, local_A.data, local_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatter(B.data, local_size, MPI_DOUBLE, local_B.data, local_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    matrix_operations(local_A, local_B, local_ADD, local_SUB, local_MUL, local_DIV);

    MPI_Gather(local_ADD.data, local_size, MPI_DOUBLE, ADD_par.data, local_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gather(local_SUB.data, local_size, MPI_DOUBLE, SUB_par.data, local_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gather(local_MUL.data, local_size, MPI_DOUBLE, MUL_par.data, local_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gather(local_DIV.data, local_size, MPI_DOUBLE, DIV_par.data, local_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    double par_time = MPI_Wtime() - par_start;

    if (rank == 0) {
        printf("Parallel time:   %.5f s\n", par_time);
    }

    free_linear_matrix(local_A);
    free_linear_matrix(local_B);
    free_linear_matrix(local_ADD);
    free_linear_matrix(local_SUB);
    free_linear_matrix(local_MUL);
    free_linear_matrix(local_DIV);

    if (rank == 0) {
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
    }

    MPI_Finalize();
    return 0;
}