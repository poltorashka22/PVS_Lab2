#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>
#include <time.h>
#include <unistd.h>
#include <getopt.h>

#define EPSILON 1e-9

typedef struct {
    int rows;
    int cols;
    double** data;
} Matrix;

Matrix create_matrix(int rows, int cols) {
    Matrix mat;
    mat.rows = rows;
    mat.cols = cols;
    mat.data = (double**)malloc(rows * sizeof(double*));
    for (int i = 0; i < rows; i++) {
        mat.data[i] = (double*)malloc(cols * sizeof(double));
    }
    return mat;
}

void free_matrix(Matrix mat) {
    for (int i = 0; i < mat.rows; i++) {
        free(mat.data[i]);
    }
    free(mat.data);
}

void fill_matrix_random(Matrix mat) {
#pragma omp parallel for collapse(2)
    for (int i = 0; i < mat.rows; i++) {
        for (int j = 0; j < mat.cols; j++) {
            mat.data[i][j] = (double)rand() / RAND_MAX * 100.0 + 1.0;
        }
    }
}

void matrix_operations(Matrix a, Matrix b, Matrix add, Matrix sub, Matrix mul, Matrix div, int parallel) {
    if (parallel) {
#pragma omp parallel for collapse(2)
        for (int i = 0; i < a.rows; i++) {
            for (int j = 0; j < a.cols; j++) {
                add.data[i][j] = a.data[i][j] + b.data[i][j];
                sub.data[i][j] = a.data[i][j] - b.data[i][j];
                mul.data[i][j] = a.data[i][j] * b.data[i][j];
                div.data[i][j] = (fabs(b.data[i][j]) > EPSILON) ? a.data[i][j] / b.data[i][j] : 0.0;
            }
        }
    }
    else {
        for (int i = 0; i < a.rows; i++) {
            for (int j = 0; j < a.cols; j++) {
                add.data[i][j] = a.data[i][j] + b.data[i][j];
                sub.data[i][j] = a.data[i][j] - b.data[i][j];
                mul.data[i][j] = a.data[i][j] * b.data[i][j];
                div.data[i][j] = (fabs(b.data[i][j]) > EPSILON) ? a.data[i][j] / b.data[i][j] : 0.0;
            }
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
        case 'n':
            total_elements = atoi(optarg);
            if (total_elements <= 100000) {
                fprintf(stderr, "Error: Total number of elements must be greater than 100000\n");
                return 1;
            }
            break;
        case 'r':
            rows = atoi(optarg);
            break;
        case 'c':
            cols = atoi(optarg);
            break;
        case 'h':
            print_usage(argv[0]);
            return 0;
        default:
            print_usage(argv[0]);
            return 1;
        }
    }

    if (total_elements == 0) {
        fprintf(stderr, "Error: Total number of elements (-n) is required\n");
        print_usage(argv[0]);
        return 1;
    }

    if (rows == 0 || cols == 0) {
        rows = (int)sqrt(total_elements);
        cols = total_elements / rows;
        while (rows * cols < total_elements) {
            cols++;
        }
    }
    else if (rows * cols < total_elements) {
        fprintf(stderr, "Error: rows * cols must be >= total number of elements\n");
        return 1;
    }

    Matrix A = create_matrix(rows, cols);
    Matrix B = create_matrix(rows, cols);
    Matrix ADD = create_matrix(rows, cols);
    Matrix SUB = create_matrix(rows, cols);
    Matrix MUL = create_matrix(rows, cols);
    Matrix DIV = create_matrix(rows, cols);

    srand(time(NULL));

    fill_matrix_random(A);
    fill_matrix_random(B);

    double seq_start = omp_get_wtime();
    matrix_operations(A, B, ADD, SUB, MUL, DIV, 0);
    double seq_time = omp_get_wtime() - seq_start;

    double par_start = omp_get_wtime();
    matrix_operations(A, B, ADD, SUB, MUL, DIV, 1);
    double par_time = omp_get_wtime() - par_start;

    printf("Последовательное время: %.5f секунд\n", seq_time);
    printf("Параллельное время: %.5f секунд\n", par_time);

    // Освобождение памяти
    free_matrix(A);
    free_matrix(B);
    free_matrix(ADD);
    free_matrix(SUB);
    free_matrix(MUL);
    free_matrix(DIV);

    return 0;
}