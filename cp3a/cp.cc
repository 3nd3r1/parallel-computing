#include <cmath>
#include <omp.h>
#include <vector>

using std::vector;

typedef double double4_t __attribute__((vector_size(4 * sizeof(double))));

/*
This is the function you need to implement. Quick reference:
- input rows: 0 <= y < ny
- input columns: 0 <= x < nx
- element at row y and column x is stored in data[x + y*nx]
- correlation between rows i and row j has to be stored in result[i + j*ny]
- only parts with 0 <= j <= i < ny need to be filled
*/
void correlate(int ny, int nx, const float *data, float *result) {
    constexpr int vec_size = 4;   // Number of elements per vector
    constexpr int block_size = 8; // Block size
    int nx_vecs = (nx + vec_size - 1) / vec_size; // Number of vectors per row
    int n_blocks = (ny + block_size - 1) / block_size; // Number of blocks
    int padded_ny = n_blocks * block_size;             // padded ny

    vector<double4_t> matrix(padded_ny * nx_vecs, double4_t{0, 0, 0, 0});

#pragma omp parallel for
    for (int y = 0; y < ny; y++) {
        double4_t sum = {0, 0, 0, 0};
        for (int x = 0; x < nx / vec_size; x++) {
            double4_t drow = {data[x * vec_size + 0 + y * nx],
                              data[x * vec_size + 1 + y * nx],
                              data[x * vec_size + 2 + y * nx],
                              data[x * vec_size + 3 + y * nx]};
            sum += drow;
            matrix[x + y * nx_vecs] = drow;
        }
        for (int k = 0; k < nx % vec_size; k++) {
            sum[0] += data[(nx / vec_size) * vec_size + k + y * nx];
            matrix[(nx / vec_size) + y * nx_vecs][k] =
                data[(nx / vec_size) * vec_size + k + y * nx];
        }
        double avg = (sum[0] + sum[1] + sum[2] + sum[3]) / nx;
        double4_t avg4 = {avg, avg, avg, avg};

        double4_t sum_sq = {0, 0, 0, 0};
        for (int x = 0; x < nx / vec_size; x++) {
            matrix[x + y * nx_vecs] -= avg4;
            sum_sq += matrix[x + y * nx_vecs] * matrix[x + y * nx_vecs];
        }
        for (int k = 0; k < nx % vec_size; k++) {
            matrix[(nx / vec_size) + y * nx_vecs][k] -= avg;
            sum_sq[0] += matrix[(nx / vec_size) + y * nx_vecs][k] *
                         matrix[(nx / vec_size) + y * nx_vecs][k];
        }
        double mag = std::sqrt(sum_sq[0] + sum_sq[1] + sum_sq[2] + sum_sq[3]);
        double4_t mag4 = {mag, mag, mag, mag};

        for (int x = 0; x < nx / vec_size; x++) {
            matrix[x + y * nx_vecs] /= mag4;
        }
        for (int k = 0; k < nx % vec_size; k++) {
            matrix[(nx / vec_size) + y * nx_vecs][k] /= mag;
        }
    }

#pragma omp parallel for schedule(dynamic, 1)
    for (int i = 0; i < n_blocks; i++) {
        for (int j = i; j < n_blocks; j++) {
            double4_t val[block_size][block_size] = {};

            for (int k = 0; k < nx_vecs; k++) {
                double4_t x0 = matrix[k + (i * block_size + 0) * nx_vecs];
                double4_t x1 = matrix[k + (i * block_size + 1) * nx_vecs];
                double4_t x2 = matrix[k + (i * block_size + 2) * nx_vecs];
                double4_t x3 = matrix[k + (i * block_size + 3) * nx_vecs];
                double4_t x4 = matrix[k + (i * block_size + 4) * nx_vecs];
                double4_t x5 = matrix[k + (i * block_size + 5) * nx_vecs];
                double4_t x6 = matrix[k + (i * block_size + 6) * nx_vecs];
                double4_t x7 = matrix[k + (i * block_size + 7) * nx_vecs];
                double4_t y0 = matrix[k + (j * block_size + 0) * nx_vecs];
                double4_t y1 = matrix[k + (j * block_size + 1) * nx_vecs];
                double4_t y2 = matrix[k + (j * block_size + 2) * nx_vecs];
                double4_t y3 = matrix[k + (j * block_size + 3) * nx_vecs];
                double4_t y4 = matrix[k + (j * block_size + 4) * nx_vecs];
                double4_t y5 = matrix[k + (j * block_size + 5) * nx_vecs];
                double4_t y6 = matrix[k + (j * block_size + 6) * nx_vecs];
                double4_t y7 = matrix[k + (j * block_size + 7) * nx_vecs];

                val[0][0] += x0 * y0;
                val[0][1] += x0 * y1;
                val[0][2] += x0 * y2;
                val[0][3] += x0 * y3;
                val[0][4] += x0 * y4;
                val[0][5] += x0 * y5;
                val[0][6] += x0 * y6;
                val[0][7] += x0 * y7;

                val[1][0] += x1 * y0;
                val[1][1] += x1 * y1;
                val[1][2] += x1 * y2;
                val[1][3] += x1 * y3;
                val[1][4] += x1 * y4;
                val[1][5] += x1 * y5;
                val[1][6] += x1 * y6;
                val[1][7] += x1 * y7;

                val[2][0] += x2 * y0;
                val[2][1] += x2 * y1;
                val[2][2] += x2 * y2;
                val[2][3] += x2 * y3;
                val[2][4] += x2 * y4;
                val[2][5] += x2 * y5;
                val[2][6] += x2 * y6;
                val[2][7] += x2 * y7;

                val[3][0] += x3 * y0;
                val[3][1] += x3 * y1;
                val[3][2] += x3 * y2;
                val[3][3] += x3 * y3;
                val[3][4] += x3 * y4;
                val[3][5] += x3 * y5;
                val[3][6] += x3 * y6;
                val[3][7] += x3 * y7;

                val[4][0] += x4 * y0;
                val[4][1] += x4 * y1;
                val[4][2] += x4 * y2;
                val[4][3] += x4 * y3;
                val[4][4] += x4 * y4;
                val[4][5] += x4 * y5;
                val[4][6] += x4 * y6;
                val[4][7] += x4 * y7;

                val[5][0] += x5 * y0;
                val[5][1] += x5 * y1;
                val[5][2] += x5 * y2;
                val[5][3] += x5 * y3;
                val[5][4] += x5 * y4;
                val[5][5] += x5 * y5;
                val[5][6] += x5 * y6;
                val[5][7] += x5 * y7;

                val[6][0] += x6 * y0;
                val[6][1] += x6 * y1;
                val[6][2] += x6 * y2;
                val[6][3] += x6 * y3;
                val[6][4] += x6 * y4;
                val[6][5] += x6 * y5;
                val[6][6] += x6 * y6;
                val[6][7] += x6 * y7;

                val[7][0] += x7 * y0;
                val[7][1] += x7 * y1;
                val[7][2] += x7 * y2;
                val[7][3] += x7 * y3;
                val[7][4] += x7 * y4;
                val[7][5] += x7 * y5;
                val[7][6] += x7 * y6;
                val[7][7] += x7 * y7;
            }

            for (int bi = 0; bi < block_size; bi++) {
                for (int bj = 0; bj < block_size; bj++) {
                    int ri = i * block_size + bi;
                    int rj = j * block_size + bj;
                    if (ri < ny && rj < ny) {
                        result[rj + ri * ny] =
                            (float)((val[bi][bj][0] + val[bi][bj][1]) +
                                    (val[bi][bj][2] + val[bi][bj][3]));
                    }
                }
            }
        }
    }
}
