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
    constexpr int vec_size = 4; // Number of elements per vector
    int nx_vecs = (nx + vec_size - 1) / vec_size; // Number of vectors per row

    constexpr int block_size = 3;                      // Block size
    int n_blocks = (ny + block_size - 1) / block_size; // Number of blocks
    int padded_ny = n_blocks * block_size;             // padded ny

    vector<double4_t> matrix(padded_ny * nx_vecs);

#pragma omp parallel for
    for (int y = 0; y < ny; y++) {
        double4_t sum = {0, 0, 0, 0};
        double4_t sum_sq = {0, 0, 0, 0};

        for (int x = 0; x + 3 < nx; x += 4) {
            double4_t drow = {data[x + y * nx], data[x + 1 + y * nx],
                              data[x + 2 + y * nx], data[x + 3 + y * nx]};

            sum += drow;
            sum_sq += drow * drow;
        }
        for (int x = nx - (nx % 4); x < nx; x++) {
            sum[0] += data[x + y * nx];
            sum_sq[0] += (double)data[x + y * nx] * (double)data[x + y * nx];
        }

        double avg = (sum[0] + sum[1] + sum[2] + sum[3]) / nx;
        double mag = std::sqrt((sum_sq[0] + sum_sq[1] + sum_sq[2] + sum_sq[3]) -
                               nx * avg * avg);

        for (int x = 0; x < nx_vecs; x++) {
            for (int k = 0; k < vec_size; k++) {
                int i = x * vec_size + k;
                matrix[x + y * nx_vecs][k] =
                    i < nx ? (data[i + y * nx] - avg) / mag : 0;
            }
        }
    }

#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < n_blocks; i++) {
        for (int j = i; j < n_blocks; j++) {
            double4_t val[block_size][block_size] = {};

            for (int k = 0; k < nx_vecs; k++) {
                double4_t i0 = matrix[k + nx_vecs * (i * block_size + 0)];
                double4_t i1 = matrix[k + nx_vecs * (i * block_size + 1)];
                double4_t i2 = matrix[k + nx_vecs * (i * block_size + 2)];
                double4_t j0 = matrix[k + nx_vecs * (j * block_size + 0)];
                double4_t j1 = matrix[k + nx_vecs * (j * block_size + 1)];
                double4_t j2 = matrix[k + nx_vecs * (j * block_size + 2)];

                val[0][0] += i0 * j0;
                val[0][1] += i0 * j1;
                val[0][2] += i0 * j2;
                val[1][0] += i1 * j0;
                val[1][1] += i1 * j1;
                val[1][2] += i1 * j2;
                val[2][0] += i2 * j0;
                val[2][1] += i2 * j1;
                val[2][2] += i2 * j2;
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
