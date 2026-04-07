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
    constexpr int k_chunk_size = 10; // Number of vectors per k-chunk
    int padded_nx_vecs = ((nx_vecs + k_chunk_size - 1) / k_chunk_size) *
                         k_chunk_size; // Padded number of vectors per row

    constexpr int block_size = 3;                      // Block size
    int n_blocks = (ny + block_size - 1) / block_size; // Number of blocks
    int padded_ny = n_blocks * block_size;             // padded ny

    vector<double4_t> matrix(padded_ny * padded_nx_vecs, double4_t{0, 0, 0, 0});

#pragma omp parallel for
    for (int y = 0; y < ny; y++) {
        double4_t sum = {0, 0, 0, 0};
        double4_t sum_sq = {0, 0, 0, 0};

        for (int x = 0; x < nx / vec_size; x++) {
            double4_t drow = {data[x * vec_size + 0 + y * nx],
                              data[x * vec_size + 1 + y * nx],
                              data[x * vec_size + 2 + y * nx],
                              data[x * vec_size + 3 + y * nx]};
            sum += drow;
            sum_sq += drow * drow;
            matrix[x + y * padded_nx_vecs] = drow;
        }
        for (int k = 0; k < nx % vec_size; k++) {
            sum[0] += data[(nx / vec_size) * vec_size + k + y * nx];
            sum_sq[0] += (double)data[(nx / vec_size) * vec_size + k + y * nx] *
                         (double)data[(nx / vec_size) * vec_size + k + y * nx];
            matrix[(nx / vec_size) + y * padded_nx_vecs][k] =
                (data[(nx / vec_size) * vec_size + k + y * nx]);
        }

        double avg = (sum[0] + sum[1] + sum[2] + sum[3]) / nx;
        double mag = std::sqrt((sum_sq[0] + sum_sq[1] + sum_sq[2] + sum_sq[3]) -
                               nx * avg * avg);

        double4_t avg4 = {avg, avg, avg, avg};
        double4_t mag4 = {mag, mag, mag, mag};

        for (int x = 0; x < nx / vec_size; x++) {
            matrix[x + y * padded_nx_vecs] =
                (matrix[x + y * padded_nx_vecs] - avg4) / mag4;
        }
        for (int k = 0; k < nx % vec_size; k++) {
            matrix[(nx / vec_size) + y * padded_nx_vecs][k] =
                (matrix[(nx / vec_size) + y * padded_nx_vecs][k] - avg) / mag;
        }
    }

    vector<double> res(padded_ny * padded_ny, 0.0);

    for (int k_start = 0; k_start < padded_nx_vecs; k_start += k_chunk_size) {

#pragma omp parallel for schedule(static, 1)
        for (int i = 0; i < n_blocks; i++) {
            double4_t local_i[block_size][k_chunk_size];
            for (int bi = 0; bi < block_size; bi++) {
                for (int kk = 0; kk < k_chunk_size; kk++) {
                    local_i[bi][kk] =
                        matrix[(k_start + kk) +
                               (i * block_size + bi) * padded_nx_vecs];
                }
            }

            for (int j = i; j < n_blocks; j++) {
                double4_t val[block_size][block_size] = {};

                for (int kk = 0; kk < k_chunk_size; kk++) {
                    double4_t i0 = local_i[0][kk];
                    double4_t i1 = local_i[1][kk];
                    double4_t i2 = local_i[2][kk];
                    double4_t j0 =
                        matrix[(k_start + kk) +
                               (j * block_size + 0) * padded_nx_vecs];
                    double4_t j1 =
                        matrix[(k_start + kk) +
                               (j * block_size + 1) * padded_nx_vecs];
                    double4_t j2 =
                        matrix[(k_start + kk) +
                               (j * block_size + 2) * padded_nx_vecs];
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
                        res[ri + rj * padded_ny] +=
                            (val[bi][bj][0] + val[bi][bj][1]) +
                            (val[bi][bj][2] + val[bi][bj][3]);
                    }
                }
            }
        }
    }

    for (int i = 0; i < ny; i++) {
        for (int j = 0; j <= i; j++) {
            result[i + j * ny] = (float)res[j + i * padded_ny];
        }
    }
}
