#include <cmath>
#include <vector>

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
    std::vector<std::vector<double>> matrix(ny, std::vector<double>(nx, 0.0));
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
            sum_sq[0] += data[x + y * nx] * data[x + y * nx];
        }

        double avg = (sum[0] + sum[1] + sum[2] + sum[3]) / nx;
        double mag = std::sqrt((sum_sq[0] + sum_sq[1] + sum_sq[2] + sum_sq[3]) -
                               nx * avg * avg);

        for (int x = 0; x < nx; x++) {
            matrix[y][x] = (data[x + y * nx] - avg) / mag;
        }
    }

    for (int j = 0; j < ny; j++) {
        for (int i = j; i < ny; i++) {
            double4_t val = {0, 0, 0, 0};
            for (int k = 0; k + 3 < nx; k += 4) {
                double4_t mi, mj;
                mi = {matrix[i][k], matrix[i][k + 1], matrix[i][k + 2],
                      matrix[i][k + 3]};
                mj = {matrix[j][k], matrix[j][k + 1], matrix[j][k + 2],
                      matrix[j][k + 3]};

                val += matrix[i][k] * matrix[j][k];
            }

            for (int k = nx - (nx % 4); k < nx; k++) {
                val[0] += matrix[i][k] * matrix[j][k];
            }

            result[i + j * ny] = (float)(val[0] + val[1] + val[2] + val[3]);
        }
    }
}
