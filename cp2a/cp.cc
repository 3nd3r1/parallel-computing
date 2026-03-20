#include <cmath>
#include <vector>

using std::vector;

/*
This is the function you need to implement. Quick reference:
- input rows: 0 <= y < ny
- input columns: 0 <= x < nx
- element at row y and column x is stored in data[x + y*nx]
- correlation between rows i and row j has to be stored in result[i + j*ny]
- only parts with 0 <= j <= i < ny need to be filled
*/
void correlate(int ny, int nx, const float *data, float *result) {
    vector<double> matrix(nx * ny, 0.0);

    for (int y = 0; y < ny; y++) {
        double sum = 0;
        double sum_sq = 0;
        for (int x = 0; x < nx; x++) {
            sum += data[x + y * nx];
            sum_sq += (double)data[x + y * nx] * (double)data[x + y * nx];
        }
        double avg = sum / nx;
        double mag = std::sqrt(sum_sq - nx * avg * avg);
        for (int x = 0; x < nx; x++) {
            matrix[x + y * nx] = (data[x + y * nx] - avg) / mag;
        }
    }

    for (int j = 0; j < ny; j++) {
        for (int i = j; i < ny; i++) {
            double v0 = 0, v1 = 0, v2 = 0, v3 = 0;

            for (int k = 0; k + 3 < nx; k += 4) {
                v0 += matrix[k + i * nx] * matrix[k + j * nx];
                v1 += matrix[k + 1 + i * nx] * matrix[k + 1 + j * nx];
                v2 += matrix[k + 2 + i * nx] * matrix[k + 2 + j * nx];
                v3 += matrix[k + 3 + i * nx] * matrix[k + 3 + j * nx];
            }

            for (int k = nx - (nx % 4); k < nx; k++) {
                v0 += matrix[k + i * nx] * matrix[k + j * nx];
            }

            result[i + j * ny] = (float)((v0 + v1) + (v2 + v3));
        }
    }
}
