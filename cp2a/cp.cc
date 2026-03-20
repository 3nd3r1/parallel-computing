#include <cmath>
#include <vector>

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
        double avg = 0;
        for (int x = 0; x < nx; x++) {
            matrix[y][x] = data[x + y * nx];
            avg += matrix[y][x];
        }
        avg /= nx;
        double mag = 0;
        for (int x = 0; x < nx; x++) {
            matrix[y][x] -= avg;
            mag += matrix[y][x] * matrix[y][x];
        }
        mag = std::sqrt(mag);
        for (int x = 0; x < nx; x++) {
            matrix[y][x] /= mag;
        }
    }

    for (int j = 0; j < ny; j++) {
        for (int i = j; i < ny; i++) {
            double val = 0;
            for (int k = 0; k < nx; k++) {
                val += matrix[i][k] * matrix[j][k];
            }
            result[i + j * ny] = (float)val;
        }
    }
}
