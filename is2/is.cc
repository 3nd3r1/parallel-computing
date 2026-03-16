#include <cmath>
#include <vector>

using std::vector;

struct Result {
    int y0;
    int x0;
    int y1;
    int x1;
    float outer[3];
    float inner[3];
};

/*
This is the function you need to implement. Quick reference:
- x coordinates: 0 <= x < nx
- y coordinates: 0 <= y < ny
- color components: 0 <= c < 3
- input: data[c + 3 * x + 3 * nx * y]
*/
Result segment(int ny, int nx, const float *data) {
    vector<vector<vector<double>>> pref_s(
        ny, vector<vector<double>>(nx, std::vector<double>(3, 0.0)));
    vector<vector<vector<double>>> pref_ss(
        ny, vector<vector<double>>(nx, std::vector<double>(3, 0.0)));

    for (int y = 0; y < ny; y++) {
        for (int x = 0; x < nx; x++) {
            for (int c = 0; c < 3; c++) {
                pref_s[y][x][c] = data[c + 3 * x + 3 * nx * y];
                pref_ss[y][x][c] =
                    data[c + 3 * x + 3 * nx * y] * data[c + 3 * x + 3 * nx * y];

                if (y - 1 >= 0) {
                    pref_s[y][x][c] += pref_s[y - 1][x][c];
                    pref_ss[y][x][c] += pref_ss[y - 1][x][c];
                }
                if (x - 1 >= 0) {
                    pref_s[y][x][c] += pref_s[y][x - 1][c];
                    pref_ss[y][x][c] += pref_ss[y][x - 1][c];
                }
                if (y - 1 >= 0 && x - 1 >= 0) {
                    pref_s[y][x][c] += pref_s[y - 1][x - 1][c];
                    pref_ss[y][x][c] += pref_ss[y - 1][x - 1][c];
                }
            }
        }
    }

    Result result = Result { 0, 0, 0, 0, {0, 0, 0}, {0, 0, 0} };
    for (int y0 = 0; y0 < ny; y0++) {
        for (int x0 = 0; x0 < nx; x0++) {
            for (int y1 = y0; y1 < ny; y1++) {
                for (int x1 = x0; x1 < nx; x1++) {
                    double tsse = 0;
                    for (int c = 0; c < 3; c++) {
                        double sum = pref_s[y1][x1][c] - pref_s[y0 - 1][x1][c] -
                                     pref_s[y1][x0 - 1][c] +
                                     pref_s[y0 - 1][x0 - 1][c];
                        double sum_sq =
                            pref_ss[y1][x1][c] - pref_ss[y0 - 1][x1][c] -
                            pref_ss[y1][x0 - 1][c] + pref_ss[y0 - 1][x0 - 1][c];

                        double sse =
                            sum_sq - ((sum * sum) / ((y1 - y0) * (x1 - x0)));

                        tsse += sse;
                    }
                }
            }
        }
    }

    return result;
}
