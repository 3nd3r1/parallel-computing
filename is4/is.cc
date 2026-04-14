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
        ny + 1, vector<vector<double>>(nx + 1, std::vector<double>(3, 0.0)));
    vector<vector<vector<double>>> pref_ss(
        ny + 1, vector<vector<double>>(nx + 1, std::vector<double>(3, 0.0)));

    for (int y = 0; y < ny; y++) {
        for (int x = 0; x < nx; x++) {
            for (int c = 0; c < 3; c++) {
                pref_s[y + 1][x + 1][c] = data[c + 3 * x + 3 * nx * y] +
                                          pref_s[y][x + 1][c] +
                                          pref_s[y + 1][x][c] - pref_s[y][x][c];
                pref_ss[y + 1][x + 1][c] = (data[c + 3 * x + 3 * nx * y] *
                                            data[c + 3 * x + 3 * nx * y]) +
                                           pref_ss[y][x + 1][c] +
                                           pref_ss[y + 1][x][c] -
                                           pref_ss[y][x][c];
            }
        }
    }

    Result result = Result{0, 0, 0, 0, {0, 0, 0}, {0, 0, 0}};
    double min_tsse = 1e9;

    for (int y0 = 0; y0 < ny; y0++) {
        for (int x0 = 0; x0 < nx; x0++) {
            for (int y1 = y0 + 1; y1 <= ny; y1++) {
                for (int x1 = x0 + 1; x1 <= nx; x1++) {
                    double tsse = 0;
                    float outer[3] = {0, 0, 0};
                    float inner[3] = {0, 0, 0};

                    double inside_n = (double)(y1 - y0) * (x1 - x0);
                    double outside_n = (double)(nx * ny) - inside_n;

                    if (outside_n <= 0)
                        continue;

                    for (int c = 0; c < 3; c++) {
                        double inside_sum =
                            pref_s[y1][x1][c] - pref_s[y0][x1][c] -
                            pref_s[y1][x0][c] + pref_s[y0][x0][c];
                        double inside_sum_sq =
                            pref_ss[y1][x1][c] - pref_ss[y0][x1][c] -
                            pref_ss[y1][x0][c] + pref_ss[y0][x0][c];

                        double outside_sum = pref_s[ny][nx][c] - inside_sum;
                        double outside_sum_sq =
                            pref_ss[ny][nx][c] - inside_sum_sq;

                        inner[c] = (float)(inside_sum / inside_n);
                        outer[c] = (float)(outside_sum / outside_n);

                        double inside_sse =
                            inside_sum_sq -
                            ((inside_sum * inside_sum) / inside_n);
                        double outside_sse =
                            outside_sum_sq -
                            ((outside_sum * outside_sum) / outside_n);

                        tsse += inside_sse + outside_sse;
                    }
                    if (tsse < min_tsse) {
                        min_tsse = tsse;
                        result = Result{y0,
                                        x0,
                                        y1,
                                        x1,
                                        {outer[0], outer[1], outer[2]},
                                        {inner[0], inner[1], inner[2]}};
                    }
                }
            }
        }
    }

    return result;
}
