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
    vector<double> pref_s((ny + 1) * (nx + 1) * 3, 0.0);
    vector<double> pref_ss((ny + 1) * (nx + 1) * 3, 0.0);

    int stride = 3 * (nx + 1);
    for (int y = 0; y < ny; y++) {
        for (int x = 0; x < nx; x++) {
            for (int c = 0; c < 3; c++) {
                pref_s[c + 3 * (x + 1) + stride * (y + 1)] =
                    data[c + 3 * x + 3 * nx * y] +
                    pref_s[c + 3 * (x + 1) + stride * y] +
                    pref_s[c + 3 * x + stride * (y + 1)] -
                    pref_s[c + 3 * x + stride * y];
                pref_ss[c + 3 * (x + 1) + stride * (y + 1)] =
                    (data[c + 3 * x + 3 * nx * y] *
                     data[c + 3 * x + 3 * nx * y]) +
                    pref_ss[c + 3 * (x + 1) + stride * y] +
                    pref_ss[c + 3 * x + stride * (y + 1)] -
                    pref_ss[c + 3 * x + stride * y];
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
                        double inside_sum = pref_s[c + 3 * x1 + stride * y1] -
                                            pref_s[c + 3 * x1 + stride * y0] -
                                            pref_s[c + 3 * x0 + stride * y1] +
                                            pref_s[c + 3 * x0 + stride * y0];
                        double inside_sum_sq =
                            pref_ss[c + 3 * x1 + stride * y1] -
                            pref_ss[c + 3 * x1 + stride * y0] -
                            pref_ss[c + 3 * x0 + stride * y1] +
                            pref_ss[c + 3 * x0 + stride * y0];

                        double outside_sum =
                            pref_s[c + 3 * nx + stride * ny] - inside_sum;
                        double outside_sum_sq =
                            pref_ss[c + 3 * nx + stride * ny] - inside_sum_sq;

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
