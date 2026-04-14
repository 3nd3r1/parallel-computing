#include <omp.h>
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

typedef double double4_t __attribute__((vector_size(4 * sizeof(double))));

/*
This is the function you need to implement. Quick reference:
- x coordinates: 0 <= x < nx
- y coordinates: 0 <= y < ny
- color components: 0 <= c < 3
- input: data[c + 3 * x + 3 * nx * y]
*/
Result segment(int ny, int nx, const float *data) {
    int nxp = nx + 1;
    int nyp = ny + 1;

    vector<double4_t> pref_s(nyp * nxp, double4_t{0, 0, 0, 0});
    vector<double4_t> pref_ss(nyp * nxp, double4_t{0, 0, 0, 0});

    for (int y = 0; y < ny; y++) {
        for (int x = 0; x < nx; x++) {
            for (int c = 0; c < 3; c++) {
                pref_s[(x + 1) + nxp * (y + 1)][c] =
                    data[c + 3 * x + 3 * nx * y] +
                    pref_s[(x + 1) + nxp * y][c] +
                    pref_s[x + nxp * (y + 1)][c] - pref_s[x + nxp * y][c];
                pref_ss[(x + 1) + nxp * (y + 1)][c] =
                    (data[c + 3 * x + 3 * nx * y] *
                     data[c + 3 * x + 3 * nx * y]) +
                    pref_ss[(x + 1) + nxp * y][c] +
                    pref_ss[x + nxp * (y + 1)][c] - pref_ss[x + nxp * y][c];
            }
        }
    }

    Result global_result = Result{0, 0, 0, 0, {0, 0, 0}, {0, 0, 0}};
    double global_min_tsse = 1e9;

    double4_t total_s = pref_s[nx + nxp * ny];
    double4_t total_ss = pref_ss[nx + nxp * ny];

#pragma omp parallel for schedule(dynamic, 1)
    for (int y0 = 0; y0 < ny; y0++) {
        double min_tsse = 1e9;
        Result result = Result{0, 0, 0, 0, {0, 0, 0}, {0, 0, 0}};
        vector<double4_t> diff_s(nxp);
        vector<double4_t> diff_ss(nxp);
        for (int y1 = y0 + 1; y1 <= ny; y1++) {
            int iy0 = nxp * y0;
            int iy1 = nxp * y1;
            for (int x = 0; x <= nx; x++) {
                diff_s[x] = pref_s[x + iy1] - pref_s[x + iy0];
                diff_ss[x] = pref_ss[x + iy1] - pref_ss[x + iy0];
            }
            for (int x0 = 0; x0 < nx; x0++) {
                double4_t d_x0 = diff_s[x0];
                double4_t dd_x0 = diff_ss[x0];
                for (int x1 = x0 + 1; x1 <= nx; x1++) {
                    double inside_n = (double)(y1 - y0) * (x1 - x0);
                    double outside_n = (double)(nx * ny) - inside_n;

                    if (outside_n <= 0)
                        continue;

                    double4_t inside_n4 = {inside_n, inside_n, inside_n,
                                           inside_n};
                    double4_t outside_n4 = {outside_n, outside_n, outside_n,
                                            outside_n};

                    double4_t inside_sum = diff_s[x1] - d_x0;
                    double4_t inside_sum_sq = diff_ss[x1] - dd_x0;

                    double4_t outside_sum = total_s - inside_sum;
                    double4_t outside_sum_sq = total_ss - inside_sum_sq;

                    double4_t inner = inside_sum / inside_n;
                    double4_t outer = outside_sum / outside_n;

                    double4_t inside_sse =
                        inside_sum_sq - (inside_sum * inside_sum) / inside_n4;
                    double4_t outside_sse =
                        outside_sum_sq -
                        (outside_sum * outside_sum) / outside_n4;

                    double tsse = inside_sse[0] + inside_sse[1] +
                                  inside_sse[2] + outside_sse[0] +
                                  outside_sse[1] + outside_sse[2];

                    if (tsse > global_min_tsse)
                        continue;

                    if (tsse < min_tsse) {
                        min_tsse = tsse;
                        result = Result{
                            y0,
                            x0,
                            y1,
                            x1,
                            {(float)outer[0], (float)outer[1], (float)outer[2]},
                            {(float)inner[0], (float)inner[1],
                             (float)inner[2]}};
                    }
                }
            }
        }
#pragma omp critical
        {
            if (min_tsse < global_min_tsse) {
                global_min_tsse = min_tsse;
                global_result = result;
            }
        }
    }

    return global_result;
}
