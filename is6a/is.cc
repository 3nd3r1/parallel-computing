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

typedef float float8_t __attribute__((vector_size(8 * sizeof(float))));

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

    vector<int> pref_s(nyp * nxp, 0);

    for (int y = 0; y < ny; y++) {
        for (int x = 0; x < nx; x++) {
            pref_s[(x + 1) + nxp * (y + 1)] =
                data[3 * x + 3 * nx * y] + pref_s[(x + 1) + nxp * y] +
                pref_s[x + nxp * (y + 1)] - pref_s[x + nxp * y];
        }
    }

    Result global_result = Result{0, 0, 0, 0, {0, 0, 0}, {0, 0, 0}};
    double global_min_tsse = 1e9;

    int total_s = pref_s[nx + nxp * ny];

#pragma omp parallel for schedule(dynamic, 1)
    for (int y0 = 0; y0 < ny; y0++) {
        float min_tsse = 1e9;
        Result result = Result{0, 0, 0, 0, {0, 0, 0}, {0, 0, 0}};
        vector<float> diff_s(nxp);
        vector<float> inv_in(nxp);
        vector<float> inv_out(nxp);
        for (int y1 = y0 + 1; y1 <= ny; y1++) {
            int iy0 = nxp * y0;
            int iy1 = nxp * y1;
            for (int x = 0; x <= nx; x++) {
                diff_s[x] = pref_s[x + iy1] - pref_s[x + iy0];
            }
            float row_n = (y1 - y0);
            float total_nd = (nx * ny);
            for (int k = 1; k <= nx; k++) {
                float in_n = row_n * k;
                float out_n = total_nd - in_n;
                inv_in[k] = 1.0 / in_n;
                inv_out[k] = (out_n > 0.0) ? 1.0 / out_n : 0.0;
            }
            for (int x0 = 0; x0 < nx; x0++) {
                float d_x0 = diff_s[x0];
                for (int x1 = x0 + 1; x1 <= nx; x1++) {
                    int k = x1 - x0;
                    float inv_i = inv_in[k];
                    float inv_o = inv_out[k];
                    if (inv_o == 0.0)
                        continue;

                    float inside_sum = diff_s[x1] - d_x0;
                    float outside_sum = total_s - inside_sum;

                    float inside_sse = inside_sum * (1 - inside_sum * inv_i);
                    float outside_sse = outside_sum * (1 - outside_sum * inv_o);

                    float tsse = inside_sse + outside_sse;

                    if (tsse > global_min_tsse)
                        continue;

                    if (tsse < min_tsse) {
                        min_tsse = tsse;
                        result = Result{
                            y0,
                            x0,
                            y1,
                            x1,
                            {(outside_sum * inv_o), (outside_sum * inv_o),
                             (outside_sum * inv_o)},
                            {(inside_sum * inv_i), (inside_sum * inv_i),
                             (inside_sum * inv_i)}};
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
