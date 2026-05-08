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
    float total_nd = (float)nx * ny;

    vector<int> pref_s(nyp * nxp, 0);
    for (int y = 0; y < ny; y++) {
        for (int x = 0; x < nx; x++) {
            pref_s[(x + 1) + nxp * (y + 1)] =
                (int)data[3 * x + 3 * nx * y] + pref_s[(x + 1) + nxp * y] +
                pref_s[x + nxp * (y + 1)] - pref_s[x + nxp * y];
        }
    }

    vector<float> inv_k(nxp, 0.0f);
    for (int k = 1; k <= nx; k++)
        inv_k[k] = 1.0f / k;

    Result global_result = Result{0, 0, 0, 0, {0, 0, 0}, {0, 0, 0}};
    float global_min_tsse = 1e9f;

    int total_s = pref_s[nx + nxp * ny];
    float total_sf = (float)total_s;

#pragma omp parallel
    {
        vector<int> diff_s(nxp);
        vector<float> inv_in(nxp);
        vector<float> inv_out(nxp);
        vector<float> tsse_arr(nxp);

#pragma omp for schedule(dynamic, 1)
        for (int y0 = 0; y0 < ny; y0++) {
            float min_tsse = 1e9f;
            Result result = Result{0, 0, 0, 0, {0, 0, 0}, {0, 0, 0}};

            for (int y1 = y0 + 1; y1 <= ny; y1++) {
                int iy0 = nxp * y0;
                int iy1 = nxp * y1;
                for (int x = 0; x <= nx; x++)
                    diff_s[x] = pref_s[x + iy1] - pref_s[x + iy0];

                int row_n = y1 - y0;
                float inv_row_n = 1.0f / row_n;
                for (int k = 1; k <= nx; k++)
                    inv_in[k] = inv_k[k] * inv_row_n;
                for (int k = 1; k <= nx; k++) {
                    float out_n = total_nd - (float)(row_n * k);
                    inv_out[k] = (out_n > 0.0f) ? 1.0f / out_n : 0.0f;
                }

                float gmt = global_min_tsse;

                for (int x0 = 0; x0 < nx; x0++) {
                    float d_x0 = (float)diff_s[x0];
                    int limit = nx - x0;

                    for (int k = 1; k <= limit; k++) {
                        float s = (float)diff_s[x0 + k] - d_x0;
                        float os = total_sf - s;
                        tsse_arr[k] =
                            total_sf - s * s * inv_in[k] - os * os * inv_out[k];
                    }

                    for (int k = 1; k <= limit; k++) {
                        float t = tsse_arr[k];
                        if (t >= min_tsse || t > gmt)
                            continue;
                        min_tsse = t;
                        gmt = t;
                        int x1 = x0 + k;
                        float s = (float)diff_s[x1] - d_x0;
                        float os = total_sf - s;
                        float inv_i = inv_in[k];
                        float inv_o = inv_out[k];
                        result = Result{y0,
                                        x0,
                                        y1,
                                        x1,
                                        {os * inv_o, os * inv_o, os * inv_o},
                                        {s * inv_i, s * inv_i, s * inv_i}};
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
    }

    return global_result;
}
