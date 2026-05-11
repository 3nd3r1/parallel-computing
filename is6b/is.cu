#include <cstdlib>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

using namespace std;

struct Result {
    int y0;
    int x0;
    int y1;
    int x1;
    float outer[3];
    float inner[3];
};

static inline void check(cudaError_t err, const char *context) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << context << ": "
                  << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

static inline int divup(int a, int b) { return (a + b - 1) / b; }

static inline int roundup(int a, int b) { return divup(a, b) * b; }

#define CHECK(x) check(x, #x)

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
                (int)data[3 * x + 3 * nx * y] + pref_s[(x + 1) + nxp * y] +
                pref_s[x + nxp * (y + 1)] - pref_s[x + nxp * y];
        }
    }

    Result global_result = Result{0, 0, 0, 0, {0, 0, 0}, {0, 0, 0}};
    float global_min_tsse = 1e9f;

    int total_s = pref_s[nx + nxp * ny];

    for (int y0 = 0; y0 < ny; y0++) {
        float min_tsse = 1e9f;
        Result result = Result{0, 0, 0, 0, {0, 0, 0}, {0, 0, 0}};

        for (int y1 = y0 + 1; y1 <= ny; y1++) {
            int iy0 = nxp * y0;
            int iy1 = nxp * y1;
            for (int x0 = 0; x0 < nx; x0++) {
                for (int x1 = x0 + 1; x1 <= nx; x1++) {
                    int in_n = (y1 - y0) * (x1 - x0);
                    int out_n = (nx * ny) - in_n;

                    float inv_i = 1.0 / in_n;
                    float inv_o = (out_n > 0.0) ? 1.0 / out_n : 0.0;

                    int inside_sum = pref_s[x1 + iy1] - pref_s[x1 + iy0] -
                                     pref_s[x0 + iy1] + pref_s[x0 + iy0];
                    int outside_sum = total_s - inside_sum;

                    float inside_sse = inside_sum * (1 - inside_sum * inv_i);
                    float outside_sse = outside_sum * (1 - outside_sum * inv_o);

                    float tsse = inside_sse + outside_sse;

                    if (tsse < min_tsse) {
                        min_tsse = tsse;
                        result =
                            Result{y0,
                                   x0,
                                   y1,
                                   x1,
                                   {outside_sum * inv_o, outside_sum * inv_o,
                                    outside_sum * inv_o},
                                   {inside_sum * inv_i, inside_sum * inv_i,
                                    inside_sum * inv_i}};
                    }
                }
            }
        }

        if (min_tsse < global_min_tsse) {
            global_min_tsse = min_tsse;
            global_result = result;
        }
    }

    return global_result;
}
