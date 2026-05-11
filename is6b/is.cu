#include <cstdlib>
#include <cuda_runtime.h>
#include <iostream>

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

// static inline int divup(int a, int b) { return (a + b - 1) / b; }

// static inline int roundup(int a, int b) { return divup(a, b) * b; }

#define CHECK(x) check(x, #x)

__device__ float min_tsseGPU = INFINITY;
__device__ Result best_resultGPU;
__device__ int lockGPU = 0;

__global__ void kernel(int *pref_s, int nx, int ny) {
    int nxp = nx + 1;
    int nyp = ny + 1;

    int total = nx * ny * nxp * nyp;

    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int step = blockDim.x * gridDim.x;

    for (int i = gid; i < total; i += step) {
        int idx = i;
        int x0 = idx % nx;
        idx /= nx;
        int y0 = idx % ny;
        idx /= ny;
        int x1 = idx % (nx + 1);
        idx /= (nx + 1);
        int y1 = idx;

        if (x1 <= x0 || y1 <= y0)
            continue;

        int in_n = (y1 - y0) * (x1 - x0);
        int out_n = (nx * ny) - in_n;

        float inv_i = 1.0 / in_n;
        float inv_o = (out_n > 0.0) ? 1.0 / out_n : 0.0;

        int inside_sum = pref_s[x1 + nxp * y1] - pref_s[x1 + nxp * y0] -
                         pref_s[x0 + nxp * y1] + pref_s[x0 + nxp * y0];
        int outside_sum = pref_s[nx + nxp * ny] - inside_sum;

        float inside_sse = inside_sum * (1 - inside_sum * inv_i);
        float outside_sse = outside_sum * (1 - outside_sum * inv_o);

        float tsse = inside_sse + outside_sse;

        if (tsse >= min_tsseGPU)
            continue;

        while (atomicCAS(&lockGPU, 0, 1) != 0) {
            __syncwarp();
        };
        __threadfence();
        if (tsse < min_tsseGPU) {
            min_tsseGPU = tsse;
            best_resultGPU = Result{
                y0,
                x0,
                y1,
                x1,
                {outside_sum * inv_o, outside_sum * inv_o, outside_sum * inv_o},
                {inside_sum * inv_i, inside_sum * inv_i, inside_sum * inv_i},
            };
        }
        __threadfence();
        atomicExch(&lockGPU, 0);
    }
}

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

    int *pref_s = new int[nyp * nxp]();

    for (int y = 0; y < ny; y++) {
        for (int x = 0; x < nx; x++) {
            pref_s[(x + 1) + nxp * (y + 1)] =
                (int)data[3 * x + 3 * nx * y] + pref_s[(x + 1) + nxp * y] +
                pref_s[x + nxp * (y + 1)] - pref_s[x + nxp * y];
        }
    }

    float inf = INFINITY;
    CHECK(cudaMemcpyToSymbol(min_tsseGPU, &inf, sizeof(float)));

    int *pref_sGPU = NULL;
    CHECK(cudaMalloc((void **)&pref_sGPU, nxp * nyp * sizeof(int)));
    CHECK(cudaMemcpy(pref_sGPU, pref_s, nxp * nyp * sizeof(int),
                     cudaMemcpyHostToDevice));

    {
        kernel<<<4096, 256>>>(pref_sGPU, nx, ny);
        CHECK(cudaGetLastError());
    }

    Result best_result;
    CHECK(cudaMemcpyFromSymbol(&best_result, best_resultGPU, sizeof(Result)));

    CHECK(cudaFree(pref_sGPU));

    return best_result;
}
