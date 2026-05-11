#include <algorithm>
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

struct BResult {
    int y0;
    int x0;
    int y1;
    int x1;
    float outer;
    float inner;
    float tsse;
};

static inline void check(cudaError_t err, const char *context) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << context << ": "
                  << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

static inline int divup(int a, int b) { return (a + b - 1) / b; }

// static inline int roundup(int a, int b) { return divup(a, b) * b; }

#define CHECK(x) check(x, #x)

const int BLOCK_SIZE = 16;

__global__ void kernel(int *pref_s, BResult *block_tsse, int nx, int ny,
                       int y0) {
    __shared__ BResult sdata[BLOCK_SIZE * BLOCK_SIZE];

    int nxp = nx + 1;
    int nyp = ny + 1;

    int tid = threadIdx.x;
    int gid = blockIdx.x * (BLOCK_SIZE * BLOCK_SIZE) + tid;
    int total = nx * ny * nxp * nyp;

    BResult result;
    if (gid < total) {
        int idx = gid;
        int x0 = idx % nx;
        idx /= nx;
        int x1 = idx % nxp;
        idx /= nxp;
        int y1 = idx;

        if (x1 <= x0 || y1 <= y0) {
            result = BResult{
                y0, x0, y1, x1, 0.0f, 0.0f, INFINITY,
            };
        } else {
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
            result = BResult{
                y0, x0, y1, x1, outside_sum * inv_o, inside_sum * inv_i, tsse,
            };
        }
    } else {
        result = BResult{
            0, 0, 0, 0, 0.0f, 0.0f, INFINITY,
        };
    }

    sdata[tid] = result;
    __syncthreads();

    for (int s = BLOCK_SIZE * BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (sdata[tid + s].tsse < sdata[tid].tsse) {
                sdata[tid] = sdata[tid + s];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        block_tsse[blockIdx.x] = sdata[0];
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

    int num_blocks = divup(nx * nxp * nyp, BLOCK_SIZE * BLOCK_SIZE);

    int *pref_sGPU = NULL;
    CHECK(cudaMalloc((void **)&pref_sGPU, nxp * nyp * sizeof(int)));
    CHECK(cudaMemcpy(pref_sGPU, pref_s, nxp * nyp * sizeof(int),
                     cudaMemcpyHostToDevice));

    BResult *block_tsseGPU = NULL;
    CHECK(cudaMalloc((void **)&block_tsseGPU, num_blocks * sizeof(BResult)));

    BResult global_best = {0, 0, 0, 0, 0, 0, INFINITY};

    for (int y0 = 0; y0 < ny; y0++) {
        kernel<<<num_blocks, BLOCK_SIZE * BLOCK_SIZE>>>(
            pref_sGPU, block_tsseGPU, nx, ny, y0);

        std::vector<BResult> winners(num_blocks);
        cudaMemcpy(winners.data(), block_tsseGPU, num_blocks * sizeof(BResult),
                   cudaMemcpyDeviceToHost);

        auto best = *std::min_element(
            winners.begin(), winners.end(),
            [](const BResult &a, const BResult &b) { return a.tsse < b.tsse; });

        if (best.tsse < global_best.tsse)
            global_best = best;
    }

    delete[] pref_s;
    cudaFree(pref_sGPU);
    cudaFree(block_tsseGPU);

    return Result{
        global_best.y0,
        global_best.x0,
        global_best.y1,
        global_best.x1,
        {global_best.outer, global_best.outer, global_best.outer},
        {global_best.inner, global_best.inner, global_best.inner},
    };
}
