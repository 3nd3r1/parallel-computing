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

#define CHECK(x) check(x, #x)

__global__ void kernel(int *pref_s, BResult *block_results, int nx, int ny,
                       long long total) {
    int nxp = nx + 1;

    long long gid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long step = (long long)blockDim.x * gridDim.x;

    BResult best_result = {0, 0, 0, 0, 0, 0, INFINITY};

    for (long long i = gid; i < total; i += step) {
        long long idx = i;
        int x0 = idx % nx;
        idx /= nx;
        int y0 = idx % ny;
        idx /= ny;
        int x1 = idx % (nx + 1);
        idx /= (nx + 1);
        int y1 = (int)idx;

        if (x1 <= x0 || y1 <= y0)
            continue;

        int in_n = (y1 - y0) * (x1 - x0);
        int out_n = nx * ny - in_n;
        if (out_n <= 0)
            continue;

        float inv_i = 1.0f / in_n;
        float inv_o = 1.0f / out_n;

        float inside_sum =
            (float)(pref_s[x1 + nxp * y1] - pref_s[x1 + nxp * y0] -
                    pref_s[x0 + nxp * y1] + pref_s[x0 + nxp * y0]);
        float outside_sum = (float)(pref_s[nx + nxp * ny]) - inside_sum;

        float inside_sse = inside_sum * (1.0f - inside_sum * inv_i);
        float outside_sse = outside_sum * (1.0f - outside_sum * inv_o);
        float tsse = inside_sse + outside_sse;

        if (tsse < best_result.tsse)
            best_result = {
                y0, x0, y1, x1, outside_sum * inv_o, inside_sum * inv_i, tsse};
    }

    __shared__ BResult sdata[256];
    int tid = threadIdx.x;
    sdata[tid] = best_result;
    __syncthreads();

    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s && sdata[tid + s].tsse < sdata[tid].tsse)
            sdata[tid] = sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0)
        block_results[blockIdx.x] = sdata[0];
}

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

    long long total = (long long)nx * ny * (nx + 1) * (ny + 1);
    int blocks = 4096;
    int threads = 256;

    int *pref_sGPU = NULL;
    CHECK(cudaMalloc(&pref_sGPU, nxp * nyp * sizeof(int)));
    CHECK(cudaMemcpy(pref_sGPU, pref_s, nxp * nyp * sizeof(int),
                     cudaMemcpyHostToDevice));

    BResult *block_resultsGPU = NULL;
    CHECK(cudaMalloc(&block_resultsGPU, blocks * sizeof(BResult)));

    {
        kernel<<<blocks, threads>>>(pref_sGPU, block_resultsGPU, nx, ny, total);
        CHECK(cudaGetLastError());
    }

    std::vector<BResult> winners(blocks);
    CHECK(cudaMemcpy(winners.data(), block_resultsGPU, blocks * sizeof(BResult),
                     cudaMemcpyDeviceToHost));

    auto best = *std::min_element(
        winners.begin(), winners.end(),
        [](const BResult &a, const BResult &b) { return a.tsse < b.tsse; });

    cudaFree(pref_sGPU);
    cudaFree(block_resultsGPU);

    return Result{
        best.y0,
        best.x0,
        best.y1,
        best.x1,
        {best.outer, best.outer, best.outer},
        {best.inner, best.inner, best.inner},
    };
}
