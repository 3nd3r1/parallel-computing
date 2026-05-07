#include <cstdlib>
#include <cuda_runtime.h>
#include <iostream>

using namespace std;

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

__global__ void normalize_kernel(float *matrix, int nx, int ny) {
    int y = blockIdx.x * blockDim.x + threadIdx.x;
    if (y >= ny)
        return;

    float sum = 0;
    for (int x = 0; x < nx; x++) {
        sum += matrix[y * nx + x];
    }
    float avg = sum / nx;

    float sum_sq = 0;
    for (int x = 0; x < nx; x++) {
        matrix[y * nx + x] -= avg;
        sum_sq += matrix[y * nx + x] * matrix[y * nx + x];
    }
    float mag = sqrt(sum_sq);
    for (int x = 0; x < nx; x++) {
        matrix[y * nx + x] /= mag;
    }
}

__global__ void pad_transpose_kernel(const float *data, float *matrix, int nx,
                                     int ny, int padded_nx, int padded_ny) {
    int ja = threadIdx.x;
    int i = blockIdx.y;

    for (int jb = 0; jb < padded_nx; jb += 64) {
        int j = ja + jb;
        float val = (i < ny && j < nx) ? data[i * nx + j] : 0;
        matrix[j * padded_ny + i] = val;
    }
}

__global__ void correlate_kernel(float *result, float *matrix, int nx, int ny,
                                 int padded_nx, int padded_ny) {
    int ia = threadIdx.x;
    int ja = threadIdx.y;
    int ic = blockIdx.x;
    int jc = blockIdx.y;

    float val[8][8];
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            val[i][j] = 0;
        }
    }

    for (int k = 0; k < nx; k++) {
        float x[8];
        float y[8];
        for (int ib = 0; ib < 8; ib++) {
            int i = ic * 64 + ib * 8 + ia;
            x[ib] = matrix[padded_ny * k + i];
        }
        for (int jb = 0; jb < 8; jb++) {
            int j = jc * 64 + jb * 8 + ja;
            y[jb] = matrix[padded_ny * k + j];
        }
        for (int ib = 0; ib < 8; ib++) {
            for (int jb = 0; jb < 8; jb++) {
                val[ib][jb] += x[ib] * y[jb];
            }
        }
    }
    for (int ib = 0; ib < 8; ++ib) {
        for (int jb = 0; jb < 8; ++jb) {
            int i = ic * 64 + ib * 8 + ia;
            int j = jc * 64 + jb * 8 + ja;
            if (i < ny && j < ny) {
                result[ny * i + j] = val[ib][jb];
            }
        }
    }
}

/*
This is the function you need to implement. Quick reference:
- input rows: 0 <= y < ny
- input columns: 0 <= x < nx
- element at row y and column x is stored in data[x + y*nx]
- correlation between rows i and row j has to be stored in result[i + j*ny]
- only parts with 0 <= j <= i < ny need to be filled
*/

void correlate(int ny, int nx, const float *data, float *result) {
    int padded_ny = roundup(ny, 64);
    int padded_nx = roundup(nx, 64);

    float *dataGPU = NULL;
    CHECK(cudaMalloc((void **)&dataGPU, ny * nx * sizeof(float)));
    CHECK(cudaMemcpy(dataGPU, data, ny * nx * sizeof(float),
                     cudaMemcpyHostToDevice));

    float *matrixGPU = NULL;
    CHECK(
        cudaMalloc((void **)&matrixGPU, padded_ny * padded_nx * sizeof(float)));
    CHECK(cudaMemset(matrixGPU, 0, padded_ny * padded_nx * sizeof(float)));

    float *resultGPU = NULL;
    CHECK(cudaMalloc((void **)&resultGPU, ny * ny * sizeof(float)));
    CHECK(cudaMemset(resultGPU, 0, ny * ny * sizeof(float)));

    {
        normalize_kernel<<<divup(ny, 256), 256>>>(dataGPU, nx, ny);
        CHECK(cudaGetLastError());
    }

    {
        dim3 dimBlock(64, 1);
        dim3 dimGrid(1, padded_ny);
        pad_transpose_kernel<<<dimGrid, dimBlock>>>(dataGPU, matrixGPU, nx, ny,
                                                    padded_nx, padded_ny);
        CHECK(cudaGetLastError());
    }

    {
        dim3 dimBlock(8, 8);
        dim3 dimGrid(padded_ny / 64, padded_ny / 64);
        correlate_kernel<<<dimGrid, dimBlock>>>(resultGPU, matrixGPU, nx, ny,
                                                padded_nx, padded_ny);
        CHECK(cudaGetLastError());
    }

    CHECK(cudaMemcpy(result, resultGPU, ny * ny * sizeof(float),
                     cudaMemcpyDeviceToHost));

    CHECK(cudaFree(dataGPU));
    CHECK(cudaFree(matrixGPU));
    CHECK(cudaFree(resultGPU));
}
