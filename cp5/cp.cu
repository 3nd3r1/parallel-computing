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

// static inline int roundup(int a, int b) { return divup(a, b) * b; }

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

#define TILE 16

__global__ void correlate_kernel(float *result, float *matrix, int nx, int ny) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    __shared__ float tile_i[TILE][TILE];
    __shared__ float tile_j[TILE][TILE];

    float val = 0;
    for (int t = 0; t < nx; t += TILE) {
        if (i < ny && t + threadIdx.y < nx) {
            tile_i[threadIdx.x][threadIdx.y] = matrix[i * nx + t + threadIdx.y];
        } else {
            tile_i[threadIdx.x][threadIdx.y] = 0;
        }

        if (j < ny && t + threadIdx.x < nx) {
            tile_j[threadIdx.y][threadIdx.x] = matrix[j * nx + t + threadIdx.x];
        } else {
            tile_j[threadIdx.y][threadIdx.x] = 0;
        }

        __syncthreads();

        for (int k = 0; k < TILE; k++) {
            val += tile_i[threadIdx.x][k] * tile_j[threadIdx.y][k];
        }

        __syncthreads();
    }

    if (i < ny && j < ny && i >= j)
        result[i + j * ny] = val;
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
    float *matrixGPU = NULL;
    CHECK(cudaMalloc((void **)&matrixGPU, ny * nx * sizeof(float)));
    CHECK(cudaMemcpy(matrixGPU, data, ny * nx * sizeof(float),
                     cudaMemcpyHostToDevice));

    normalize_kernel<<<divup(ny, 256), 256>>>(matrixGPU, nx, ny);

    float *resultGPU = NULL;
    CHECK(cudaMalloc((void **)&resultGPU, ny * ny * sizeof(float)));
    CHECK(cudaMemset(resultGPU, 0, ny * ny * sizeof(float)));

    dim3 dimBlock(16, 16);
    dim3 dimGrid(divup(ny, dimBlock.x), divup(ny, dimBlock.y));
    correlate_kernel<<<dimGrid, dimBlock>>>(resultGPU, matrixGPU, nx, ny);

    CHECK(cudaGetLastError());
    CHECK(cudaMemcpy(result, resultGPU, ny * ny * sizeof(float),
                     cudaMemcpyDeviceToHost));
    CHECK(cudaFree(matrixGPU));
    CHECK(cudaFree(resultGPU));
}
