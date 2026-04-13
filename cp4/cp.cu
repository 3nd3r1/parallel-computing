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

__global__ void mykernel(float *result, double *matrix, int nx, int ny) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i >= ny || j >= ny || i < j)
        return;
    double val = 0;
    for (int k = 0; k < nx; k++) {
        val += matrix[i * nx + k] * matrix[j * nx + k];
    }
    result[i + j * ny] = (float)val;
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
    double *matrix = new double[ny * nx];

    for (int y = 0; y < ny; y++) {
        double avg = 0;
        for (int x = 0; x < nx; x++) {
            matrix[y * nx + x] = data[x + y * nx];
            avg += matrix[y * nx + x];
        }
        avg /= nx;
        double mag = 0;
        for (int x = 0; x < nx; x++) {
            matrix[y * nx + x] -= avg;
            mag += matrix[y * nx + x] * matrix[y * nx + x];
        }
        mag = std::sqrt(mag);
        for (int x = 0; x < nx; x++) {
            matrix[y * nx + x] /= mag;
        }
    }

    double *matrixGPU = NULL;
    CHECK(cudaMalloc((void **)&matrixGPU, ny * nx * sizeof(double)));
    float *resultGPU = NULL;
    CHECK(cudaMalloc((void **)&resultGPU, ny * ny * sizeof(float)));
    CHECK(cudaMemcpy(matrixGPU, matrix, ny * nx * sizeof(double),
                     cudaMemcpyHostToDevice));

    dim3 dimBlock(16, 16);
    dim3 dimGrid(divup(nx, dimBlock.x), divup(ny, dimBlock.y));
    mykernel<<<dimGrid, dimBlock>>>(resultGPU, matrixGPU, nx, ny);
    CHECK(cudaGetLastError());

    CHECK(cudaMemcpy(result, resultGPU, ny * ny * sizeof(float),
                     cudaMemcpyDeviceToHost));
    CHECK(cudaFree(matrixGPU));
    CHECK(cudaFree(resultGPU));
}
