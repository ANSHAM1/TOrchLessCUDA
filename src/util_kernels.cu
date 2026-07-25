#include "template.cuh"

#include "device_launch_parameters.h"
#include <curand_kernel.h>




__global__ void tensorAssignKernel(float* data, float value, size_t size) {
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (idx < size) data[idx] = value;
}




void tensorAssign(float* ptr, size_t value, size_t n) {
    dim3 blocks((n + 255) / 256);
    dim3 threads(256);

    ExecuteKernel("tensorAssignKernel", blocks, threads, 0, 0, tensorAssignKernel, ptr, value, n);
}




__global__ void uniformDistKernel(float* data, size_t size, unsigned long long seed, float min, float max) {
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (idx >= size)
        return;

    curandStatePhilox4_32_10_t state;

    curand_init(seed, idx, 0, &state );

    float random = curand_uniform(&state);

    data[idx] = min + random * (max - min);
}




void uniformDist(float* data, size_t size, unsigned long long seed, float min, float max) {
    if (size == 0)
        return;

    int threads = 256;
    int blocks = (size + threads - 1) / threads;

    ExecuteKernel("uniformDistKernel", blocks, threads, 0, 0, uniformDistKernel, data, size, seed, min, max);
}