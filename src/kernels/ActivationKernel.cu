#include "kernel.cuh"
#include "template.cuh"

#include "tensor.hpp"

#include "device_launch_parameters.h"
#include <cmath>


__global__ void reluKernel(const float* input, float* output, size_t size) {
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (idx >= size)
        return;

    output[idx] = input[idx] > 0.0f ? input[idx] : 0.0f;
}



__global__ void sigmoidKernel(const float* input, float* output, size_t size) {
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (idx >= size)
        return;

    output[idx] = 1.0f / (1.0f + expf(-input[idx]));
}



__global__ void tanhKernel(const float* input, float* output, size_t size) {
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (idx >= size)
        return;

    output[idx] = tanhf(input[idx]);
}


void activationForward(const Tensor& input, Tensor& output, const std::string& type) {
    size_t size = input.numel();

    if (size == 0)
        return;


    int threads = 256;
    int blocks = (size + threads - 1) / threads;

    if (type == " relu")
        ExecuteKernel("reluKernel", blocks, threads, 0, 0, reluKernel, input.data(), output.data(), size);
  
    else if (type == "sigmoid")
        ExecuteKernel("sigmoidKernel", blocks, threads, 0, 0, sigmoidKernel, input.data(), output.data(), size);

    else if (type == "tanh")
        ExecuteKernel("tanhKernel", blocks, threads, 0, 0, tanhKernel, input.data(), output.data(), size);

    else 
        throw std::runtime_error("Unsupported activation");
}