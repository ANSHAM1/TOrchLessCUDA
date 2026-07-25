#include "kernel.cuh"
#include "template.cuh"

#include "tensor.hpp"

#include "device_launch_parameters.h"




__global__ void sgdKernel(float* Parameter, const float* Gradient, float LearningRate, size_t Size) {
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (idx >= Size)
        return;

    Parameter[idx] -= LearningRate * Gradient[idx];
}




void sgdUpdate(Tensor& Parameter, const Tensor& Gradient, float LearningRate) {
    if (Parameter.numel() != Gradient.numel())
        throw std::runtime_error("Parameter and Gradient size mismatch.");

    constexpr int threads = 256;
    int blocks = (Parameter.numel() + threads - 1) / threads;

    ExecuteKernel("sgdKernel", blocks, threads, 0, 0, sgdKernel, Parameter.data(),
        Gradient.data(), LearningRate, Parameter.numel());
}