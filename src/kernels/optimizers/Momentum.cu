#include "kernel.cuh"
#include "template.cuh"

#include "tensor.hpp"

#include "device_launch_parameters.h"
#include <cuda_runtime.h>
#include <stdexcept>




__global__ void momentumKernel(float* Parameter, const float* Gradient, float* Velocity, float LearningRate, float Beta, size_t Size) {
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (idx >= Size)
        return;

    Velocity[idx] = Beta * Velocity[idx] - LearningRate * Gradient[idx];

    Parameter[idx] += Velocity[idx];
}




void momentumUpdate(Tensor& Parameter, const Tensor& Gradient, Tensor& Velocity, float LearningRate, float Beta) {
    if (Parameter.numel() != Gradient.numel() || Parameter.numel() != Velocity.numel())
        throw std::runtime_error("Momentum tensor size mismatch.");

    constexpr int threads = 256;
    int blocks = (Parameter.numel() + threads - 1) / threads;

    ExecuteKernel("momentumKernel", blocks, threads, 0, 0, momentumKernel, Parameter.data(), Gradient.data(), 
        Velocity.data(),  LearningRate, Beta, Parameter.numel());
}