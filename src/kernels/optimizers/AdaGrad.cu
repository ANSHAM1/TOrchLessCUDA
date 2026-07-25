#include "kernel.cuh"
#include "template.cuh"

#include "tensor.hpp"

#include "device_launch_parameters.h"
#include <cuda_runtime.h>
#include <stdexcept>
#include <math.h>




__global__ void adagradKernel(float* Parameter, const float* Gradient, float* Accumulator, float LearningRate, float Epsilon, size_t Size) {
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (idx >= Size)
        return;

    float grad = Gradient[idx];

    Accumulator[idx] += grad * grad;

    Parameter[idx] -= LearningRate * grad / (sqrtf(Accumulator[idx]) + Epsilon);
}




void adagradUpdate(Tensor& Parameter, const Tensor& Gradient, Tensor& Accumulator, float LearningRate, float Epsilon) {
    if (Parameter.numel() != Gradient.numel() || Parameter.numel() != Accumulator.numel())
        throw std::runtime_error("AdaGrad tensor size mismatch.");

    constexpr int threads = 256;
    int blocks = (Parameter.numel() + threads - 1) / threads;

    ExecuteKernel("adagradKernel", blocks, threads, 0, 0, adagradKernel, Parameter.data(), Gradient.data(), Accumulator.data(),
        LearningRate, Epsilon, Parameter.numel());
}