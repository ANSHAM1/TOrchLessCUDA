#include "kernel.cuh"
#include "template.cuh"

#include "tensor.hpp"

#include "device_launch_parameters.h"
#include <cuda_runtime.h>
#include <stdexcept>
#include <math.h>




__global__ void rmsPropKernel(float* Parameter, const float* Gradient, float* MeanSquare, float LearningRate, 
    float Beta, float Epsilon, size_t Size) {

    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (idx >= Size)
        return;

    float grad = Gradient[idx];

    MeanSquare[idx] = Beta * MeanSquare[idx] + (1.0f - Beta) * grad * grad;

    Parameter[idx] -= LearningRate * grad / (sqrtf(MeanSquare[idx]) + Epsilon);
}




void rmsPropUpdate(Tensor& Parameter, const Tensor& Gradient, Tensor& MeanSquare, float LearningRate, float Beta, float Epsilon) {
    if (Parameter.numel() != Gradient.numel() || Parameter.numel() != MeanSquare.numel())
        throw std::runtime_error("RMSProp tensor size mismatch.");

    constexpr int threads = 256;

    int blocks = (Parameter.numel() + threads - 1) / threads;

    ExecuteKernel("rmsPropKernel", blocks, threads, 0, 0, rmsPropKernel, Parameter.data(), Gradient.data(), MeanSquare.data(),
        LearningRate, Beta, Epsilon, Parameter.numel());
}