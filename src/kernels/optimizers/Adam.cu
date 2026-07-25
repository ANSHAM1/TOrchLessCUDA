#include "kernel.cuh"
#include "template.cuh"

#include "tensor.hpp"

#include "device_launch_parameters.h"
#include <cuda_runtime.h>
#include <stdexcept>
#include <cmath>




__global__ void adamKernel(float* Parameter, const float* Gradient, float* FirstMoment, float* SecondMoment, float LearningRate,
    float Beta1, float Beta2, float Epsilon, float BiasCorrection1, float BiasCorrection2, size_t Size) {

    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (idx >= Size)
        return;

    float grad = Gradient[idx];

    FirstMoment[idx] = Beta1 * FirstMoment[idx] + (1.0f - Beta1) * grad;
    SecondMoment[idx] = Beta2 * SecondMoment[idx] + (1.0f - Beta2) * grad * grad;

    float mHat = FirstMoment[idx] / BiasCorrection1;
    float vHat = SecondMoment[idx] / BiasCorrection2;

    Parameter[idx] -= LearningRate * mHat / (sqrtf(vHat) + Epsilon);
}




void adamUpdate(Tensor& Parameter, const Tensor& Gradient, Tensor& FirstMoment, Tensor& SecondMoment, float LearningRate,
    float Beta1, float Beta2, float Epsilon, int TimeStep) {

    if (Parameter.numel() != Gradient.numel() || Parameter.numel() != FirstMoment.numel() || Parameter.numel() != SecondMoment.numel())
        throw std::runtime_error("Adam tensor size mismatch.");

    constexpr int threads = 256;
    int blocks = (Parameter.numel() + threads - 1) / threads;

    float BiasCorrection1 = 1.0f - powf(Beta1, TimeStep);
    float BiasCorrection2 = 1.0f - powf(Beta2, TimeStep);


    ExecuteKernel("adamKernel", blocks, threads, 0, 0, adamKernel, Parameter.data(), Gradient.data(), FirstMoment.data(), SecondMoment.data(),
        LearningRate, Beta1, Beta2, Epsilon, BiasCorrection1, BiasCorrection2, Parameter.numel());
}