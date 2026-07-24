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

    if (type == "relu")
        ExecuteKernel("reluKernel", blocks, threads, 0, 0, reluKernel, input.data(), output.data(), size);
  
    else if (type == "sigmoid")
        ExecuteKernel("sigmoidKernel", blocks, threads, 0, 0, sigmoidKernel, input.data(), output.data(), size);

    else if (type == "tanh")
        ExecuteKernel("tanhKernel", blocks, threads, 0, 0, tanhKernel, input.data(), output.data(), size);

    else 
        throw std::runtime_error("Unsupported activation");
}





// ============================================================
// ReLU Backward
//
// gradInput = gradOutput * (input > 0)
// ============================================================

__global__ void reluBackwardKernel(const float* input, const float* gradOutput, float* gradInput, size_t size) {
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (idx >= size)
        return;

    gradInput[idx] = (input[idx] > 0.0f) ? gradOutput[idx] : 0.0f;
}


static void reluBackward(const Tensor& input, const Tensor& gradOutput, Tensor& gradInput) {
    size_t size = input.numel();

    int threads = 256;
    int blocks = (size + threads - 1) / threads;

    ExecuteKernel("reluBackwardKernel", blocks, threads, 0, 0, reluBackwardKernel, input.data(), 
        gradOutput.data(), gradInput.data(), size);
}


// ============================================================
// Sigmoid Backward
//
// sigmoid derivative:
//
// y * (1-y)
//
// gradInput = gradOutput * output * (1-output)
//
// Here output is sigmoid output
// ============================================================

__global__ void sigmoidBackwardKernel(const float* output, const float* gradOutput, float* gradInput, size_t size) {
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (idx >= size)
        return;

    float y = output[idx];

    gradInput[idx] = gradOutput[idx] * y * (1.0f - y);
}


static void sigmoidBackward(const Tensor& output, const Tensor& gradOutput, Tensor& gradInput) {
    size_t size = output.numel();


    int threads = 256;
    int blocks = (size + threads - 1) / threads;


    ExecuteKernel("sigmoidBackwardKernel", blocks, threads, 0, 0, sigmoidBackwardKernel,
        output.data(), gradOutput.data(), gradInput.data(), size);
}


// ============================================================
// Tanh Backward
//
// derivative:
//
// 1 - y^2
//
// gradInput = gradOutput * (1-y*y)
//
// ============================================================

__global__ void tanhBackwardKernel(const float* output, const float* gradOutput, float* gradInput, size_t size) {
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (idx >= size)
        return;

    float y = output[idx];

    gradInput[idx] = gradOutput[idx] * (1.0f - y * y);
}


static void tanhBackward(const Tensor& output, const Tensor& gradOutput, Tensor& gradInput) {
    size_t size = output.numel();


    int threads = 256;
    int blocks = (size + threads - 1) / threads;


    ExecuteKernel("tanhBackwardKernel", blocks, threads, 0, 0, tanhBackwardKernel, output.data(),
        gradOutput.data(), gradInput.data(), size);
}


void activationBackward(const Tensor& input, const Tensor& output, const Tensor& gradOutput, Tensor& gradInput, const std::string& Type) {
    if (Type == "relu")
        reluBackward(input, gradOutput, gradInput);

    else if (Type == "sigmoid")
        sigmoidBackward(output, gradOutput, gradInput);

    else if (Type == "tanh")
        tanhBackward(output, gradOutput, gradInput);

    else
        throw std::runtime_error("Unsupported activation backward");
}