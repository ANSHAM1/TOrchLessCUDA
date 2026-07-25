#include "kernel.cuh"
#include "template.cuh"

#include "tensor.hpp"

#include "device_launch_parameters.h"
#include <cuda_runtime.h>

#include <cmath>
#include <cstring>
#include <stdexcept>




__global__ void sigmoidOutKernel(const float* input, float* output, size_t size) {
    size_t idx =
        static_cast<size_t>(blockIdx.x) * blockDim.x +
        threadIdx.x;

    if (idx >= size)
        return;

    output[idx] = 1.0f / (1.0f + expf(-input[idx]));
}




__global__ void softmaxKernel(const float* input, float* output, size_t Batch, size_t Classes) {
    size_t batch = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (batch >= Batch)
        return;

    size_t baseIdx = batch * Classes;

    // Step 1: Find maximum value (numerical stability)
    float maxValue = input[baseIdx];
    for (size_t i = 1; i < Classes; i++)
        maxValue = fmaxf(maxValue, input[baseIdx + i]);

    // Step 2: Compute exponentials and their sum
    float sumExp = 0.0f;
    for (size_t i = 0; i < Classes; i++) {
        float value = expf(input[baseIdx + i] - maxValue);

        output[baseIdx + i] = value;
        sumExp += value;
    }

    // Step 3: Normalize
    for (size_t i = 0; i < Classes; i++)
        output[baseIdx + i] /= sumExp;
}




void outputForward(const Tensor& input, Tensor& output, const std::string& type) {
    if (type == "Sigmoid") {
        constexpr int Threads = 256;
        int Blocks = static_cast<int>((input.numel() + Threads - 1) / Threads);

        ExecuteKernel("sigmoidKernel", Blocks, Threads, 0, 0, sigmoidOutKernel, input.data(), output.data(), input.numel());
        return;
    }

    if (type == "Softmax") {
        const auto& shape = input.shape();

        if (shape.size() != 2)
            throw std::runtime_error("Softmax expects a 2D tensor.");

        size_t Batch = shape[0];
        size_t Classes = shape[1];

        constexpr int Threads = 256;
        int Blocks = static_cast<int>((Batch + Threads - 1) / Threads);

        ExecuteKernel("softmaxKernel", Blocks, Threads, 0, 0, softmaxKernel, input.data(), output.data(), Batch, Classes);
        return;
    }

    throw std::runtime_error("Unsupported output layer type.");
}