#include "kernel.cuh"
#include "template.cuh"

#include "tensor.hpp"

#include "device_launch_parameters.h"




constexpr int TILE_SIZE = 16;


__global__ void denseForwardKernel(const float* input, const float* weight, const float* bias, float* output,
    size_t Batch, size_t InFeatures, size_t OutFeatures) {

    __shared__ float sInput[TILE_SIZE][TILE_SIZE];
    __shared__ float sWeight[TILE_SIZE][TILE_SIZE];

    const size_t out = static_cast<size_t>(blockIdx.x) * TILE_SIZE + threadIdx.x;
    const size_t batch = static_cast<size_t>(blockIdx.y) * TILE_SIZE + threadIdx.y;

    float sum = 0.0f;

    const size_t NumTiles = (InFeatures + TILE_SIZE - 1) / TILE_SIZE;
    for (size_t tile = 0; tile < NumTiles; tile++) {
        const size_t inCol = tile * TILE_SIZE + threadIdx.x;

        if (batch < Batch && inCol < InFeatures)
            sInput[threadIdx.y][threadIdx.x] = input[batch * InFeatures + inCol];
        else
            sInput[threadIdx.y][threadIdx.x] = 0.0f;


        const size_t weightRow = tile * TILE_SIZE + threadIdx.y;


        if (weightRow < InFeatures && out < OutFeatures)
            sWeight[threadIdx.y][threadIdx.x] = weight[weightRow * OutFeatures + out];
        else
            sWeight[threadIdx.y][threadIdx.x] = 0.0f;


        __syncthreads();


        for (int k = 0; k < TILE_SIZE; k++)
            sum += sInput[threadIdx.y][k] * sWeight[k][threadIdx.x];


        __syncthreads();
    }

    if (batch < Batch && out < OutFeatures)
        output[batch * OutFeatures + out] = sum + bias[out];
}




void denseForward(const Tensor& input, const Tensor& weight, const Tensor& bias, Tensor& output) {
    const auto& inputShape = input.shape();

    const size_t Batch = inputShape[0];
    const size_t InFeatures = inputShape[1];
    const size_t OutFeatures = output.shape()[1];

    dim3 threads(TILE_SIZE, TILE_SIZE);

    dim3 blocks(
        static_cast<unsigned int>((OutFeatures + TILE_SIZE - 1) / TILE_SIZE),
        static_cast<unsigned int>((Batch + TILE_SIZE - 1) / TILE_SIZE)
    );

    ExecuteKernel("denseForwardKernel", blocks, threads, 0, 0, denseForwardKernel, input.data(), weight.data(),
        bias.data(), output.data(), Batch, InFeatures, OutFeatures);
}




__global__ void denseInputBackwardKernel(const float* gradOutput, const float* weight, float* gradInput, 
    size_t Batch, size_t InFeatures, size_t OutFeatures) {

    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    size_t total = Batch * InFeatures;
    if (idx >= total)
        return;

    size_t batch = idx / InFeatures;
    size_t in = idx % InFeatures;


    float sum = 0.0f;
    for (size_t out = 0; out < OutFeatures; out++)
        sum += gradOutput[batch * OutFeatures + out] * weight[in * OutFeatures + out];

    gradInput[idx] = sum;
}




__global__ void denseWeightBackwardKernel(const float* input, const float* gradOutput, float* gradWeight, 
    size_t Batch, size_t InFeatures, size_t OutFeatures) {

    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    size_t total = InFeatures * OutFeatures;
    if (idx >= total)
        return;

    size_t in = idx / OutFeatures;
    size_t out = idx % OutFeatures;


    float sum = 0.0f;
    for (size_t b = 0; b < Batch; b++)
        sum += input[b * InFeatures + in] * gradOutput[b * OutFeatures + out];

    gradWeight[idx] = sum;
}




__global__ void denseBiasBackwardKernel(const float* gradOutput, float* gradBias, size_t Batch, size_t OutFeatures) {
    size_t out = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (out >= OutFeatures)
        return;

    float sum = 0.0f;
    for (size_t b = 0; b < Batch; b++) 
        sum += gradOutput[b * OutFeatures + out];

    gradBias[out] = sum;
}




void denseInputBackward(const Tensor& gradOutput, const Tensor& weight, Tensor& gradInput) {
    size_t Batch = gradOutput.shape()[0];
    size_t Out = gradOutput.shape()[1];
    size_t In = weight.shape()[0];

    size_t total = Batch * In;


    int threads = 256;
    int blocks = (total + threads - 1) / threads;


    ExecuteKernel("denseInputBackwardKernel", blocks, threads, 0, 0, denseInputBackwardKernel, gradOutput.data(), weight.data(),
        gradInput.data(), Batch, In, Out);
}




void denseWeightBackward(const Tensor& input, const Tensor& gradOutput, Tensor& gradWeight) {
    size_t Batch = input.shape()[0];
    size_t In = input.shape()[1];
    size_t Out = gradOutput.shape()[1];

    size_t total = In * Out;


    int threads = 256;
    int blocks = (total + threads - 1) / threads;


    ExecuteKernel("denseWeightBackwardKernel", blocks, threads, 0, 0, denseWeightBackwardKernel, input.data(),
        gradOutput.data(), gradWeight.data(), Batch, In, Out);
}




void denseBiasBackward(const Tensor& gradOutput, Tensor& gradBias) {
    size_t Batch = gradOutput.shape()[0];
    size_t Out = gradOutput.shape()[1];


    int threads = 256;
    int blocks = (Out + threads - 1) / threads;


    ExecuteKernel("denseBiasBackwardKernel", blocks, threads, 0, 0, denseBiasBackwardKernel, gradOutput.data(),
        gradBias.data(), Batch, Out);
}