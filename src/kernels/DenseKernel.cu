#include "kernel.cuh"
#include "template.cuh"

#include "tensor.hpp"

#include "device_launch_parameters.h"



//__global__ void denseForwardKernel(const float* input, const float* weight, const float* bias, float* output,
//    size_t Batch, size_t InFeatures, size_t OutFeatures) {
//
//    size_t out = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
//
//    size_t batch = blockIdx.y;
//    if (batch >= Batch || out >= OutFeatures)
//        return;
//
//    float sum = bias[out];
//    for (size_t i = 0; i < InFeatures; i++)
//        sum += input[batch * InFeatures + i] * weight[i * OutFeatures + out];
//    
//    output[batch * OutFeatures + out] = sum;
//}


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