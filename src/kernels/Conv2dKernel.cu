#include "kernel.cuh"
#include "template.cuh"

#include "tensor.hpp"

#include "device_launch_parameters.h"




__global__ void conv2dForwardKernel(const float* input, const float* kernel, const float* bias, float* output,
    const int N, const int C, const int H, const int W,
    const int K, const int KH, const int KW,
    const int S, const int P,
    const int OH, const int OW) {

    constexpr int TILE_DIM = 16;
    constexpr int BLOCK_ROWS = 16;


    const int PADDED_TILE_DIM = (TILE_DIM - 1) * S + KW;

    extern __shared__ float s_data[];

    float* s_input = s_data;
    float* s_kernel = &s_data[PADDED_TILE_DIM * PADDED_TILE_DIM];


    const int tile_x = blockIdx.x * TILE_DIM;
    const int tile_y = blockIdx.y * TILE_DIM;

    const int thread_x = threadIdx.x;
    const int thread_y = threadIdx.y;

    const int out_x = tile_x + thread_x;
    const int out_y = tile_y + thread_y;

    const int batch_idx = blockIdx.z / K;
    const int kernel_idx = blockIdx.z % K;


    float accumulator = 0.0f;

    for (int c = 0; c < C; ++c) {
        int input_tile_start_y = tile_y * S - P;
        int input_tile_start_x = tile_x * S - P;

        for (int i = thread_y; i < PADDED_TILE_DIM; i += BLOCK_ROWS) {
            for (int j = thread_x; j < PADDED_TILE_DIM; j += TILE_DIM) {
                int load_y = input_tile_start_y + i;
                int load_x = input_tile_start_x + j;

                if (load_y >= 0 && load_y < H && load_x >= 0 && load_x < W)
                    s_input[i * PADDED_TILE_DIM + j] = input[((batch_idx * C + c) * H + load_y) * W + load_x];      
                else
                    s_input[i * PADDED_TILE_DIM + j] = 0.0f;
            }

        }


        if (thread_y < KH && thread_x < KW)
            s_kernel[thread_y * KW + thread_x] = kernel[((kernel_idx * C + c) * KH + thread_y) * KW + thread_x];
        


        __syncthreads();



        if (out_y < OH && out_x < OW) 
            for (int kh = 0; kh < KH; ++kh)
                for (int kw = 0; kw < KW; ++kw)
                    accumulator += s_input[(thread_y * S + kh) * PADDED_TILE_DIM + (thread_x * S + kw)] * s_kernel[kh * KW + kw];

  

        __syncthreads();
    }



    if (out_y < OH && out_x < OW)
        output[((batch_idx * K + kernel_idx) * OH + out_y) * OW + out_x] = accumulator + bias[kernel_idx];
}

void conv2dForward(const Tensor& input, const Tensor& kernel, const Tensor& bias, Tensor& output, size_t stride, size_t padding) {
    const auto& I_Shape = input.shape();
    const auto& K_Shape = kernel.shape();
    const auto& O_Shape = output.shape();

    const size_t N = I_Shape[0];
    const size_t C = I_Shape[1];
    const size_t H = I_Shape[2];
    const size_t W = I_Shape[3];

    const size_t K = K_Shape[0];
    const size_t KH = K_Shape[2];
    const size_t KW = K_Shape[3];

    const size_t OH = O_Shape[2];
    const size_t OW = O_Shape[3];


    constexpr size_t TILE_DIM = 16;
    constexpr size_t BLOCK_ROWS = 16;

    const int PADDED_TILE_DIM = (TILE_DIM - 1) * stride + KW;

    size_t SharedMem = (static_cast<unsigned long long>(PADDED_TILE_DIM) * PADDED_TILE_DIM + KW * KH) * sizeof(float);

    dim3 threads(TILE_DIM, BLOCK_ROWS);
    dim3 blocks(
        (OW + TILE_DIM - 1) / TILE_DIM,
        (OH + TILE_DIM - 1) / TILE_DIM,
        N * K
    );


    ExecuteKernel("conv2dForwardKernel", blocks, threads, SharedMem, 0, conv2dForwardKernel, input.data(), kernel.data(),
        bias.data(), output.data(), N, C, H, W, K, KH, KW, stride, padding, OH, OW);
}




__global__ void conv2dInputBackwardKernel(const float* gradOutput, const float* kernel, float* gradInput, 
    const int N, const int C, const int H, const int W, const int K, const int KH, const int KW, 
    const int S, const int P, const int OH, const int OW) {

    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    size_t total = static_cast<size_t>(N) * C * H * W;
    if (idx >= total)
        return;

    int w = idx % W;
    int h = (idx / W) % H;
    int c = (idx / (static_cast<unsigned long long>(H) * W)) % C;
    int n = idx / (static_cast<unsigned long long>(C) * H * W);

    float gradient = 0.0f;

    for (int k = 0; k < K; k++)
        for (int kh = 0; kh < KH; kh++)
            for (int kw = 0; kw < KW; kw++) {
                int ohNumerator = h + P - kh;
                int owNumerator = w + P - kw;

                if (ohNumerator < 0 || owNumerator < 0)
                    continue;
                if (ohNumerator % S != 0 || owNumerator % S != 0)
                    continue;

                int oh = ohNumerator / S;
                int ow = owNumerator / S;

                if (oh < 0 || ow < 0 || oh >= OH || ow >= OW)
                    continue;

                size_t outputIndex = ((static_cast<size_t>(n) * K + k) * OH + oh) * OW + ow;
                size_t kernelIndex = ((static_cast<size_t>(k) * C + c) * KH + kh) * KW + kw;

                gradient += gradOutput[outputIndex] * kernel[kernelIndex];
            }

    gradInput[idx] = gradient;
}


void conv2dInputBackward(const Tensor& gradOutput, const Tensor& kernel, Tensor& gradInput, size_t stride, size_t padding) {
    auto& IShape = gradInput.shape();
    auto& KShape = kernel.shape();
    auto& OShape = gradOutput.shape();

    int N = IShape[0];
    int C = IShape[1];
    int H = IShape[2];
    int W = IShape[3];

    int K = KShape[0];
    int KH = KShape[2];
    int KW = KShape[3];

    int OH = OShape[2];
    int OW = OShape[3];

    size_t total = static_cast<size_t>(N) * C * H * W;


    int threads = 256;
    int blocks = (total + threads - 1) / threads;


    ExecuteKernel("conv2dInputBackwardKernel", blocks, threads, 0, 0, conv2dInputBackwardKernel, gradOutput.data(), 
        kernel.data(), gradInput.data(), N, C, H, W, K, KH, KW, stride, padding, OH, OW);
}


constexpr auto TILE_SIZE = 16;

__global__ void conv2dWeightBackwardKernel(const float* input, const float* gradOutput, float* gradKernel,
    const int N, const int C, const int H, const int W, const int K, const int KH, const int KW,
    const int S, const int P, const int OH, const int OW) {

    __shared__ float s_input[TILE_SIZE][TILE_SIZE];
    __shared__ float s_grad[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;


    int kernelElements = C * KH * KW;

    float sum = 0.0f;

    int totalPositions = N * OH * OW;

    for (int tile = 0; tile < (totalPositions + TILE_SIZE - 1) / TILE_SIZE; tile++) {
        int inputRow = tile * TILE_SIZE + threadIdx.x;

        if (inputRow < totalPositions && row < kernelElements) {
            int temp = row;
            int kw = temp % KW;
            temp /= KW;
            int kh = temp % KH;
            int c = temp / KH;
            int n = inputRow / (OH * OW);
            int pos = inputRow % (OH * OW);
            int oh = pos / OW;
            int ow = pos % OW;
            int ih = oh * S - P + kh;
            int iw = ow * S - P + kw;

            if (ih >= 0 && iw >= 0 && ih < H && iw < W)
                s_input[threadIdx.y][threadIdx.x] = input[((n * C + c) * H + ih) * W + iw];

            else
                s_input[threadIdx.y][threadIdx.x] = 0.0f;
        }
        else
            s_input[threadIdx.y][threadIdx.x] = 0.0f;

        if (inputRow < totalPositions && col < K) {
            int n = inputRow / (OH * OW);
            int pos = inputRow % (OH * OW);
            int oh = pos / OW;
            int ow = pos % OW;

            //s_grad[threadIdx.x][threadIdx.y] = gradOutput[((n * K + col) * OH + oh) * OW + ow];
            s_grad[threadIdx.y][threadIdx.x] = gradOutput[((n * K + col) * OH + oh) * OW + ow];
        }
        else
            s_grad[threadIdx.x][threadIdx.y] = 0.0f;


        __syncthreads();


        for (int i = 0; i < TILE_SIZE; i++)
            //sum += s_input[threadIdx.y][i] * s_grad[threadIdx.x][i];
            sum += s_input[threadIdx.y][i] * s_grad[threadIdx.y][i];

        __syncthreads();
    }

    if (row < kernelElements && col < K) {
        int c = row / (KH * KW);
        int rem = row % (KH * KW);
        int kh = rem / KW;
        int kw = rem % KW;

        gradKernel[((col * C + c) * KH + kh) * KW + kw] = sum;
    }
}


void conv2dWeightBackward(const Tensor& input, const Tensor& gradOutput, Tensor& gradKernel, size_t stride, size_t padding) {
    auto& IShape = input.shape();
    auto& KShape = gradKernel.shape();
    auto& OShape = gradOutput.shape();

    int N = IShape[0];
    int C = IShape[1];
    int H = IShape[2];
    int W = IShape[3];

    int K = KShape[0];
    int KH = KShape[2];
    int KW = KShape[3];

    int OH = OShape[2];
    int OW = OShape[3];

    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks(
        (K + TILE_SIZE - 1) / TILE_SIZE,
        (C * KH * KW + TILE_SIZE - 1) / TILE_SIZE
    );


    ExecuteKernel("conv2dWeightBackwardKernel", blocks, threads, 0, 0, conv2dWeightBackwardKernel, input.data(), 
        gradOutput.data(), gradKernel.data(), N, C, H, W, K, KH, KW, stride, padding, OH, OW);
}



__global__ void conv2dBiasBackwardKernel(const float* gradOutput, float* gradBias, const int N, const int K, const int OH, const int OW) {
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (idx >= K)
        return;

    float sum = 0.0f;

    for (int n = 0; n < N; n++)
        for (int h = 0; h < OH; h++) 
            for (int w = 0; w < OW; w++)
                sum += gradOutput[((static_cast<unsigned long long>(n) * K + idx) * OH + h) * OW + w];

    gradBias[idx] = sum;
}


void conv2dBiasBackward(const Tensor& gradOutput, Tensor& gradBias) {
    auto& Shape = gradOutput.shape();

    int N = Shape[0];
    int K = Shape[1];
    int OH = Shape[2];
    int OW = Shape[3];


    int threads = 256;
    int blocks = (K + threads - 1) / threads;


    ExecuteKernel("conv2dBiasBackwardKernel", blocks, threads, 0, 0, conv2dBiasBackwardKernel, gradOutput.data(), 
        gradBias.data(), N, K, OH, OW);
}