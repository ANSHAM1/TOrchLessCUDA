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

void conv2dForward(const Tensor& input, const Tensor& kernel, const Tensor& bias, Tensor& output,
	size_t stride, size_t padding) {

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